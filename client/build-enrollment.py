#!/usr/bin/env python3
"""Cut an enrollment sample for one speaker out of a meeting they were already
identified in, using the diarizer's own cluster spans.

The report's `embed_segments` are the segments the diarizer chose to build that
cluster's embedding from -- longest first, minimum duration enforced, degenerate
ones dropped -- so they are already that speaker's best audio in the recording.
This pulls them back out as a single wav.

Enrolling from the setup someone is actually recorded on beats enrolling from a
clean reference by a wide margin, so a recent meeting is better source material
than a purpose-made recording on other hardware.

Nothing is enrolled unless --enroll is given, and the extract is always
verified single-speaker first: a sample containing two voices is worse than no
sample at all.

A speaker whose turns are short can run out of material: the report exposes at
most MAX_EMBED_SEGMENTS spans per cluster, so someone averaging five seconds a
turn tops out around 100s from one meeting however much they talked. Passing a
second --from pools the spans across both and still takes the longest first, so
the extra audio comes from the best of either meeting rather than padding the
first one with its own leftovers.

Usage:
  build-enrollment.py "Speaker Name" --from <audio> [--from <audio2>]
                      [--out DIR] [--enroll [NAME]]
                      [--threshold 0.45] [--max-seconds 120]
"""

import json
import os
import subprocess
import sys
import tempfile
import urllib.request

HOST = os.environ.get("DIARIZER_HOST", "192.168.1.25")
PORT = os.environ.get("DIARIZER_PORT", "10301")
BASE = f"http://{HOST}:{PORT}"

INSET = 0.10          # trimmed from each end of a span, so an adjacent
                      # speaker's first syllable cannot bleed in
MIN_SPAN = 1.0        # spans shorter than this are not worth the seams
MIN_TOTAL = 45.0      # refuse to enroll from less audio than this

# Per-clip verification. Each selected clip is identified on its own and has to
# come back as the target; clips that do not are dropped.
#
# The earlier check ran the finished concatenation back through the diarizer and
# looked at cluster shares. That is circular: the same clustering that decided
# which segments belonged to this speaker gets to re-judge its own work, so a
# cluster that had merged two people reports one clean speaker on the way back
# out. It rated a sample 98.4% pure that turned out to be roughly half somebody
# else, and passed two others whose interjections were obvious by ear.
#
# Identifying each clip separately removes the clustering step, so a clip
# belonging to another person is judged on its own merits and shows up.
CLIP_MIN_SCORE = 0.40   # a clip scoring below this for anyone is inconclusive
CLIP_MAX_REJECT_PCT = 15.0   # give up if this much of the audio is rejected


def post(path, fields, files=None):
    b = "----MDBoundary"
    parts = []
    for k, v in fields.items():
        parts += [f"--{b}\r\n".encode(),
                  f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode(),
                  str(v).encode(), b"\r\n"]
    for k, path_ in (files or {}).items():
        parts += [f"--{b}\r\n".encode(),
                  (f'Content-Disposition: form-data; name="{k}"; '
                   f'filename="{os.path.basename(path_)}"\r\n').encode(),
                  b"Content-Type: application/octet-stream\r\n\r\n",
                  open(path_, "rb").read(), b"\r\n"]
    parts.append(f"--{b}--\r\n".encode())
    req = urllib.request.Request(
        BASE + path, data=b"".join(parts),
        headers={"Content-Type": f"multipart/form-data; boundary={b}"},
        method="POST")
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())


def judge_clip(src, a, b, target, threshold, workdir, tag):
    """Identify one clip on its own. Returns (ok, note, score).

    Judged in isolation, so there is no clustering step that could absorb a
    second voice into the target's cluster and hide it.
    """
    clip = os.path.join(workdir, f"{tag}.wav")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.3f}", "-to", f"{b:.3f}",
         "-i", src, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", clip],
        check=True)
    try:
        rep = post("/identify", {"threshold": threshold}, {"audio": clip})
    except Exception as e:
        return False, f"error: {e}", 0.0
    finally:
        if os.path.exists(clip):
            os.remove(clip)

    clusters = rep.get("clusters") or []
    if not clusters:
        return False, "no speech", 0.0
    if len(clusters) > 1:
        # two voices in one clip split into two clusters even at this length
        return False, f"{len(clusters)} voices", 0.0
    scores = clusters[0].get("scores") or []
    if not scores:
        return False, "no score", 0.0

    top = scores[0]
    if top["name"] != target:
        return False, f"sounds like {top['name']}", top["score"]
    if top["score"] < CLIP_MIN_SCORE:
        return False, "too weak to confirm", top["score"]
    return True, "", top["score"]


def select_verified(pool, target, threshold, want, workdir):
    """Walk the pooled spans longest-first, verifying each before accepting it,
    until `want` seconds have been gathered.

    Screening as we go means a rejected clip is replaced by the next-longest
    candidate rather than simply lost, so contamination costs quality only if
    it runs out the supply.
    """
    kept, rejected, scores, acc = [], [], [], 0.0
    for i, (src, a, b) in enumerate(pool, 1):
        if acc >= want:
            break
        ok, note, score = judge_clip(src, a, b, target, threshold, workdir,
                                     f"chk{i:03d}")
        if ok:
            kept.append((src, a, b))
            scores.append(score)
            acc += b - a
            print(f"    {b - a:>5.1f}s  ok    {score:.3f}   "
                  f"[{acc:.0f}s of {want:.0f}s]")
        else:
            rejected.append((b - a, note, score))
            print(f"    {b - a:>5.1f}s  DROP  {note}"
                  + (f" {score:.3f}" if score else ""))
    return kept, rejected, scores, acc


def extract(spans, dest):
    """Cut each (src, start, end) out and concatenate. Done as separate files
    rather than one aselect filter, which silently passes the whole recording
    through."""
    with tempfile.TemporaryDirectory() as tmp:
        listfile = os.path.join(tmp, "list.txt")
        with open(listfile, "w", encoding="utf8") as lf:
            for i, (src, a, b) in enumerate(spans):
                part = os.path.join(tmp, f"{i:03d}.wav")
                subprocess.run(
                    ["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.3f}",
                     "-to", f"{b:.3f}", "-i", src, "-ac", "1", "-ar", "16000",
                     "-c:a", "pcm_s16le", part], check=True)
                lf.write(f"file '{part}'\n")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
             "-i", listfile, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
             dest], check=True)


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    target = args[0]
    sources = []
    outdir = r"M:\media\meetings\enrollments"
    threshold, max_seconds = "0.45", 120.0
    enroll_as = None
    do_enroll = False

    i = 1
    while i < len(args):
        if args[i] == "--from":
            sources.append(args[i + 1]); i += 2
        elif args[i] == "--out":
            outdir = args[i + 1]; i += 2
        elif args[i] == "--threshold":
            threshold = args[i + 1]; i += 2
        elif args[i] == "--max-seconds":
            max_seconds = float(args[i + 1]); i += 2
        elif args[i] == "--enroll":
            do_enroll = True
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                enroll_as = args[i + 1]; i += 2
            else:
                i += 1
        else:
            i += 1

    if not sources:
        print("no --from given")
        sys.exit(1)

    pool, stamps = [], []
    for src in sources:
        print(f"Analysing {os.path.basename(src)} ...")
        report = post("/identify", {"threshold": threshold}, {"audio": src})
        cluster = next((c for c in report["clusters"]
                        if c.get("matched") == target), None)
        if cluster is None:
            found = ", ".join(sorted({c["cluster"] for c in report["clusters"]}))
            print(f"  {target!r} not identified here. Found: {found}")
            continue
        score = cluster["scores"][0]["score"]
        spans = [(src, a + INSET, b - INSET) for a, b in cluster["embed_segments"]]
        spans = [s for s in spans if s[2] - s[1] >= MIN_SPAN]
        pool.extend(spans)
        stamps.append(os.path.basename(src)[:10])
        print(f"  matched at {score:.4f} over {cluster['duration_sec']:.0f}s; "
              f"{len(spans)} usable span(s), "
              f"{sum(b - a for _, a, b in spans):.1f}s")

    if not pool:
        print(f"\n{target!r} was not identified in any source given.")
        sys.exit(2)

    # longest first across every source, so the best audio is tried first and
    # anything rejected is replaced by the next-best rather than simply lost
    pool.sort(key=lambda s: s[2] - s[1], reverse=True)

    print(f"\nVerifying each clip on its own ({len(pool)} candidate span(s)) ...")
    with tempfile.TemporaryDirectory() as work:
        chosen, rejected, clip_scores, acc = select_verified(
            pool, target, threshold, max_seconds, work)

    dropped = sum(d for d, _, _ in rejected)
    total_seen = acc + dropped
    pct = (dropped / total_seen * 100) if total_seen else 0.0
    print(f"\nkept {len(chosen)} clip(s), {acc:.1f}s; "
          f"dropped {len(rejected)}, {dropped:.1f}s ({pct:.0f}% of what was checked)")
    if rejected:
        reasons = {}
        for d, note, _ in rejected:
            reasons[note] = reasons.get(note, 0) + 1
        for note, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"    {n:>2}x  {note}")

    if not chosen:
        print(f"\nNothing survived verification -- this cluster is not {target}.")
        sys.exit(4)
    if acc < MIN_TOTAL:
        print(f"\nOnly {acc:.1f}s verified as {target} -- under the "
              f"{MIN_TOTAL:.0f}s minimum. Try a meeting where they speak more.")
        sys.exit(3)
    # Two different failures hide behind a rejection, and only one is alarming.
    # A clip dropped for holding two voices is ordinary crosstalk -- longer
    # spans catch more of it, so the best material fails most often, and
    # dropping it is the tool working. A clip that identifies as somebody else
    # means the source cluster itself is mixed, which the kept clips may share.
    wrong = sum(d for d, note, _ in rejected if note.startswith("sounds like"))
    wrong_pct = (wrong / total_seen * 100) if total_seen else 0.0
    if wrong_pct > CLIP_MAX_REJECT_PCT:
        print(f"\n{wrong_pct:.0f}% of the checked audio identified as somebody "
              f"else. The source cluster is mixed, and the kept clips may be "
              f"too -- a clip has to be wrong enough to fail on its own. "
              f"Treat this sample with suspicion.")

    kept_scores = clip_scores
    if kept_scores:
        lo, hi = min(kept_scores), max(kept_scores)
        print(f"\nkept clips scored {lo:.2f} to {hi:.2f} for {target}")
        if hi < 0.70:
            print(f"  every clip is weak. Either the clips are too short to "
                  f"judge, or this cluster is not reliably {target}. Listen "
                  f"before enrolling.")

    # play back grouped by source, chronological within each
    chosen.sort(key=lambda s: (sources.index(s[0]), s[1]))
    used = [s for s in sources if any(c[0] == s for c in chosen)]

    stamp = "+".join(sorted({os.path.basename(s)[:10] for s in used}))
    slug = target.replace(" ", "-").replace(",", "")
    os.makedirs(outdir, exist_ok=True)
    dest = os.path.join(outdir, f"{slug}_voice-enrollment_{stamp}.wav")
    extract(chosen, dest)
    print(f"\nwrote {dest} ({acc:.1f}s from {len(used)} source(s))")
    if not do_enroll:
        print(f"\nNot enrolled (no --enroll). Sample kept at:\n  {dest}")
        return

    name = enroll_as or target
    res = post("/enroll", {"name": name}, {"audio": dest})
    print(f"\nEnrolled as {res['name']!r}.")
    if name != target:
        print(f"Compare it against {target!r} on a real meeting before promoting.")


if __name__ == "__main__":
    main()
