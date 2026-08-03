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
VERIFY_SHARE = 90.0   # the extract must be this % one speaker to pass


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

    # longest first across every source, so the extra seconds come from the
    # best available audio rather than padding one meeting with its leftovers
    pool.sort(key=lambda s: s[2] - s[1], reverse=True)
    chosen, acc = [], 0.0
    for s in pool:
        if acc >= max_seconds:
            break
        chosen.append(s)
        acc += s[2] - s[1]
    # play back grouped by source, chronological within each
    chosen.sort(key=lambda s: (sources.index(s[0]), s[1]))
    used = [s for s in sources if any(c[0] == s for c in chosen)]

    print(f"\nselected {len(chosen)} span(s) totalling {acc:.1f}s "
          f"from {len(used)} source(s)")

    if acc < MIN_TOTAL:
        print(f"\nOnly {acc:.1f}s available -- under the {MIN_TOTAL:.0f}s minimum.")
        sys.exit(3)

    stamp = "+".join(sorted({os.path.basename(s)[:10] for s in used}))
    slug = target.replace(" ", "-").replace(",", "")
    os.makedirs(outdir, exist_ok=True)
    dest = os.path.join(outdir, f"{slug}_voice-enrollment_{stamp}.wav")
    extract(chosen, dest)
    print(f"  wrote {dest}")

    def reject(msg, code):
        """A sample that fails verification is deleted rather than left behind.
        A mixed clip in the enrollment folder is worse than no clip -- it looks
        like a usable asset and would quietly poison whoever picks it up."""
        os.remove(dest)
        print(f"\n{msg}\nRemoved {os.path.basename(dest)}.")
        sys.exit(code)

    print("\nVerifying the extract is one speaker ...")
    check = post("/identify", {"threshold": threshold}, {"audio": dest})
    speakers = check.get("speakers", [])
    if not speakers:
        reject("No speech detected in the extract.", 4)
    top = speakers[0]
    print(f"  {len(speakers)} cluster(s); dominant {top['label']} "
          f"at {top['speech_pct']}% of speech")
    for s in speakers[1:]:
        print(f"    also {s['label']} {s['speech_pct']}%")

    if top["speech_pct"] < VERIFY_SHARE:
        reject(f"Extract is only {top['speech_pct']}% one speaker "
               f"(want >= {VERIFY_SHARE}%) -- the sample is mixed.", 5)
    if top["label"] != target:
        reject(f"Extract identifies as {top['label']!r}, not {target!r}.", 6)

    print("  clean")
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
