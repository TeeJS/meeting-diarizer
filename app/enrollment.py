"""
Speaker enrollment store — saves and loads speaker voice embeddings as .npy files.
"""

import logging
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

log = logging.getLogger(__name__)


class EnrollmentStore:
    def __init__(self, directory: Path):
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, name: str) -> Path:
        """Resolve a speaker name to its file, refusing anything that would
        escape the enrollment directory. Names arrive from HTTP form fields and
        are used directly as filenames."""
        n = (name or "").strip()
        if not n or n in (".", "..") or any(c in n for c in "/\\\x00"):
            raise ValueError(f"invalid speaker name: {name!r}")
        path = (self._dir / f"{n}.npy").resolve()
        if path.parent != self._dir.resolve():
            raise ValueError(f"invalid speaker name: {name!r}")
        return path

    def save(self, name: str, embedding: np.ndarray):
        path = self._path_for(name)
        np.save(path, embedding)
        log.info("Saved embedding for speaker: %s", name)

    def delete_speaker(self, name: str):
        self._path_for(name).unlink(missing_ok=True)
        log.info("Deleted embedding for speaker: %s", name)

    def rename_speaker(self, old: str, new: str):
        """Move an existing embedding to a new name.

        Renaming does not need the original audio, which matters because
        directory renames (Matt -> Matthew) would otherwise force a
        re-enrollment nobody has a reference recording for.
        """
        src = self._path_for(old)
        dst = self._path_for(new)
        if not src.exists():
            raise FileNotFoundError(f"no enrolled speaker named {old!r}")
        if dst.exists():
            raise FileExistsError(f"a speaker named {new!r} is already enrolled")
        src.rename(dst)
        log.info("Renamed enrolled speaker: %s -> %s", old, new)

    def list_speakers(self) -> List[str]:
        return sorted(p.stem for p in self._dir.glob("*.npy"))

    def list_details(self) -> List[Dict]:
        """Speakers with the date their embedding was written.

        The file's mtime is the enrollment date -- nothing separate is stored.
        A rename keeps the inode, so renaming a profile preserves the date it
        was originally enrolled, which is the behaviour worth having: the date
        answers "how stale is this voice profile", not "when was this label
        last edited". Restoring the directory from a backup would reset it.
        """
        out = []
        for p in sorted(self._dir.glob("*.npy"), key=lambda q: q.stem):
            try:
                enrolled_at = datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)
                stamp = enrolled_at.isoformat()
            except OSError:
                stamp = None
            out.append({"name": p.stem, "enrolled_at": stamp})
        return out

    def all_embeddings(self) -> Dict[str, np.ndarray]:
        return {p.stem: np.load(p) for p in self._dir.glob("*.npy")}
