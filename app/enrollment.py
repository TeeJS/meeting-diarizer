"""
Speaker enrollment store — saves and loads speaker voice embeddings as .npy files.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List

log = logging.getLogger(__name__)


class EnrollmentStore:
    def __init__(self, directory: Path):
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, embedding: np.ndarray):
        path = self._dir / f"{name}.npy"
        np.save(path, embedding)
        log.info("Saved embedding for speaker: %s", name)

    def delete_speaker(self, name: str):
        path = self._dir / f"{name}.npy"
        path.unlink(missing_ok=True)
        log.info("Deleted embedding for speaker: %s", name)

    def list_speakers(self) -> List[str]:
        return sorted(p.stem for p in self._dir.glob("*.npy"))

    def all_embeddings(self) -> Dict[str, np.ndarray]:
        return {p.stem: np.load(p) for p in self._dir.glob("*.npy")}
