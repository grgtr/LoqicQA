"""MVTec LOCO AD dataset utilities: presence check and download."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


# MVTec LOCO AD class names (as they appear in the dataset directory)
MVTEC_LOCO_CLASSES = [
    "breakfast_box",
    "juice_bottle",
    "pushpins",
    "screw_bag",
    "splicing_connectors",
]


@dataclass
class ImageSample:
    """A single image sample from the dataset."""
    path: Path
    label: str        # "good" / "logical_anomalies" / "structural_anomalies"
    is_anomaly: bool  # True if label != "good"
    class_name: str


class MVTecLOCODataset:
    """
    MVTec LOCO AD dataset loader.

    Expected directory structure (dataset-ninja layout):
        <data_dir>/
            MVTec LOCO AD/
                <class_name>/
                    train/
                        good/
                            *.png
                    test/
                        good/
                            *.png
                        logical_anomalies/
                            *.png
                        structural_anomalies/
                            *.png        (present for some classes)

    Args:
        data_dir:   Root directory where the dataset lives (e.g., ~/dataset-ninja/).
        class_name: One of the 5 MVTec LOCO AD class names.
        download_if_missing: If True, download via dataset_tools when not found.
    """

    DATASET_FOLDER_NAME = "MVTec LOCO AD"

    def __init__(
        self,
        data_dir: str | Path,
        class_name: str,
        download_if_missing: bool = True,
    ):
        self.data_dir = Path(os.path.expanduser(str(data_dir)))
        self.class_name = class_name.lower().replace(" ", "_")
        self.dataset_root = self.data_dir / self.DATASET_FOLDER_NAME

        if not self._is_present():
            if download_if_missing:
                self.download(self.data_dir)
            else:
                raise FileNotFoundError(
                    f"MVTec LOCO AD dataset not found at {self.dataset_root}. "
                    f"Set download_if_missing=True or download manually."
                )

        self.class_root = self._find_class_dir()

    # ------------------------------------------------------------------ #
    # Presence check & download
    # ------------------------------------------------------------------ #

    def _is_present(self) -> bool:
        """Return True if the dataset directory exists and is non-empty."""
        if not self.dataset_root.exists():
            return False
        # Check that at least one class folder exists
        for cls in MVTEC_LOCO_CLASSES:
            candidate = self.dataset_root / cls
            # Also try with spaces (some naming conventions)
            alt = self.dataset_root / cls.replace("_", " ")
            if candidate.exists() or alt.exists():
                return True
        return False

    @staticmethod
    def download(dst_dir: str | Path) -> None:
        """
        Download the MVTec LOCO AD dataset using dataset_tools.

        Args:
            dst_dir: Destination directory (e.g., '~/dataset-ninja/').
        """
        dst_dir = str(os.path.expanduser(str(dst_dir)))
        try:
            import dataset_tools as dtools
        except ImportError:
            raise ImportError(
                "dataset_tools is not installed. "
                "Install it with: pip install dataset-tools"
            )

        print(f"[Dataset] Downloading 'MVTec LOCO AD' to {dst_dir} ...")
        dtools.download(dataset="MVTec LOCO AD", dst_dir=dst_dir)
        print("[Dataset] Download complete.")

    def _find_class_dir(self) -> Path:
        """Locate the directory for self.class_name (handles spaces/underscores)."""
        candidates = [
            self.dataset_root / self.class_name,
            self.dataset_root / self.class_name.replace("_", " "),
        ]
        for c in candidates:
            if c.exists():
                return c

        # Fuzzy match: find closest
        existing = [d.name for d in self.dataset_root.iterdir() if d.is_dir()]
        norm_name = self.class_name.replace("_", " ").lower()
        for ex in existing:
            if ex.lower() == norm_name or ex.lower().replace(" ", "_") == self.class_name:
                return self.dataset_root / ex

        raise FileNotFoundError(
            f"Class directory for '{self.class_name}' not found "
            f"under {self.dataset_root}. Available: {existing}"
        )

    # ------------------------------------------------------------------ #
    # Data loading
    # ------------------------------------------------------------------ #

    def get_train_normal(self) -> List[Path]:
        """Return sorted list of train-normal image paths."""
        train_good = self.class_root / "train" / "good"
        if not train_good.exists():
            raise FileNotFoundError(f"Train-normal dir not found: {train_good}")
        images = sorted(train_good.glob("*.png")) + sorted(train_good.glob("*.jpg"))
        return images

    def get_test_images(self) -> List[ImageSample]:
        """
        Return all test image samples with labels.

        Returns:
            List of ImageSample (path, label, is_anomaly, class_name).
        """
        test_dir = self.class_root / "test"
        if not test_dir.exists():
            raise FileNotFoundError(f"Test dir not found: {test_dir}")

        samples: List[ImageSample] = []
        for label_dir in sorted(test_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            label = label_dir.name
            is_anomaly = label != "good"
            for img_path in sorted(label_dir.glob("*.png")):
                samples.append(ImageSample(
                    path=img_path,
                    label=label,
                    is_anomaly=is_anomaly,
                    class_name=self.class_name,
                ))
            for img_path in sorted(label_dir.glob("*.jpg")):
                samples.append(ImageSample(
                    path=img_path,
                    label=label,
                    is_anomaly=is_anomaly,
                    class_name=self.class_name,
                ))
        return samples

    def sample_train_normal(
        self,
        n: int = 3,
        seed: Optional[int] = None,
    ) -> List[Path]:
        """
        Randomly sample `n` normal images from the training set (few-shot).

        Args:
            n:    Number of images to sample.
            seed: Random seed for reproducibility.

        Returns:
            List of n image paths.
        """
        all_normal = self.get_train_normal()
        if len(all_normal) < n:
            raise ValueError(
                f"Only {len(all_normal)} train-normal images available for "
                f"'{self.class_name}', but n={n} requested."
            )
        rng = random.Random(seed)
        return rng.sample(all_normal, n)
