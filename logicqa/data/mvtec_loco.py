"""MVTec LOCO AD dataset utilities: presence check and download."""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


# MVTec LOCO AD class names
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
    label: str        # "good" / "logical_anomalies" / "structural_anomalies" (or similar from tags)
    is_anomaly: bool  # True if label != "good"
    class_name: str


class MVTecLOCODataset:
    """
    MVTec LOCO AD dataset loader for the dataset-ninja format.

    Expected directory structure:
        dataset-ninja/
            mvtec-loco-ad/
                train/
                    ann/
                        *.json
                    img/
                        *.png, *.jpg
                test/
                    ann/
                        *.json
                    img/
                        *.png, *.jpg
                validation/
                    ...

    The JSON files contain tags such as:
    - name: "breakfast_box" (the class)
    - name: "good" (the defect type / label)
    """

    DATASET_FOLDER_NAME = "mvtec-loco-ad"

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

    # ------------------------------------------------------------------ #
    # Presence check & download
    # ------------------------------------------------------------------ #

    def _is_present(self) -> bool:
        """Return True if the dataset directory exists and contains train/test dirs."""
        return (
            self.dataset_root.exists() and
            (self.dataset_root / "train" / "img").exists() and
            (self.dataset_root / "test" / "img").exists()
        )

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

    # ------------------------------------------------------------------ #
    # Data loading helpers
    # ------------------------------------------------------------------ #

    def _parse_annotation(self, ann_file: Path) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse dataset-ninja JSON annotation to extract (class_name, label).
        Tags typically contain the class (e.g. 'breakfast_box') and the label (e.g. 'good').
        """
        with open(ann_file, "r") as f:
            data = json.load(f)

        tags = [t.get("name", "").lower() for t in data.get("tags", [])]

        found_class = None
        found_label = None

        for t in tags:
            if t in MVTEC_LOCO_CLASSES:
                found_class = t
            elif t:
                # E.g. "good", "logical_anomalies", "structural_anomalies"
                found_label = t

        return found_class, found_label

    def _get_samples_from_split(self, split: str) -> List[ImageSample]:
        """Load all images from a specific split (train, test, validation) for this class."""
        img_dir = self.dataset_root / split / "img"
        ann_dir = self.dataset_root / split / "ann"

        if not img_dir.exists() or not ann_dir.exists():
            return []

        samples = []
        # Get all images
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            for img_path in img_dir.glob(ext):
                # Anns have same stem but .json extension (usually)
                # Note: dataset-ninja sometimes names it {original_name}.json
                # e.g., breakfast_box_000.png -> breakfast_box_000.png.json
                # or breakfast_box_000.json. Let's try both.
                ann_path1 = ann_dir / f"{img_path.name}.json"
                ann_path2 = ann_dir / f"{img_path.stem}.json"
                
                ann_path = ann_path1 if ann_path1.exists() else ann_path2
                if not ann_path.exists():
                    continue

                cls_name, label = self._parse_annotation(ann_path)

                # Filter by our requested class
                if cls_name == self.class_name:
                    is_anomaly = (label != "good")
                    samples.append(ImageSample(
                        path=img_path,
                        label=label or "unknown",
                        is_anomaly=is_anomaly,
                        class_name=self.class_name,
                    ))

        return sorted(samples, key=lambda s: s.path.name)

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def get_train_normal(self) -> List[Path]:
        """Return sorted list of train-normal image paths."""
        samples = self._get_samples_from_split("train")
        normal_paths = [s.path for s in samples if not s.is_anomaly]
        
        if not normal_paths:
            raise FileNotFoundError(
                f"No 'good' train samples found for class '{self.class_name}'"
            )
        return normal_paths

    def get_test_images(self) -> List[ImageSample]:
        """
        Return all test image samples with labels.
        (Includes both test and validation splits if available, per standard AD practice, 
         or just 'test' split depending on dataset-ninja split mapping).
        """
        samples = self._get_samples_from_split("test")
        # Optionally add validation if dataset separated them
        samples.extend(self._get_samples_from_split("val"))
        samples.extend(self._get_samples_from_split("validation"))
        return sorted(samples, key=lambda s: s.path.name)

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
