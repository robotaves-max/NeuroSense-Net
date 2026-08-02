import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.utils import MELD_EMOTIONS, save_json

logger = logging.getLogger("neurosense.inspection")


class DatasetInspector:
    """Discovers, inspects, and analyzes multimodal MELD preprocessed datasets."""

    SEARCH_PATHS = [
        Path("/kaggle/input"),
        Path("/kaggle/working"),
        Path("./data"),
        Path("."),
    ]

    KEY_ALIASES = {
        "text": ["text", "text_features", "text_emb", "bert", "roberta", "glove", "t"],
        "audio": ["audio", "audio_features", "audio_emb", "wav2vec", "openSmile", "a"],
        "video": ["video", "visual", "video_features", "visual_features", "v"],
        "label": ["label", "labels", "emotion", "target", "y"],
        "id": ["id", "utterance_id", "sample_id", "clip_id"],
    }

    def __init__(self, data_root: Optional[str] = None, key_overrides: Optional[Dict[str, str]] = None):
        self.data_root = Path(data_root) if data_root else None
        self.key_overrides = key_overrides or {}
        self.discovered_root: Optional[Path] = None
        self.splits: Dict[str, Path] = {}
        self.report: Dict[str, Any] = {}

    def discover(self) -> Path:
        """Locate dataset directory across candidate locations."""
        candidates = []
        if self.data_root:
            candidates.append(self.data_root)
        candidates.extend(self.SEARCH_PATHS)

        for candidate in candidates:
            if not candidate.exists():
                continue
            # Check if this candidate itself contains split files/folders
            if self._is_meld_dir(candidate):
                self.discovered_root = candidate
                return candidate
            # Search recursively for matching child subdirectories
            for sub in candidate.rglob("*"):
                if sub.is_dir() and self._is_meld_dir(sub):
                    self.discovered_root = sub
                    return sub

        raise FileNotFoundError(
            f"Could not locate MELD preprocessed dataset in candidate paths: {[str(c) for c in candidates]}"
        )

    def _is_meld_dir(self, path: Path) -> bool:
        """Check if directory contains MELD split files or subfolders."""
        split_keywords = ["train", "dev", "val", "validation", "test"]
        files = list(path.glob("*"))
        names_lower = [f.name.lower() for f in files]
        found_splits = sum(1 for kw in split_keywords if any(kw in n for n in names_lower))
        return found_splits >= 1

    def inspect(self) -> Dict[str, Any]:
        """Perform full structural inspection of dataset files and tensors."""
        root = self.discover()
        logger.info(f"Discovered dataset root at: {root}")

        files_found = [str(f.relative_to(root)) for f in root.rglob("*") if f.is_file()]
        splits_found = self._find_splits(root)
        # Keep the discovered paths for callers such as the training and
        # evaluation scripts. Previously ``inspect`` returned split metadata
        # but left ``self.splits`` empty, making every real-data run fail.
        self.splits = splits_found

        split_details = {}
        inferred_dims = {"text": None, "audio": None, "video": None}
        inferred_keys = {"text": None, "audio": None, "video": None, "label": None}
        missing_stats = {}

        for split_name, split_path in splits_found.items():
            sample_count, keys_found, dims_found, sample_summary = self._inspect_split_file(split_path)
            split_details[split_name] = {
                "path": str(split_path),
                "num_samples": sample_count,
                "detected_keys": keys_found,
                "detected_dims": dims_found,
                "sample_summary": sample_summary,
            }
            # Aggregate inferred dimensions
            for mod in ["text", "audio", "video"]:
                if dims_found.get(mod) is not None:
                    inferred_dims[mod] = dims_found[mod]
                if keys_found.get(mod) is not None:
                    inferred_keys[mod] = keys_found[mod]
            if keys_found.get("label") is not None:
                inferred_keys["label"] = keys_found["label"]

        self.report = {
            "discovered_root": str(root),
            "total_files": len(files_found),
            "files": files_found[:20],  # sample list
            "splits": split_details,
            "inferred_keys": inferred_keys,
            "inferred_dims": inferred_dims,
            "canonical_emotions": MELD_EMOTIONS,
        }
        return self.report

    def _find_splits(self, root: Path) -> Dict[str, Path]:
        """Identify train, val/dev, test splits."""
        splits = {}
        split_map = {
            "train": ["train"],
            "val": ["dev", "val", "validation"],
            "test": ["test"],
        }
        all_items = list(root.glob("*")) + list(root.rglob("*.pt")) + list(root.rglob("*.pth")) + list(root.rglob("*.pkl"))

        for canonical, aliases in split_map.items():
            for item in all_items:
                name_lower = item.name.lower()
                if any(alias in name_lower for alias in aliases):
                    splits[canonical] = item
                    break
        return splits

    def _inspect_split_file(
        self, split_path: Path
    ) -> Tuple[int, Dict[str, str], Dict[str, Optional[int]], Dict[str, Any]]:
        """Inspect a single split file or directory."""
        keys_found = {}
        dims_found = {"text": None, "audio": None, "video": None}
        sample_count = 0
        sample_summary = {}

        if split_path.suffix in [".pt", ".pth"]:
            try:
                data = torch.load(split_path, map_location="cpu", weights_only=False)
                if isinstance(data, dict):
                    sample_count = len(next(iter(data.values()))) if isinstance(next(iter(data.values())), (list, torch.Tensor)) else len(data)
                    keys_found, dims_found = self._infer_dict_keys_and_dims(data)
                elif isinstance(data, (list, tuple)):
                    sample_count = len(data)
                    if sample_count > 0 and isinstance(data[0], dict):
                        keys_found, dims_found = self._infer_dict_keys_and_dims(data[0])
            except Exception as e:
                logger.warning(f"Error loading {split_path}: {e}")

        elif split_path.is_dir():
            files = list(split_path.glob("*"))
            sample_count = len(files)
            if sample_count > 0 and files[0].suffix in [".pt", ".pth"]:
                try:
                    first_obj = torch.load(files[0], map_location="cpu", weights_only=False)
                    if isinstance(first_obj, dict):
                        keys_found, dims_found = self._infer_dict_keys_and_dims(first_obj)
                except Exception as e:
                    logger.warning(f"Error inspecting file in {split_path}: {e}")

        return sample_count, keys_found, dims_found, sample_summary

    def _infer_dict_keys_and_dims(self, obj: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, Optional[int]]]:
        """Match modality keys and infer feature vector/sequence dimension."""
        keys_found = {}
        dims_found = {"text": None, "audio": None, "video": None}

        for modality, aliases in self.KEY_ALIASES.items():
            if modality in self.key_overrides:
                override = self.key_overrides[modality]
                if override in obj:
                    keys_found[modality] = override
                    continue

            for alias in aliases:
                for actual_key in obj.keys():
                    if alias.lower() == str(actual_key).lower() or alias.lower() in str(actual_key).lower():
                        keys_found[modality] = str(actual_key)
                        break
                if modality in keys_found:
                    break

        for mod in ["text", "audio", "video"]:
            k = keys_found.get(mod)
            if k and k in obj:
                val = obj[k]
                if isinstance(val, torch.Tensor):
                    dims_found[mod] = val.shape[-1]
                elif isinstance(val, np.ndarray):
                    dims_found[mod] = val.shape[-1]
                elif isinstance(val, (list, tuple)) and len(val) > 0:
                    first = val[0]
                    if isinstance(first, torch.Tensor):
                        dims_found[mod] = first.shape[-1]
                    elif isinstance(first, np.ndarray):
                        dims_found[mod] = first.shape[-1]

        return keys_found, dims_found

    def generate_reports(self, output_dir: Path) -> Tuple[Path, Path]:
        """Save dataset_report.txt and dataset_report.json to output directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_json_path = output_dir / "dataset_report.json"
        report_txt_path = output_dir / "dataset_report.txt"

        save_json(self.report, report_json_path)

        lines = [
            "=" * 60,
            "NEUROSENSE-NET DATASET INSPECTION REPORT",
            "=" * 60,
            f"Discovered Root: {self.report.get('discovered_root')}",
            f"Total Files Scanned: {self.report.get('total_files')}",
            "\nSplits Discovered:",
        ]

        for split, info in self.report.get("splits", {}).items():
            lines.append(f"  - {split}: {info.get('num_samples')} samples | path: {info.get('path')}")

        lines.extend([
            "\nInferred Keys:",
            f"  {json.dumps(self.report.get('inferred_keys'), indent=4)}",
            "\nInferred Feature Dimensions:",
            f"  {json.dumps(self.report.get('inferred_dims'), indent=4)}",
            "\nCanonical Emotions (7):",
            f"  {MELD_EMOTIONS}",
            "=" * 60,
        ])

        report_txt = "\n".join(lines)
        report_txt_path.write_text(report_txt, encoding="utf-8")
        logger.info(f"Dataset reports saved to:\n  - {report_json_path}\n  - {report_txt_path}")
        return report_json_path, report_txt_path
