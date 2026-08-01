import json
import logging
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

MELD_EMOTIONS = ["neutral", "joy", "surprise", "sadness", "anger", "disgust", "fear"]


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds across python, numpy, pytorch for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def setup_logger(
    name: str = "neurosense",
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Set up and return a clean logger writing to stdout and optionally a log file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return torch.device (CUDA if available and preferred, else CPU)."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_git_hash() -> str:
    """Attempt to get current git commit hash, returning unknown if unavailable."""
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("utf-8").strip()
        return output
    except Exception:
        return "unknown"


def get_environment_info() -> Dict[str, Any]:
    """Capture environment diagnostic information for run reproducibility."""
    return {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "git_commit": get_git_hash(),
        "platform": sys.platform,
    }


def save_json(data: Any, filepath: Path) -> None:
    """Save serializable data to JSON file with UTF-8 encoding."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: Path) -> Any:
    """Load JSON data from file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_yaml(data: Any, filepath: Path) -> None:
    """Save data to YAML file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_yaml(filepath: Path) -> Dict[str, Any]:
    """Load YAML config from file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
