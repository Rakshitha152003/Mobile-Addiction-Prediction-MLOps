import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union
import joblib
import yaml


def load_params(params_path: Union[str, Path] = "params.yaml") -> Dict[str, Any]:
    """Load configuration parameters from a YAML file."""
    path = Path(params_path)
    if not path.is_absolute():
        # Search relative to project root (2 levels up from utils or current working dir)
        project_root = Path(__file__).resolve().parent.parent.parent
        candidate = project_root / path
        if candidate.exists():
            path = candidate

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    return params or {}


def create_directories(paths: List[Union[str, Path]]) -> None:
    """Create a list of directories if they do not exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path: Union[str, Path], data: Any) -> None:
    """Save serializable data to a JSON file."""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_json(path: Union[str, Path]) -> Any:
    """Load data from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_joblib(path: Union[str, Path], obj: Any) -> None:
    """Persist an object using joblib."""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, target_path)


def load_joblib(path: Union[str, Path]) -> Any:
    """Load an object using joblib."""
    return joblib.load(path)


def setup_logger(name: str, log_file: Union[str, Path], level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger with file and console handlers."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if logger was already created
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
