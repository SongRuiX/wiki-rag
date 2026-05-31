from pathlib import Path
import yaml

_CONFIG: dict | None = None


def load_config(path: str | None = None) -> dict:
    """Load config.yaml, resolving relative to project root."""
    global _CONFIG
    if path is None:
        path = Path(__file__).parent.parent / "config.yaml"
    with open(path, encoding="utf-8") as f:
        _CONFIG = yaml.safe_load(f)
    return _CONFIG


def get_config() -> dict:
    """Return cached config; call load_config first."""
    if _CONFIG is None:
        return load_config()
    return _CONFIG
