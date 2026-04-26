"""I/O helpers: YAML, paths, parquet."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def project_root() -> Path:
    """Trả về thư mục gốc của dự án (chứa pyproject.toml)."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Không tìm thấy project root (pyproject.toml).")


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Đọc YAML. Path tương đối được resolve theo project root."""
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_csv_safe(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Đọc CSV với utf-8-sig (handle BOM Vietnamese sources)."""
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    return pd.read_csv(p, encoding="utf-8-sig", **kwargs)


def write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """Ghi parquet với compression snappy."""
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, compression="snappy", index=False)
    return p


def read_parquet(path: str | Path) -> pd.DataFrame:
    """Đọc parquet."""
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    return pd.read_parquet(p)
