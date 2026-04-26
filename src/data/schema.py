"""Schema validation cho raw data files."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import load_yaml, project_root, read_csv_safe
from src.utils.logging import get_logger

log = get_logger(__name__)


class SchemaError(ValueError):
    """Raised khi raw file không khớp schema."""


def validate_raw_file(path: str | Path, expected_cols: list[str], min_rows: int = 100) -> pd.DataFrame:
    """Đọc CSV và validate schema. Raise SchemaError nếu thiếu cột hoặc quá ít dòng."""
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    if not p.exists():
        raise SchemaError(f"File không tồn tại: {p}")
    df = read_csv_safe(p)
    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise SchemaError(f"{p.name} thiếu cột: {missing}. Có: {list(df.columns)}")
    if len(df) < min_rows:
        raise SchemaError(f"{p.name} chỉ có {len(df)} dòng, tối thiểu {min_rows}.")
    return df


def validate_all_raw(config_path: str = "configs/data.yaml") -> dict[str, pd.DataFrame]:
    """Validate toàn bộ raw files theo schema config. Trả về dict {name: df}."""
    cfg = load_yaml(config_path)
    raw_dir = project_root() / cfg["paths"]["raw_dir"]
    expected = cfg["schema"]["expected_columns"]
    min_rows = cfg["schema"]["min_rows_per_file"]

    name_to_path = {
        "Gold_Futures":   raw_dir / "Gold_Futures_ohlcv.csv",
        "USD_Index":      raw_dir / "USD_Index_ohlcv.csv",
        "VN_Index":       raw_dir / "VN_Index_ohlcv.csv",
        "Oil_WTI":        raw_dir / "Oil_WTI_ohlcv.csv",
        "SJC":            raw_dir / "SJC_gold_historical.csv",
        "Interest_Rate":  raw_dir / "interest_rate.csv",
    }
    results: dict[str, pd.DataFrame] = {}
    for name, path in name_to_path.items():
        if not path.exists():
            log.warning(f"Bỏ qua {name}: file chưa có ({path.name})")
            continue
        # Map name → expected key in config (Gold_Futures has its own; others use generic OHLCV)
        cols_key = name if name in expected else "Gold_Futures"  # default OHLCV
        cols = expected.get(cols_key, ["Date", "Close"])
        df = validate_raw_file(path, expected_cols=cols, min_rows=min_rows)
        log.info(f"OK {name}: {len(df)} dòng, cột {list(df.columns)}")
        results[name] = df
    return results


if __name__ == "__main__":
    validate_all_raw()
