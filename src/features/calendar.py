"""Calendar-based features cho chuỗi thời gian VN.

Bao gồm: dow, dom, month, quarter, is_month_end, holidays VN.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

# ============================================================
# VIETNAM HOLIDAYS (cố định, không tính ngày bù)
# ============================================================
# Tết âm lịch (xấp xỉ — chỉ dùng để tạo cờ "near_tet" thay vì exact date)
# Các ngày dương lịch cố định:
VN_FIXED_HOLIDAYS_MMDD: list[tuple[int, int]] = [
    (1, 1),     # Tết Dương lịch
    (4, 30),    # Giải phóng miền Nam
    (5, 1),     # Quốc tế Lao động
    (9, 2),     # Quốc khánh
    (3, 8),     # 8/3 Phụ nữ — không nghỉ chính thức nhưng quan trọng văn hóa
    (10, 20),   # Phụ nữ VN
]

# Tết âm lịch dương lịch (tra bảng — 2018-2027)
VN_LUNAR_NEW_YEAR: dict[int, tuple[date, date]] = {
    2018: (date(2018, 2, 14), date(2018, 2, 21)),
    2019: (date(2019, 2, 2),  date(2019, 2, 10)),
    2020: (date(2020, 1, 23), date(2020, 1, 29)),
    2021: (date(2021, 2, 10), date(2021, 2, 16)),
    2022: (date(2022, 1, 29), date(2022, 2, 6)),
    2023: (date(2023, 1, 20), date(2023, 1, 28)),
    2024: (date(2024, 2, 8),  date(2024, 2, 14)),
    2025: (date(2025, 1, 25), date(2025, 2, 2)),
    2026: (date(2026, 2, 15), date(2026, 2, 22)),
    2027: (date(2027, 2, 5),  date(2027, 2, 12)),
}


def _is_vn_holiday(d: date) -> bool:
    """Kiểm tra 1 ngày có là lễ VN không (cố định + Tết)."""
    if (d.month, d.day) in VN_FIXED_HOLIDAYS_MMDD:
        return True
    tet_range = VN_LUNAR_NEW_YEAR.get(d.year)
    if tet_range and tet_range[0] <= d <= tet_range[1]:
        return True
    return False


def _days_to_tet(d: date) -> int:
    """Số ngày tới Tết gần nhất (signed: âm nếu sau Tết trong cùng năm)."""
    tet_range = VN_LUNAR_NEW_YEAR.get(d.year)
    if tet_range is None:
        return 365
    tet_start = tet_range[0]
    if d <= tet_start:
        return (tet_start - d).days
    # Đã qua Tết: dùng Tết năm sau
    next_tet = VN_LUNAR_NEW_YEAR.get(d.year + 1)
    if next_tet is None:
        return 365
    return (next_tet[0] - d).days


def add_calendar_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Thêm calendar features vào df. Không sửa df gốc."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out["cal_dow"] = out[date_col].dt.dayofweek                 # 0=Mon
    out["cal_dom"] = out[date_col].dt.day
    out["cal_month"] = out[date_col].dt.month
    out["cal_quarter"] = out[date_col].dt.quarter
    out["cal_is_month_end"] = out[date_col].dt.is_month_end.astype(int)
    out["cal_is_month_start"] = out[date_col].dt.is_month_start.astype(int)
    out["cal_is_quarter_end"] = out[date_col].dt.is_quarter_end.astype(int)
    # Cyclical encoding (DOW, month) — giúp model nhận pattern tuần/năm
    out["cal_dow_sin"] = np.sin(2 * np.pi * out["cal_dow"] / 7.0)
    out["cal_dow_cos"] = np.cos(2 * np.pi * out["cal_dow"] / 7.0)
    out["cal_month_sin"] = np.sin(2 * np.pi * (out["cal_month"] - 1) / 12.0)
    out["cal_month_cos"] = np.cos(2 * np.pi * (out["cal_month"] - 1) / 12.0)
    # VN holidays
    out["cal_is_vn_holiday"] = out[date_col].dt.date.apply(_is_vn_holiday).astype(int)
    out["cal_days_to_tet"] = out[date_col].dt.date.apply(_days_to_tet).clip(0, 60)
    return out
