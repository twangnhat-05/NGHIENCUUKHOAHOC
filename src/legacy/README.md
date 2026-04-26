# Legacy Scripts (Frozen)

Đây là source code gốc của dự án trước khi Claude Architect refactor (commit `pre-claude-v0`).

**KHÔNG SỬA** các file trong folder này. Chúng được giữ lại để:
1. Reproduce các kết quả cũ (output/ legacy plots)
2. So sánh với pipeline mới
3. Audit trail cho hội đồng nghiệm thu

Code mới sống trong `src/{data,features,models,training,evaluation,xai,utils}/`.

## Mapping cũ → mới

| Legacy | New module |
|---|---|
| `download_data.py` | `src/data/fetch.py` + `src/data/refresh.py` |
| `eda_merge_analysis.py` | `src/data/merge.py` + `notebooks/00_eda_v2.ipynb` |
| `feature_engineering.py` | `src/features/build.py` + `src/features/technical.py` |
| `compare_baselines.py` | `src/models/classical.py` + `src/evaluation/leaderboard.py` |
| `train_xgboost.py` | `src/models/ml.py` (XGBoost class) |
| `train_random_forest.py` | `src/models/ml.py` (RandomForest class) |
| `train_lstm.py` | `src/models/dl.py` (LSTM class) |

## Để chạy lại pipeline cũ
```bash
cd src/legacy
python download_data.py     # ⚠️ data path hardcoded "data/", phải symlink hoặc edit
python eda_merge_analysis.py
python feature_engineering.py
python compare_baselines.py
python train_xgboost.py
python train_random_forest.py
python train_lstm.py
```

> **Lưu ý**: legacy scripts mong data ở `data/`, nhưng đã chuyển sang `data/raw/`.
> Nếu muốn chạy lại, tạo symlink: `mklink /D data\subset data\raw` (Windows)
> hoặc copy CSV về `data/`.
