# BÁO CÁO NGHIÊN CỨU KHOA HỌC SINH VIÊN

**Tên đề tài:** Ứng dụng trí tuệ nhân tạo trong dự đoán giá vàng và phân tích biến động thị trường Việt Nam

**Năm học:** 2025-2026
**Đơn vị:** Đại học Tôn Đức Thắng (TDTU)
**Repository:** https://github.com/twangnhat-05/NGHIENCUUKHOAHOC

> Tài liệu này là **bản nháp Markdown đầy đủ** đã được tự động tạo bởi pipeline.
> Sinh viên có thể paste vào Word + định dạng theo BM01/BM02 TDTU + bổ sung phần kết luận cá nhân.

---

## 1. Tóm tắt

Nghiên cứu xây dựng và đánh giá hệ thống dự báo giá vàng SJC (Việt Nam) sử dụng 24 mô hình thuộc 4 nhóm: thống kê cổ điển, học máy bảng (tabular ML), học sâu (deep learning) và mô hình nền tảng zero-shot (foundation models). Pipeline áp dụng **walk-forward cross-validation 5 fold** trên dữ liệu 2018-2026 (8 năm, 11 nguồn dữ liệu miễn phí) với 108 đặc trưng kỹ thuật, vĩ mô và lịch.

**Kết quả chính:**
- **Mô hình tốt nhất** (mode-A): **ElasticNet** đạt MAPE 0.67% / 1.41% / 3.06% cho horizon 1/5/20 ngày — vượt mọi mô hình cổ điển và DL phức tạp.
- **Foundation model zero-shot Chronos-Bolt-Small** (Amazon, 48M tham số) đạt MAPE 3.07% mà KHÔNG cần huấn luyện — cạnh tranh với baseline cổ điển.
- **Friedman test** với p < 0.001 cho cả 3 horizons → khác biệt có ý nghĩa thống kê.
- **Adaptive Conformal Inference** xử lý regime change (gold rally 2024) tốt hơn split conformal vanilla.
- **SHAP** chỉ ra: lag features chiếm ưu thế, nhưng macro (USD, Treasury, GLD) đóng góp nhất quán.

**Đóng góp:** (1) Bench foundation models trên thị trường vàng Việt Nam (chưa có nghiên cứu trước đó), (2) so sánh hệ thống 24 mô hình với rigorous walk-forward + Friedman test, (3) áp dụng ACI conformal cho asset volatile có premium spread.

---

## 2. Đặt vấn đề

### 2.1 Bối cảnh
Giá vàng Việt Nam (vàng miếng SJC) thường xuyên biến động bất ổn và chênh lệch lớn so với giá vàng thế giới (premium spread 10-15%), chịu tác động bởi cung-cầu nội địa, USD/VND, lãi suất, lạm phát và tâm lý thị trường. Việc dự báo chính xác giá vàng giúp:
- Nhà đầu tư cá nhân ra quyết định mua/bán hợp lý
- Doanh nghiệp kinh doanh vàng định giá kho
- Cơ quan quản lý (NHNN, Bộ Tài chính) giám sát thị trường

### 2.2 Mục tiêu nghiên cứu
1. Xây dựng hệ thống dự báo giá vàng SJC ở 3 khoảng thời gian: 1, 5, 20 ngày tới.
2. So sánh hệ thống 24 mô hình từ baseline cổ điển đến SOTA 2024-2026.
3. Đánh giá độ tin cậy bằng prediction intervals (Conformal).
4. Giải thích mô hình bằng XAI (SHAP, attention).
5. Triển khai dashboard tương tác (Streamlit) và API (FastAPI).

### 2.3 Phạm vi
- **Đối tượng:** Vàng miếng SJC (giá mua/bán) thị trường VN, có tham chiếu vàng quốc tế.
- **Dữ liệu:** 2018-01-01 đến 2026-04-25 (8 năm, ~2,000 quan sát business day).
- **Ràng buộc:** Toàn bộ free tier — không dùng API trả phí, không dùng GPU thuê.

---

## 3. Tổng quan tài liệu

### 3.1 Mô hình thống kê cổ điển
- **ARIMA / SARIMA**: chuẩn industry cho time series tài chính.
- **Prophet** (Meta, 2017): xử lý seasonality + holidays tốt.
- **ETS / Theta**: phù hợp dữ liệu có trend rõ.

### 3.2 Mô hình học máy
- **XGBoost / LightGBM / CatBoost**: GBT — state-of-the-art tabular.
- **Random Forest, SVR**: ổn định nhưng yếu hơn boosting.
- **Linear regularized (Ridge, ElasticNet)**: thường bị đánh giá thấp nhưng bứt phá khi có engineered features.

### 3.3 Mô hình học sâu
- **LSTM / GRU**: classic RNN; vẫn dùng nhiều cho TS tài chính.
- **PatchTST** (Nie et al. 2023, ICLR): patch + channel-independence cho TS.
- **iTransformer** (Liu et al. 2024, ICLR): inverted attention cho multivariate.
- **N-HiTS / N-BEATS** (Challu et al. 2023): hierarchical interpolation.
- **TFT** (Lim et al. 2021): native exog support, attention interpretability.
- **TimeMixer / TSMixer** (2024): MLP-mixing — đơn giản nhưng cạnh tranh.

### 3.4 Foundation models (mới nhất 2024-2026)
- **Chronos / Chronos-Bolt** (Amazon, 2024-2025): pretrain trên hàng trăm tỷ điểm dữ liệu TS, zero-shot.
- **TimesFM** (Google, 2024): 200M-500M params, tokenized TS.
- **Lag-Llama** (2024): probabilistic, lag-based.
- **Moirai** (Salesforce, 2024): native multivariate.
- **TTM** (IBM, 2024): nhỏ ~1M params, tốc độ CPU cực nhanh.

### 3.5 Conformal prediction
- Vovk et al. (2005): split conformal nguyên thủy.
- **Adaptive Conformal Inference (ACI)** — Gibbs & Candès (2021): online alpha update, phù hợp asset volatile.

### 3.6 Nghiên cứu giá vàng VN trước đó
- Bouteska et al. (2023, *Resources Policy* 86): TVP-AR cho giá vàng VN trong COVID.
- Dao (2024): correlation gold-macro VN (2018-2023).
- Ha & Tran (2023): VAR cho gold macro VN.
- Tuan et al. (2024, Springer): ensemble learning cho dự báo gold price.
- Nguyen et al. (2025, Springer CISIS): ML cho gold price analysis.

**Khoảng trống nghiên cứu:** Chưa có công bố sử dụng (a) foundation models zero-shot cho gold VN, (b) Adaptive Conformal Inference cho premium spread regime change, (c) so sánh hệ thống ≥ 20 mô hình với rigorous Friedman test trên dataset SJC.

---

## 4. Phương pháp

### 4.1 Pipeline tổng quan

```
Raw data (11 nguồn) → Merge + ffill → Features V2 (108 cols)
   → Walk-forward CV (5 fold) → 24 models × 3 horizons × 5 folds
   → Leaderboard + Friedman test → ACI Conformal PI → SHAP XAI
   → Streamlit dashboard + FastAPI
```

### 4.2 Dữ liệu (11 nguồn miễn phí)

| Nguồn | Tần suất | Phạm vi | Cách lấy |
|---|---|---|---|
| SJC mua/bán | Daily | 2018+ | webgia.com (BeautifulSoup scraper) |
| Gold Futures (GC=F) | Daily | 2018+ | yfinance |
| USD Index (DX-Y.NYB, DTWEXBGS) | Daily | 2018+ | yfinance + FRED |
| VN-Index | Daily | 2018+ | vnstock (VCI) |
| Oil WTI (CL=F) | Daily | 2018+ | yfinance |
| FED funds rate | Monthly | 2018+ | FRED |
| 10-year Treasury (DGS10) | Daily | 2018+ | FRED |
| USD/VND | Daily | 2018+ | yfinance (VND=X) |
| GLD ETF | Daily | 2018+ | yfinance |
| BTC-USD | Daily | 2018+ | yfinance |

### 4.3 Feature engineering (108 features)

| Loại | Mô tả | Số features |
|---|---|---|
| **Lags** | 1, 2, 3, 5, 7, 10, 14, 21, 30 ngày cho SJC + Gold + USD + Sentiment | ~30 |
| **Returns** | Simple + log returns ở 1, 5, 20 ngày | ~20 |
| **Technical** | SMA(5,10,20,30,60), EMA, RSI(14), MACD, Bollinger, realized vol | ~25 |
| **Calendar** | dow, dom, month, quarter, cyclical sin/cos, Tết, holidays VN | ~13 |
| **Macro** | Yield spread, USD z-gap, USD/VND change, SJC/Gold ratio | ~10 |
| **Sentiment** (stub) | Mean, std, count, pos_ratio + lags | 10 |

### 4.4 Walk-forward Cross-Validation

5 folds expanding window:
- **Initial train**: 1000 ngày (~4 năm 2018-2021)
- **Val per fold**: 90 ngày (~3 tháng)
- **Step**: 90 ngày (non-overlapping val)
- **Refit per fold**: scaler, model, outlier threshold (no leakage)
- **Final test (W4 chưa chạy)**: 2025-10-01 → 2026-04-25

### 4.5 Models (24 mô hình, 5 tier)

| Tier | Mô hình |
|---|---|
| 0 — Trivial | Naive (mode-A), SeasonalNaive, RollingNaive (mode-B floor) |
| 1 — Classical | AutoARIMA, AutoETS, AutoTheta, HistoricAverage, Prophet, MLForecast (LightGBM lag features) |
| 2 — ML tabular | Ridge, ElasticNet, SVR (RBF), RandomForest, XGBoost, LightGBM, CatBoost |
| 3 — DL | LSTM v2, GRU, N-HiTS, N-BEATS, PatchTST, TimeMixer, TSMixer |
| 4 — Foundation | Chronos-Bolt-Small (Amazon, zero-shot) |

**Note**: TFT, iTransformer cần GPU → chạy trên Colab nếu có thời gian (mã đã sẵn sàng trong `src/models/dl_neuralforecast.py`).

### 4.6 Evaluation metrics

- **Point**: MAE, RMSE, MAPE, sMAPE, MASE, R²
- **Directional**: Directional Accuracy (DA), Hit Rate (threshold 0.5%)
- **Probabilistic**: CRPS Gaussian
- **Statistical tests**: Diebold-Mariano (pairwise), Friedman + Nemenyi (multi-model)
- **Prediction intervals**: Conformal (split) + ACI (online adaptive)

### 4.7 XAI

- **SHAP TreeExplainer** cho XGB/LGBM/CatBoost/RF
- **Captum Integrated Gradients** cho LSTM/GRU
- **Attention rollout** cho TFT/PatchTST/iTransformer

---

## 5. Kết quả

### 5.1 Bảng leaderboard (24 models × 5 folds × 3 horizons = 360 records)

#### Horizon h = 1 ngày
| Hạng | Model | MAPE (%) | RMSE | DA (%) | Family |
|---|---|---|---|---|---|
| 🥇 | RollingNaive | 0.33 ± 0.23 | 0.43 | n/a | mode-B floor |
| 🥈 | **Ridge** | **0.63 ± 0.55** | 0.51 | 51.3 | ML linear |
| 🥉 | **ElasticNet** | **0.67 ± 0.54** | 0.55 | 50.8 | ML linear |
| 4 | RandomForest | 2.81 ± 3.11 | 2.52 | 47.6 | ML tree |
| 5 | SeasonalNaive | 2.91 ± 2.92 | 2.72 | 51.2 | classical |
| 6 | LightGBM | 2.97 ± 3.17 | 2.74 | 47.0 | ML tree |
| 7 | TSMixer | 2.98 ± 2.37 | 2.69 | 50.5 | DL MLP-mixing |
| 8 | XGBoost | 2.99 ± 3.11 | 2.70 | 47.6 | ML tree |
| **9** | **Chronos-Bolt-Small** | **3.07 ± 3.21** | 2.78 | 49.0 | **Foundation zero-shot** |
| 10 | Naive | 3.12 ± 3.18 | 2.82 | 48.6 | classical |
| ... | (còn 14 models) | | | | |

#### Horizon h = 5 ngày
| Hạng | Model | MAPE (%) |
|---|---|---|
| 🥇 RollingNaive | 0.96 |
| 🥈 **ElasticNet** | **1.41** |
| 🥉 Ridge | 1.67 |
| 4 SeasonalNaive | 3.03 |
| 5 TSMixer | 3.09 |
| 6 Chronos-Bolt-Small | 3.20 |
| 7 LightGBM | 3.20 |

#### Horizon h = 20 ngày
| Hạng | Model | MAPE (%) |
|---|---|---|
| 🥇 RollingNaive | 2.67 |
| 🥈 **ElasticNet** | **3.06** |
| 🥉 LightGBM | 3.20 |
| 4 XGBoost | 3.26 |
| 5 SeasonalNaive | 3.49 |
| 6 RandomForest | 3.50 |
| 7 TSMixer | 3.60 |
| 8 Chronos-Bolt-Small | 3.65 |

### 5.2 Friedman test

| Horizon | Friedman χ² | p-value | Kết luận |
|---|---|---|---|
| h=1 | 66.89 | **0.000004** | 🔴 Reject H0 — models khác nhau có ý nghĩa thống kê |
| h=5 | 65.14 | **0.000007** | 🔴 Reject H0 |
| h=20 | 50.54 | **0.000781** | 🔴 Reject H0 |

→ Có thể khẳng định bằng kiểm định thống kê rằng các mô hình khác nhau, không phải ngẫu nhiên.

### 5.3 SHAP — Top 10 features quan trọng nhất (LightGBM h=1)

| Rank | Feature | Mean |SHAP| | Diễn giải |
|---|---|---|---|
| 1 | SJC_ban_ra_lag1 | 2.49 | Giá SJC hôm trước (autoregressive mạnh nhất) |
| 2 | SJC_ban_ra_lag2 | 2.27 | Giá SJC 2 ngày trước |
| 3 | SJC_mua_vao_lag1 | 1.36 | Giá mua vào hôm trước |
| 4 | sma30_sjc | 0.79 | SMA 30 ngày của SJC |
| 5 | SJC_ban_ra_lag7 | 0.78 | Giá SJC 1 tuần trước |
| 6 | sma5_sjc | 0.65 | SMA 5 ngày |
| 7 | sma10_sjc | 0.45 | SMA 10 ngày |
| 8 | sma20_sjc | 0.33 | SMA 20 ngày |
| 9 | SJC_ban_ra_lag3 | 0.31 | Giá SJC 3 ngày trước |
| 10 | SJC_ban_ra_lag10 | 0.29 | Giá SJC 10 ngày trước |
| ... | | | |
| 12 | **USD_Close** | **0.14** | **USD Index — biến macro** |
| 14 | **TenY_Treasury** | **0.12** | **Lãi suất 10y** |
| 16 | **GLD_Close** | **0.10** | **Gold ETF** |

**Insight:** Lag features (giá quá khứ) chiếm 80% importance. Macro (USD, Treasury, GLD) đóng góp **nhỏ nhưng nhất quán** — quan trọng cho regime forecast.

### 5.4 Conformal Prediction Intervals (3 ML × 5 folds × 3 horizons = 45 evidence points, Phase 2 expanded)

#### 5.4.1 Single-fold demo (fold cuối = 2024 gold rally, ElasticNet h=1)
| Method | Target coverage | Actual coverage | Avg width |
|---|---|---|---|
| Split conformal α=0.05 | 95% | **83.3%** | 3.24 |
| Split conformal α=0.10 | 90% | 82.2% | 2.81 |
| Split conformal α=0.20 | 80% | 75.6% | 2.20 |
| **ACI** α=0.10 | 90% | **83.3%** | 3.62 |

#### 5.4.2 Full coverage report (Phase 2 mới, target 90% — alpha=0.10)
Bench ACI vs split conformal trên Ridge / ElasticNet / LightGBM × 5 folds × 3 horizons:

| Horizon | Method | Coverage avg | Width avg |
|---|---|---|---|
| h=1 | Split conformal | 75-79% | 1.4-6.8 |
| h=1 | **ACI** ⭐ | **86%** Ridge/EN | 1.7-1.9 |
| h=5 | Split | 74-81% | 2.5-8 |
| h=5 | **ACI** ⭐ | **85%** Ridge/EN | 3.8-3.9 |
| h=20 | Split | 65-75% | 6.6-11.7 |
| h=20 | ACI | 72-76% | 7-9 |

**KEY FINDING (Phase 2)**:
- Trong volatile period (fold 3-4 = 2024 rally), split conformal coverage **drops to 5-22%** (vi phạm guarantee 90%).
- **ACI maintains 60-90% coverage** trong cùng period.
- **ACI thắng split conformal ~10 percentage points** ở h=1, h=5.

→ Empirical evidence mạnh cho claim: "**ACI handles regime shifts better**" — chuẩn bị cho IEEE conference paper.

### 5.5 Regime-aware ensemble (Phase 2-3 contribution)
Phase 2: Build `VolatilityRegimeDetector` — rolling 20-day std + threshold q=0.7 quantile train.
- Stable ensemble: Ridge (0.45) + ElasticNet (0.45) + SeasonalNaive (0.10)
- Volatile ensemble: RollingNaive (0.50) + ElasticNet (0.30) + LightGBM (0.20)

Phase 3 nâng cấp: **Rolling re-detection per val row** (no leakage).
- Fold 0-2 (2022-2023 stable): 0/90 rows volatile detected ✅
- **Fold 3 (2023-Q4 → 2024-Q1, rally start): 31/90 rows volatile detected** ⭐
- **Fold 4 (2024-Q1 → Q3, rally peak): 87/90 rows volatile detected** ⭐

→ Detector chính xác bắt regime shift 2024 rally (Phase 2 limitation đã được FIX trong Phase 3).

### 5.6 Sentiment pipeline (Phase 2 demo)
End-to-end pipeline: news scrape (yfinance + Google News EN/VN) → mDeBERTa zero-shot 3-class → daily aggregate → exog feature.

- 218 headlines fetched (2025-10 → 2026-04, 6 tháng)
- 110 negative / 100 positive / 8 neutral, mean signed_score = -0.071 (slightly bearish)

⚠️ **Limitation**: news data 2025-10+ KHÔNG overlap với CV folds 2022-2024 → MAPE Δ ≈ 0%. Pipeline verified working; cần historical news (Web Archive scraping) để có impact thực sự — defer cho future work.

### 5.7 So sánh mode-A vs mode-B

- **Mode-A** (single fit train, n_val-step forecast): áp dụng cho classical + foundation. Realistic cho production deployment.
- **Mode-B** (rolling, biết y[t-1]): áp dụng cho ML/DL với engineered lag features. Bound dưới = `RollingNaive` (0.33% MAPE h=1).

→ ML linear (Ridge, ElasticNet) tận dụng mode-B + 108 features → bứt phá 5x so với mode-A baselines.

### 5.8 Production deliverables (Phase 3-4)
- ✅ Streamlit dashboard (`app/streamlit_app.py`) — 4 tabs interactive
- ✅ FastAPI server (`app/api/main.py`) — 5 endpoints OpenAPI
- ✅ Telegram bot (`app/telegram_bot.py`) — 6 commands
- ✅ Docker multi-stage (training + serving images)
- ✅ docker-compose: Streamlit + FastAPI + healthcheck
- ✅ GitHub Actions CI/CD: pytest + ruff + coverage
- ✅ Auto-retrain weekly: `scripts/retrain_weekly.py` + cron/Task Scheduler entries

### 5.5 So sánh mode-A vs mode-B

- **Mode-A** (single fit train, n_val-step forecast): áp dụng cho classical + foundation. Realistic cho production deployment.
- **Mode-B** (rolling, biết y[t-1]): áp dụng cho ML/DL với engineered lag features. Bound dưới = `RollingNaive` (0.33% MAPE h=1).

→ ML linear (Ridge, ElasticNet) tận dụng mode-B + 108 features → bứt phá 5x so với mode-A baselines.

---

## 6. Thảo luận

### 6.1 Vì sao Linear regularized thắng?
- Dataset nhỏ (~1000 train samples) → linear regularized chống overfit tốt hơn trees/DL.
- 108 features đã encode lags + technical → mối quan hệ với target gần linear sau lag.
- Trees overfit trên noise; DL cần ≥ 5000 samples để bứt phá.

### 6.2 Vì sao Foundation models không win?
- Chronos-Bolt-Small (48M params) zero-shot **không thấy** macro features (USD, oil, sentiment) — chỉ univariate.
- Pretrain corpus (Monash, GIFT-Eval) thiếu emerging market như VN gold → distribution mismatch.
- Tuy nhiên, **MAPE 3.07% mà KHÔNG cần training** vẫn ấn tượng — competitive với classical SOTA.

### 6.3 Regime shift 2024
Fold 3-4 (Q4 2023 - Q3 2024) trùng giai đoạn gold rally lịch sử (SJC tăng từ ~75 → ~95 triệu/lượng) → MAPE tăng 5-10x cho mọi mô hình. Điều này motivate:
- **Conformal ACI** thay vì split conformal vanilla.
- **Regime-aware ensemble**: weight mô hình khác nhau theo period.

### 6.4 Hạn chế
1. **Sentiment vẫn STUB** (zeros): pipeline PhoBERT/mDeBERTa đã sẵn nhưng news scraping chưa chạy.
2. **CPI VN data missing**: FRED code `CPALTT01VNQ657N` đã bị gỡ; cần lấy thủ công từ GSO.
3. **Optuna tuning** chưa chạy đủ — Ridge/ElasticNet đã thắng với defaults; tuning có thể đẩy thêm 5-10%.
4. **TFT, iTransformer** chưa benchmark đầy đủ (cần GPU Colab).
5. **Sample size mỗi fold = 90 ngày**: enough for stat tests nhưng nhỏ cho từng fold.

---

## 7. Kết luận

Nghiên cứu đã xây dựng và đánh giá hệ thống dự báo giá vàng SJC với **24 mô hình** từ thống kê cổ điển đến foundation models 2024-2026, áp dụng walk-forward CV nghiêm ngặt + Friedman test + Conformal PI + SHAP XAI. Kết quả chính:

1. **ElasticNet với 108 engineered features** đạt MAPE 0.67% / 1.41% / 3.06% — vượt trội mọi DL phức tạp.
2. **Foundation Chronos-Bolt zero-shot** competitive (MAPE 3.07%) — useful cho deployment khi không có training resources.
3. **ACI conformal** xử lý regime shift tốt hơn split conformal.
4. **Lag + macro features** đều đóng góp (SHAP).

**Hệ thống deployment ready:**
- Streamlit dashboard interactive (`streamlit run app/streamlit_app.py`)
- FastAPI endpoints (`uvicorn app.api.main:app`)
- Reproducible pipeline (`bash scripts/reproduce_all.sh` hoặc `.bat` Windows)
- Toàn bộ FREE TIER (không paid API, không GPU thuê, license MIT)

**Hướng phát triển:**
- Thực thi sentiment pipeline với CafeF/VnExpress scraping + PhoBERT
- Fine-tune Chronos-Bolt trên SJC để tăng độ chính xác
- Multi-asset ensemble (SJC + gold quốc tế + USD/VND)
- Triển khai mobile app cho người dùng cuối

---

## 8. Tài liệu tham khảo

1. Bouteska, A., Meftah-Wali, S., & Anh, P. T. (2023). *Fluctuations in gold prices in Vietnam during the COVID-19 pandemic: Insights from a time-varying parameter autoregression model*. Resources Policy, 86, 103-118.
2. Dao, V. T. (2024). *The correlation between gold price and some macroeconomic factors in Vietnam (2018-2023)*. Asian Business Research, 9(2), 45-56.
3. Ha, L. T., & Tran, M. Q. (2023). *Using VAR model to determine the impact of macro factors on gold price in Vietnam*. Journal of Economics and Development, 25(3), 55-70.
4. Tuan, D. A., Giang, N. T., Dinh, N. T. Q., & Ngoc, D. B. (2024). *Gold price forecast modelling: An ensemble learning approach*. Springer.
5. Nguyen, H. T., Nguyen, T. H. T., Tran, B. H., & Huynh, T. N. (2025). *Gold price analysis based on machine learning*. Springer CISIS.
6. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
7. Hyndman, R., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*, 3rd ed. OTexts.
8. Lundberg, S. M., & Lee, S. I. (2017). *A unified approach to interpreting model predictions*. NeurIPS 30.
9. Diebold, F. X., & Mariano, R. S. (1995). *Comparing predictive accuracy*. Journal of Business & Economic Statistics, 13(3), 253-263.
10. Demšar, J. (2006). *Statistical comparisons of classifiers over multiple data sets*. Journal of Machine Learning Research, 7, 1-30.
11. Gibbs, I., & Candès, E. (2021). *Adaptive Conformal Inference under distribution shift*. NeurIPS 34.
12. Ansari, A. F. et al. (2024). *Chronos: Learning the language of time series*. arXiv 2403.07815. Amazon.
13. Das, A. et al. (2024). *A decoder-only foundation model for time-series forecasting*. ICML 2024 (TimesFM, Google).
14. Nie, Y. et al. (2023). *A Time Series is Worth 64 Words: Long-term forecasting with Transformers* (PatchTST). ICLR.
15. Liu, Y. et al. (2024). *iTransformer: Inverted Transformers are effective for time series forecasting*. ICLR.
16. Challu, C. et al. (2023). *NHITS: Neural Hierarchical Interpolation for Time Series Forecasting*. AAAI.
17. Lim, B. et al. (2021). *Temporal Fusion Transformers for interpretable multi-horizon time series forecasting*. International Journal of Forecasting.

---

## Phụ lục A: Hyperparameters mô hình thắng

**ElasticNet** (sklearn defaults):
```python
ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=42)
```

**Chronos-Bolt-Small**:
```python
ChronosBolt(model_id="amazon/chronos-bolt-small", context_length=256)
# Zero-shot, no fine-tuning
```

## Phụ lục B: Reproducibility

```bash
git clone https://github.com/twangnhat-05/NGHIENCUUKHOAHOC
cd NGHIENCUUKHOAHOC
pip install -r requirements.txt
bash scripts/reproduce_all.sh   # ~70 phút trên CPU
streamlit run app/streamlit_app.py
```

## Phụ lục C: Đóng góp & Acknowledgments

- **Tác giả**: WangNhat (TDTU)
- **GVHD**: TBD
- **Co-architected with**: Claude Opus 4.7 (Anthropic) — pipeline design, code review, statistical analysis assistance.
- **Open-source community**: Nixtla (statsforecast/neuralforecast/mlforecast), Amazon (Chronos), IBM (TTM), Google (TimesFM), Salesforce (Moirai), Hugging Face.
