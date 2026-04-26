# 🚀 Deploy Guide — Streamlit Cloud + FastAPI (Render)

> **TDTU NCKH 2025-2026 Gold Forecasting Project**
> Hướng dẫn deploy MIỄN PHÍ — không cần card tín dụng, không cần GPU.

---

## A. Deploy Streamlit Cloud (5-10 phút)

### Yêu cầu
- ✅ Repo GitHub đã có code (đã xong: `twangnhat-05/NGHIENCUUKHOAHOC`)
- ✅ Branch `main` (hoặc `claude/auto-execution`) đã push
- ✅ File `app/streamlit_app.py` (đã có)
- ✅ File `requirements.txt` (đã có, pinned)
- ❌ Cần: tài khoản Streamlit Cloud (đăng ký free bằng GitHub)

### Bước 1 — Đăng ký Streamlit Cloud
1. Vào https://share.streamlit.io
2. Click **"Continue with GitHub"** → authorize Streamlit truy cập repos của bạn
3. Confirm email (nếu chưa)

### Bước 2 — Trước khi deploy: chuẩn bị repo

**Quan trọng**: branch `main` hiện vẫn rỗng. Bạn cần MERGE branch `claude/auto-execution` vào `main` trước:

```bash
cd D:\WangNhat\Study\NCKH
git checkout main
git merge claude/auto-execution --no-ff -m "merge: full project from claude/auto-execution"
git push origin main
```

Hoặc nếu muốn test trước, có thể deploy thẳng từ branch `claude/auto-execution`.

### Bước 3 — Tạo Streamlit App

1. Vào dashboard Streamlit Cloud → click **"New app"**
2. Form điền:
   - **Repository**: `twangnhat-05/NGHIENCUUKHOAHOC`
   - **Branch**: `main` (hoặc `claude/auto-execution` nếu chưa merge)
   - **Main file path**: `app/streamlit_app.py`
   - **App URL**: tùy chỉnh, ví dụ `gold-sjc-tdtu` → URL sẽ là `https://gold-sjc-tdtu.streamlit.app`
3. **Python version**: Streamlit Cloud sẽ auto-detect từ `requirements.txt` (cần Python 3.11). Nếu app fail, vào Settings → Advanced settings → set Python version = `3.11`
4. Click **"Deploy!"**

### Bước 4 — Chờ build (5-15 phút)

- Streamlit Cloud sẽ:
  1. Clone repo
  2. `pip install -r requirements.txt` (~5-10 phút vì có torch, prophet, neuralforecast)
  3. Start `streamlit run app/streamlit_app.py`
- Theo dõi build log realtime trong dashboard

### Bước 5 — Lỗi thường gặp & fix

**❌ Build fails: "out of memory" khi cài torch/prophet**
- Streamlit Cloud free tier có 1GB RAM build → có thể fail với heavy deps
- **Fix**: tạo file `requirements-streamlit.txt` chỉ với deps tối thiểu cho dashboard:
  ```
  pandas==2.2.3
  numpy==1.26.4
  pyarrow==17.0.0
  pyyaml==6.0.2
  scikit-learn==1.5.2
  matplotlib==3.9.2
  seaborn==0.13.2
  plotly==5.24.1
  streamlit==1.56.0
  ```
- Vào Streamlit settings → "Custom requirements file" → set thành `requirements-streamlit.txt`
- Lưu ý: Predictions tab (cần ElasticNet) vẫn hoạt động vì sklearn

**❌ App load nhưng không hiện data**
- File `data/processed/features_v2_with_sentiment.parquet` cần được **commit + push** lên GitHub
- Verify: `git ls-files data/processed/`

**❌ "Quota exceeded"**
- Streamlit Cloud free tier: 1 app private, không giới hạn public apps
- App sleep sau 7 ngày không có traffic → tự wake up khi visit lại

### Bước 6 — Custom domain (optional)
- Streamlit Cloud free tier KHÔNG hỗ trợ custom domain
- Workaround: dùng Cloudflare Worker proxy (vẫn miễn phí)

---

## B. Deploy FastAPI lên Render (15 phút)

### Yêu cầu
- Repo GitHub đã có
- File `app/api/main.py` (đã có)

### Bước 1 — Tạo `render.yaml` ở root repo

```yaml
services:
  - type: web
    name: gold-sjc-api
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app.api.main:app --host 0.0.0.0 --port $PORT
    plan: free
    envVars:
      - key: PYTHON_VERSION
        value: 3.11
```

### Bước 2 — Đăng ký Render

1. Vào https://render.com → "Get Started" với GitHub
2. Click **"New +"** → **"Web Service"**
3. Connect repo `twangnhat-05/NGHIENCUUKHOAHOC`
4. Render sẽ auto-detect `render.yaml`
5. Click **"Create Web Service"**

### Bước 3 — Chờ build (~10 phút)

- URL sẽ có dạng: `https://gold-sjc-api.onrender.com`
- Endpoints:
  - `GET /` — health
  - `GET /predict?h=1` — forecast
  - `GET /docs` — Swagger UI
  - `GET /leaderboard?h=1` — top models

### ⚠️ Render free tier limitations
- **Sleep sau 15 phút không request** → cold start ~30s lần đầu
- **750 giờ/tháng** = 31 days × 24h = 744 → đủ cho 1 service chạy 24/7
- Bandwidth: 100GB/month
- **Khắc phục cold start**: dùng Cron-job.org ping `/` mỗi 14 phút

---

## C. Deploy lên Hugging Face Spaces (alternative cho Streamlit)

### Tại sao chọn HF Spaces?
- Free tier mạnh hơn Streamlit Cloud (16GB RAM, 2 vCPU)
- Hỗ trợ Streamlit/Gradio/FastAPI native
- Có thể dùng GPU free (cho foundation models)

### Bước 1 — Tạo Space

1. Vào https://huggingface.co → đăng ký
2. Click **"+ New"** → **"Space"**
3. Form:
   - **Owner**: username của bạn
   - **Space name**: `gold-sjc-tdtu`
   - **License**: MIT
   - **SDK**: Streamlit
   - **Hardware**: CPU basic (free)
   - **Visibility**: Public
4. Click **"Create Space"**

### Bước 2 — Push code lên HF Space

```bash
# HF Spaces là 1 git repo riêng
git remote add huggingface https://huggingface.co/spaces/<your-username>/gold-sjc-tdtu
git push huggingface claude/auto-execution:main

# Cần cài git-lfs nếu có file lớn (foundation model weights)
git lfs install
git lfs track "*.parquet"
git add .gitattributes
```

### Bước 3 — Tạo file `app.py` ở root (HF tìm file này)

Tạo symlink hoặc copy:
```bash
cp app/streamlit_app.py app.py
git add app.py && git commit -m "add app.py for HF Spaces" && git push huggingface
```

### Bước 4 — Chờ build (~10-15 phút)

URL: `https://huggingface.co/spaces/<your-username>/gold-sjc-tdtu`

---

## D. Quick comparison

| Platform | Free RAM | Build time | Sleep? | Custom domain | Best for |
|---|---|---|---|---|---|
| **Streamlit Cloud** | 1GB | 5-10 min | 7 days idle | ❌ | Quick demo |
| **Render** (FastAPI) | 512MB | ~10 min | 15 min idle | ✅ paid | API endpoints |
| **HF Spaces** (Streamlit) | 16GB | 10-15 min | 48h idle | ❌ | Heavy ML demos |
| **GitHub Pages** | n/a | n/a | n/a | ✅ free | Static (chỉ docs/notebook HTML) |

**Recommendation cho TDTU NCKH**:
- 🥇 **Streamlit Cloud** → demo nhanh cho hội đồng
- 🥈 **HF Spaces** nếu muốn nhúng cả foundation models
- 🥉 Render cho FastAPI nếu cần API public

---

## E. Sau khi deploy thành công

### Test smoke
```bash
# Streamlit app
curl -I https://<your-app>.streamlit.app
# → HTTP/2 200

# FastAPI
curl https://gold-sjc-api.onrender.com/predict?h=1
# → JSON response
```

### Update README badge
```markdown
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://<your-app>.streamlit.app)
[![API Docs](https://img.shields.io/badge/API-docs-blue)](https://gold-sjc-api.onrender.com/docs)
```

### Share cho hội đồng TDTU
```
🌐 Demo dashboard: https://<your-app>.streamlit.app
📊 API endpoints:  https://gold-sjc-api.onrender.com/docs
📁 Source code:   https://github.com/twangnhat-05/NGHIENCUUKHOAHOC
📄 Báo cáo:       reports/paper/tdtu_vi/report.docx
```

---

## F. Troubleshooting

| Lỗi | Nguyên nhân | Fix |
|---|---|---|
| "ModuleNotFoundError: src" | PYTHONPATH thiếu | `streamlit_app.py` đã thêm `sys.path.insert(0, project_root)` — kiểm tra đoạn này |
| "FileNotFoundError: features_v2.parquet" | File không có trong repo | `git add data/processed/features_v2_with_sentiment.parquet && git commit && git push` |
| "torch installation failed" | RAM/disk overflow trên free tier | Dùng `requirements-streamlit.txt` minimal — bỏ torch/prophet/neuralforecast |
| Build > 30 phút | Quá nhiều deps | Giảm bớt — chỉ giữ deps cho demo dashboard |
| App nhưng "loading..." mãi | Trang đầu chạy ML nặng | Dùng `@st.cache_data(ttl=3600)` — đã làm trong `streamlit_app.py` |

---

## G. Tài liệu chính thức

- Streamlit Cloud docs: https://docs.streamlit.io/streamlit-community-cloud
- Render docs: https://render.com/docs
- Hugging Face Spaces docs: https://huggingface.co/docs/hub/spaces
- Pandoc (DOCX conversion): https://pandoc.org/

---

🤖 *Generated by Claude Opus 4.7 — TDTU NCKH 2026*
