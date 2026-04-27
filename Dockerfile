# Multi-stage Dockerfile cho gold price forecasting project
# Stage 1: training (torch + foundation models, ~3GB)
# Stage 2: serving (Streamlit + FastAPI minimal, ~500MB)
#
# Build:
#   docker build -t gold-sjc:training --target training .
#   docker build -t gold-sjc:serving --target serving .
# Run:
#   docker run -p 8501:8501 gold-sjc:serving streamlit run app/streamlit_app.py
#   docker run -p 8000:8000 gold-sjc:serving uvicorn app.api.main:app --host 0.0.0.0

# ====================
# STAGE 1: training
# ====================
FROM python:3.11-slim AS training

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps in cache-friendly order
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy source
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY pyproject.toml .

# Default: shell for training jobs
CMD ["python", "-c", "print('Gold SJC training image — run scripts/reproduce_all.sh'); import sys; sys.exit(0)"]


# ====================
# STAGE 2: serving (lighter)
# ====================
FROM python:3.11-slim AS serving

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install minimal serving deps (no torch/prophet/neuralforecast/foundation)
RUN pip install --upgrade pip && pip install \
    pandas==2.2.3 \
    numpy==1.26.4 \
    scipy==1.14.1 \
    pyarrow==17.0.0 \
    pyyaml==6.0.2 \
    scikit-learn==1.5.2 \
    xgboost==2.1.2 \
    lightgbm==4.5.0 \
    matplotlib==3.9.2 \
    seaborn==0.13.2 \
    plotly==5.24.1 \
    streamlit==1.40.1 \
    fastapi==0.115.4 \
    uvicorn==0.32.0

# Copy minimal source for serving
COPY src/ ./src/
COPY app/ ./app/
COPY data/processed/features_v2_with_sentiment.parquet ./data/processed/
COPY reports/leaderboard/combined_v2_summary.csv ./reports/leaderboard/
COPY reports/leaderboard/friedman_test.csv ./reports/leaderboard/
COPY reports/figures/shap_lightgbm_h1_top20.csv ./reports/figures/
COPY reports/figures/aci_conformal_elasticnet_h1.png ./reports/figures/
COPY pyproject.toml .

EXPOSE 8501 8000

# Default: Streamlit. Override với:
#   docker run -p 8000:8000 gold-sjc:serving uvicorn app.api.main:app --host 0.0.0.0
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.headless=true", "--server.port=8501", "--browser.gatherUsageStats=false"]
