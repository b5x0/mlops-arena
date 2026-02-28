# ⚔️ MLOps Arena

> Built as part of the **AI Night Challenge by CTRL+S — 2026**

A full end-to-end MLOps system for CIFAR-10 image classification, featuring automated training pipelines, experiment tracking, data drift monitoring, and an arcade-themed gamification dashboard.

## 🎬 Demo

[![MLOps Arena Demo](https://img.youtube.com/vi/YKDUO4wPkdw/maxresdefault.jpg)](https://youtu.be/YKDUO4wPkdw)

▶️ **[Watch the full demo on YouTube](https://youtu.be/YKDUO4wPkdw)**

---

## 🏗️ Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | [ZenML](https://zenml.io) |
| Experiment Tracking | [MLflow](https://mlflow.org) |
| Artifact Store | [MinIO](https://min.io) (S3-compatible) |
| Model Registry | MLflow Model Registry |
| Drift Monitoring | [Evidently AI](https://evidentlyai.com) |
| Dashboard | [Streamlit](https://streamlit.io) |
| ML Framework | TensorFlow / Keras |
| Data Versioning | DVC |
| Infrastructure | Docker Compose |

---

## 🚀 Phases

### Phase 1 — Model Optimization
- CNN with **3 Conv blocks + Dense + Dropout** trained on CIFAR-10
- Data augmentation (flip, rotation, zoom)
- Hyperparameter tracking: learning rate, dropout rate, filter counts
- Model registered in MLflow Model Registry as `cifar10_classifier`

### Phase 2 — Drift Monitoring
- ZenML pipeline using Evidently `DataDriftPreset`
- Automated drift detection between reference and production data
- Pipeline runs are cached and replayable

### Phase 3 — Gamification Dashboard
- **HP Bar**: Model accuracy displayed as a fighter's health bar (🟢 >80% · 🟡 >50% · 🔴 <50%)
- **XP Counter**: `XP = accuracy × 1000 + 100`
- **Red Alert**: Flashing drift detection banner
- Arcade / Tech-Noir dark mode UI

---

## 🛠️ Quick Start

### 1. Start infrastructure
```bash
docker compose up -d
```

### 2. Activate environment & install deps
```bash
python -m venv zen_env
.\zen_env\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 3. Configure ZenML stack
```bash
python setup_zenml.py
```

### 4. Run Phase 1 — Training Pipeline
```bash
python run_pipeline.py
```

### 5. Run Phase 2 — Monitoring Pipeline
```bash
python monitoring_pipeline.py
```

### 6. Launch Arena Dashboard
```bash
streamlit run arena_dashboard.py
```

Open → **http://localhost:8501**

---

## 📊 Services

| Service | URL |
|---------|-----|
| Arena Dashboard | http://localhost:8501 |
| MLflow UI | http://localhost:5000 |
| ZenML UI | http://localhost:8080 |
| MinIO Console | http://localhost:9001 |

---

## 📁 Project Structure

```
mlops-arena/
├── pipeline.py              # Phase 1: ZenML training pipeline
├── run_pipeline.py          # Pipeline entrypoint
├── monitoring_pipeline.py   # Phase 2: Evidently drift monitoring
├── arena_dashboard.py       # Phase 3: Streamlit gamification dashboard
├── setup_zenml.py           # ZenML stack configuration
├── download_data.py         # CIFAR-10 data download
├── docker-compose.yml       # Infrastructure (Postgres, MinIO, MLflow, ZenML)
├── requirements.txt         # Python dependencies
├── data.dvc                 # DVC data pointer (actual data excluded from git)
└── .env                     # Local credentials (not committed)
```

---

## 🏆 Built for

**AI Night Challenge by CTRL+S — 2026**

---

*"Full MLOps pipeline — training, tracking, monitoring, and gamified observability — all in one repo, running with a single command."*