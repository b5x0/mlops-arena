"""
MLOps Arena Dashboard - Phase 3 Gamification
Arcade/Tech-Noir themed Streamlit dashboard
"""
import os
import streamlit as st
import mlflow
import time

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MLOps Arena",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# ARCADE CSS INJECTION
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=Orbitron:wght@400;700;900&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f;
    color: #e0e0ff;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at top, #0d0d2b 0%, #0a0a0f 70%);
}

h1, h2, h3 {
    font-family: 'Press Start 2P', monospace !important;
}

.title-glow {
    font-family: 'Press Start 2P', monospace;
    font-size: 2rem;
    color: #00ffff;
    text-shadow: 0 0 10px #00ffff, 0 0 30px #00ffff, 0 0 60px #0088ff;
    text-align: center;
    padding: 1rem 0;
    letter-spacing: 4px;
}

.subtitle {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.85rem;
    color: #7878cc;
    text-align: center;
    margin-bottom: 2rem;
    letter-spacing: 2px;
}

.stat-box {
    background: linear-gradient(135deg, #0d0d2b, #1a1a3e);
    border: 1px solid #2a2a6a;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 0 20px #0044ff22, inset 0 0 20px #00001a;
    text-align: center;
}

.stat-label {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.7rem;
    color: #6666aa;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.stat-value {
    font-family: 'Press Start 2P', monospace;
    font-size: 2rem;
    color: #00ffcc;
    text-shadow: 0 0 10px #00ffcc;
}

.xp-value {
    font-family: 'Press Start 2P', monospace;
    font-size: 2.2rem;
    color: #ffcc00;
    text-shadow: 0 0 10px #ffcc00, 0 0 25px #ff8800;
}

/* HP Bar */
.hp-bar-wrapper {
    background: #1a1a3e;
    border: 2px solid #2a2a6a;
    border-radius: 8px;
    padding: 4px;
    margin: 0.5rem 0;
}

.hp-bar-fill-green  { background: linear-gradient(90deg, #00ff44, #44ff00); box-shadow: 0 0 12px #00ff44; }
.hp-bar-fill-yellow { background: linear-gradient(90deg, #ffcc00, #ff8800); box-shadow: 0 0 12px #ffcc00; }
.hp-bar-fill-red    { background: linear-gradient(90deg, #ff2200, #ff6600); box-shadow: 0 0 12px #ff2200; }

.hp-bar-fill-green, .hp-bar-fill-yellow, .hp-bar-fill-red {
    height: 28px;
    border-radius: 4px;
    transition: width 0.5s ease;
}

.hp-label {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.65rem;
    color: #aaaadd;
    letter-spacing: 2px;
    margin-bottom: 4px;
}

/* DRIFT ALERT */
.drift-alert {
    background: linear-gradient(135deg, #220000, #440000);
    border: 3px solid #ff0000;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    animation: flash 1s infinite;
    box-shadow: 0 0 30px #ff0000, 0 0 60px #ff000055;
}

.drift-alert-text {
    font-family: 'Press Start 2P', monospace;
    font-size: 1rem;
    color: #ff4444;
    text-shadow: 0 0 10px #ff0000;
    line-height: 2;
}

@keyframes flash {
    0%, 100% { border-color: #ff0000; box-shadow: 0 0 30px #ff0000; }
    50%       { border-color: #ff6600; box-shadow: 0 0 60px #ff6600; }
}

/* STABLE */
.stable-box {
    background: linear-gradient(135deg, #001a00, #003300);
    border: 2px solid #00ff44;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 0 20px #00ff4422;
}

.stable-text {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.85rem;
    color: #00ff88;
    letter-spacing: 2px;
}

/* Divider */
hr { border-color: #2a2a6a !important; }

/* Streamlit metric overrides */
[data-testid="stMetricValue"] {
    font-family: 'Orbitron', sans-serif !important;
    color: #00ffcc !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────
st.markdown('<div class="title-glow">⚔️ MLOps ARENA ⚔️</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">REAL-TIME FIGHTER STATS — CIFAR-10 CLASSIFIER</div>', unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────
# SIDEBAR CONTROLS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Arena Controls")
    mlflow_uri = st.text_input("MLflow URI", value="http://localhost:5000")
    drift_override = st.toggle("🔴 Simulate Drift Alert", value=False)
    refresh = st.button("🔄 Refresh Stats")

# ─────────────────────────────────────────────
# MLFLOW DATA FETCH
# ─────────────────────────────────────────────
@st.cache_data(ttl=30)
def get_latest_metrics(uri: str):
    try:
        mlflow.set_tracking_uri(uri)
        # Try known experiment names
        experiment = None
        for name in ["cifar10_training_pipeline", "CIFAR-10", "Default", "default"]:
            experiment = mlflow.get_experiment_by_name(name)
            if experiment:
                break

        if experiment is None:
            # Fallback: get any experiment that has runs
            experiments = mlflow.search_experiments()
            for exp in experiments:
                experiment = exp
                break

        if experiment:
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1,
            )
            if not runs.empty:
                row = runs.iloc[0]
                # Try multiple possible metric names
                acc = (
                    row.get("metrics.val_accuracy")
                    or row.get("metrics.test_accuracy")
                    or row.get("metrics.accuracy")
                    or 0.0
                )
                drift = bool(row.get("metrics.data_drift_detected", False))
                run_id = row.get("run_id", "N/A")
                return float(acc), drift, run_id

    except Exception as e:
        return None, False, str(e)

    return 0.0, False, "no-run"

# ─────────────────────────────────────────────
# COMPUTE STATS
# ─────────────────────────────────────────────
if refresh:
    st.cache_data.clear()

with st.sidebar:
    st.markdown("---")
    sim_mode = st.selectbox("🏟️ Arena Simulation State", ["Live (MLflow)", "Mock: Optimal", "Mock: Degraded", "Mock: Drifted"], index=0)

with st.spinner("Connecting to MLflow..."):
    accuracy, drift_from_mlflow, run_id = get_latest_metrics(mlflow_uri)

# Logic for Cloud Demo or Mock Selection
if accuracy is None or "Mock" in sim_mode:
    if "Drifted" in sim_mode:
        st.info("☁️ **Cloud Simulation**: Environment COMPROMISED.")
        accuracy = 0.3240
        drift_from_mlflow = True
        run_id = "drift-alert-demo"
    elif "Degraded" in sim_mode:
        st.info("☁️ **Cloud Simulation**: Performance DEGRADED.")
        accuracy = 0.4820
        drift_from_mlflow = False
        run_id = "degraded-run-demo"
    else:
        st.info("☁️ **Cloud Demo Mode**: Optimal local run results.")
        accuracy = 0.5080
        drift_from_mlflow = False
        run_id = "c589d649-cloud-demo"

drift_detected = drift_override or drift_from_mlflow
xp = int(accuracy * 1000) + 100

# ─────────────────────────────────────────────
# HP BAR
# ─────────────────────────────────────────────
hp_pct = int(accuracy * 100)
if accuracy >= 0.8:
    hp_class, hp_emoji = "hp-bar-fill-green", "💚"
elif accuracy >= 0.5:
    hp_class, hp_emoji = "hp-bar-fill-yellow", "💛"
else:
    hp_class, hp_emoji = "hp-bar-fill-red", "❤️"

# ─────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-label">{hp_emoji} Fighter Health (Accuracy)</div>
        <div class="hp-label">HP: {hp_pct}%</div>
        <div class="hp-bar-wrapper">
            <div class="{hp_class}" style="width:{hp_pct}%;"></div>
        </div>
        <div class="stat-value" style="margin-top:1rem;">{accuracy:.2%}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-label">✨ Fighter XP</div>
        <div class="xp-value">{xp} XP</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-label">🧬 Run ID</div>
        <div style="font-family:monospace;color:#7878cc;font-size:0.7rem;word-break:break-all;">{str(run_id)[:16]}...</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
# DRIFT ALERT / STABLE
# ─────────────────────────────────────────────
if drift_detected:
    st.markdown("""
    <div class="drift-alert">
        <div class="drift-alert-text">
            ⚠️ RED ALERT ⚠️<br><br>
            DATA DRIFT DETECTED<br>
            RETRAINING REQUIRED<br><br>
            ⚠️ ARENA ENVIRONMENT COMPROMISED ⚠️
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="stable-box">
        <div class="stable-text">✅ ENVIRONMENT STABLE — FIGHTER PERFORMING OPTIMALLY ✅</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;font-family:'Orbitron',sans-serif;font-size:0.6rem;color:#333366;padding:1rem;">
    MLOps ARENA v1.0 · Powered by ZenML · MLflow · Evidently · TensorFlow
</div>
""", unsafe_allow_html=True)
