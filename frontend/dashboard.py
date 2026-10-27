"""
IndustrialAI — Streamlit Dashboard
7-page professional dashboard for the predictive maintenance platform.
"""
import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

logging.basicConfig(level=logging.WARNING)

# ── Config ────────────────────────────────────────────────────────────────────
import os
from dotenv import load_dotenv
load_dotenv()

# Resolve API base URL: Streamlit Cloud secrets → env var → local default
def _get_api_base() -> str:
    try:
        # st.secrets is available on Streamlit Cloud or with a local secrets.toml
        url = st.secrets.get("API_BASE_URL", "")
        if url:
            return url.rstrip("/")
    except Exception:
        pass
    return os.getenv("API_BASE_URL", "http://localhost:8000")

API_BASE = _get_api_base()
API_TIMEOUT = 30       # General endpoints: /predict, /anomaly, /explain, /history
RAG_TIMEOUT = 120      # /maintenance/ask: first call loads SentenceTransformer (~60s on cloud)
                       # Override via RAG_TIMEOUT env var
try:
    RAG_TIMEOUT = int(os.getenv("RAG_TIMEOUT", str(RAG_TIMEOUT)))
except ValueError:
    pass
REPORTS_DIR = Path(__file__).parent.parent / "reports"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

st.set_page_config(
    page_title="IndustrialAI — Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Theme ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #0d1117; color: #e6edf3; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid #30363d;
}

.metric-card {
    background: linear-gradient(135deg, #161b22, #1c2128);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover { transform: translateY(-2px); border-color: #58a6ff; }
.metric-value { font-size: 2rem; font-weight: 700; color: #58a6ff; line-height: 1.2; }
.metric-label { font-size: 0.8rem; color: #8b949e; margin-top: 6px; text-transform: uppercase; letter-spacing: 0.5px; }

.risk-HIGH { color: #f85149; font-weight: 700; }
.risk-MEDIUM-HIGH { color: #e3b341; font-weight: 600; }
.risk-MEDIUM { color: #d29922; font-weight: 600; }
.risk-LOW { color: #3fb950; font-weight: 600; }

.status-ok {
    background: #1a4a1e; color: #56d364;
    border: 1px solid #2ea043; border-radius: 8px; padding: 8px 16px;
    display: inline-block; font-weight: 600;
}
.status-error {
    background: #3d1a1a; color: #f85149;
    border: 1px solid #da3633; border-radius: 8px; padding: 8px 16px;
    display: inline-block; font-weight: 600;
}
.warning-box {
    background: #2d2316; border: 1px solid #e3b341; border-radius: 8px;
    padding: 12px 16px; color: #e3b341; font-size: 0.85rem; margin: 8px 0;
}
.info-box {
    background: #162032; border: 1px solid #58a6ff; border-radius: 8px;
    padding: 12px 16px; color: #79c0ff; font-size: 0.85rem; margin: 8px 0;
}
.chat-user {
    background: #1c2128; border-left: 3px solid #58a6ff;
    padding: 10px 14px; border-radius: 0 8px 8px 0; margin: 8px 0;
}
.chat-assistant {
    background: #161b22; border-left: 3px solid #3fb950;
    padding: 10px 14px; border-radius: 0 8px 8px 0; margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)


# ── API helpers ───────────────────────────────────────────────────────────────
_API_OFFLINE_SHOWN = False  # module-level flag to avoid spamming the banner


def _show_offline_banner():
    """Show a one-time demo-mode banner when the API is unreachable."""
    st.markdown(
        '<div class="warning-box">'
        '🌐 <strong>Demo Mode</strong> — The prediction API is not connected. '
        'Static pages (Overview metrics, Model Performance, Project Info) work fully. '
        'Interactive pages (Predict, Anomaly, Explain, GenAI) require the FastAPI backend. '
        'See README for deployment options.</div>',
        unsafe_allow_html=True,
    )


def api_get(endpoint: str) -> Optional[Dict]:
    try:
        r = requests.get(f"{API_BASE}{endpoint}", timeout=API_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        return None  # Handled gracefully by callers; banner shown in sidebar
    except Exception as e:
        st.error(f"❌ API error: {e}")
        return None


def api_post(endpoint: str, payload: Dict) -> Optional[Dict]:
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=API_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        _show_offline_banner()
        return None
    except requests.HTTPError as e:
        try:
            detail = r.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        st.error(f"❌ API error ({r.status_code}): {detail}")
        return None
    except Exception as e:
        st.error(f"❌ Request failed: {e}")
        return None


def api_post_rag(payload: dict):
    """
    POST to /maintenance/ask with RAG_TIMEOUT (default 120s) instead of
    the global API_TIMEOUT. The first call lazily loads SentenceTransformer
    (~300MB) on the cloud server which takes 30-60s; this longer timeout
    prevents the dashboard from giving up before the model is ready.
    """
    try:
        r = requests.post(
            f"{API_BASE}/maintenance/ask",
            json=payload,
            timeout=(10, RAG_TIMEOUT),  # (connect_timeout, read_timeout)
        )
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        _show_offline_banner()
        return None
    except requests.Timeout:
        st.warning(
            "⏱️ The GenAI assistant took too long to respond. "
            f"(timeout: {RAG_TIMEOUT}s) "
            "On first use the AI model loads into memory on the cloud server. "
            "**Please wait 30–60 seconds and try again** — subsequent questions will be faster."
        )
        return None
    except requests.HTTPError as e:
        try:
            detail = r.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        st.error(f"❌ Assistant error ({r.status_code}): {detail}")
        return None
    except Exception as e:
        st.error(f"❌ Request failed: {e}")
        return None


def api_warmup_rag() -> None:
    """Fire-and-forget GET to /maintenance/warmup to pre-load the embedding model."""
    try:
        requests.get(f"{API_BASE}/maintenance/warmup", timeout=5)
    except Exception:
        pass  # Warmup is best-effort; any failure is harmless


def check_api_health():
    return api_get("/health")


def load_test_metrics():
    path = REPORTS_DIR / "metrics.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def load_val_metrics():
    path = REPORTS_DIR / "val_metrics.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def load_raw_dataset():
    csv_files = list(RAW_DIR.glob("*.csv"))
    if csv_files:
        return pd.read_csv(csv_files[0])
    return None


def metric_card(value, label, color="#58a6ff"):
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value" style="color:{color}">{value}</div>'
        f'<div class="metric-label">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ IndustrialAI")
    st.markdown("*Predictive Maintenance Platform*")
    st.divider()

    health = check_api_health()
    if health:
        st.markdown(
            f'<span class="status-ok">✓ API Online</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"Model: `{health.get('model_version', 'N/A')}`")
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            st.caption("🤖 ML Model")
            st.markdown(
                "✅" if health.get("model_ready") else "⚠️ Not trained",
            )
        with col_h2:
            st.caption("🔍 Anomaly")
            st.markdown(
                "✅" if health.get("anomaly_ready") else "⚠️ Not trained"
            )
    else:
        st.markdown(
            '<span class="status-error">✗ API Offline</span>',
            unsafe_allow_html=True,
        )
        st.caption("Static pages still work. See README.")

    st.divider()

    page = st.selectbox(
        "Navigate",
        [
            "📊 Overview",
            "🔮 Machine Prediction",
            "📈 Model Performance",
            "🧠 Explainable AI",
            "🔍 Anomaly Monitoring",
            "💬 GenAI Assistant",
            "ℹ️ Project Info",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown(
        '<div class="warning-box">⚠️ <strong>Synthetic Data Only</strong><br>'
        'All predictions are from a model trained on the UCI AI4I 2020 <em>synthetic</em> dataset. '
        'Do not use for real operational decisions.</div>',
        unsafe_allow_html=True,
    )

# Store last prediction in session state
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "last_sensor" not in st.session_state:
    st.session_state.last_sensor = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 IndustrialAI — Overview")
    st.caption("Explainable Predictive Maintenance & Failure Analysis Platform")

    st.markdown(
        '<div class="info-box">ℹ️ This platform uses the UCI AI4I 2020 <strong>synthetic</strong> '
        'predictive maintenance dataset. All statistics reflect the training/test dataset, '
        'not real factory operations.</div>',
        unsafe_allow_html=True,
    )

    # Dataset stats
    df = load_raw_dataset()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card(f"{len(df):,}" if df is not None else "N/A", "Total Records")
    with c2:
        if df is not None and "Machine failure" in df.columns:
            fails = int(df["Machine failure"].sum())
            metric_card(str(fails), "Failure Records", color="#f85149")
        else:
            metric_card("N/A", "Failure Records")
    with c3:
        if df is not None and "Machine failure" in df.columns:
            rate = f"{df['Machine failure'].mean():.1%}"
            metric_card(rate, "Failure Rate", color="#e3b341")
        else:
            metric_card("N/A", "Failure Rate")
    with c4:
        metrics = load_test_metrics()
        metric_card(
            f"{metrics['f1_score']:.3f}" if metrics else "—",
            "Best Model F1",
            color="#3fb950",
        )

    st.divider()

    # Last prediction panel
    col_pred, col_hist = st.columns([1, 2])

    with col_pred:
        st.subheader("Current Prediction")
        pred = st.session_state.last_prediction
        if pred:
            risk = pred.get("risk_level", "UNKNOWN")
            prob = pred.get("failure_probability", 0)
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value risk-{risk}">'
                f'{"🔴 FAILURE" if pred["predicted_failure"] else "🟢 NORMAL"}</div>'
                f'<div class="metric-label">Predicted Class</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%", "font": {"size": 28, "color": "#e6edf3"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#8b949e"},
                    "bar": {"color": "#f85149" if prob > 0.5 else "#e3b341" if prob > 0.3 else "#3fb950"},
                    "bgcolor": "#1c2128",
                    "steps": [
                        {"range": [0, 30], "color": "#1a4a1e"},
                        {"range": [30, 60], "color": "#2d2316"},
                        {"range": [60, 100], "color": "#3d1a1a"},
                    ],
                    "threshold": {
                        "line": {"color": "#58a6ff", "width": 2},
                        "thickness": 0.75,
                        "value": pred.get("decision_threshold", 0.5) * 100,
                    },
                },
                title={"text": "Failure Probability", "font": {"color": "#8b949e", "size": 13}},
            ))
            fig.update_layout(
                height=220, margin=dict(t=40, b=0, l=20, r=20),
                paper_bgcolor="#0d1117", font_color="#e6edf3",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Model: {pred.get('model_name', '—')} | Threshold: {pred.get('decision_threshold', '—')}")
        else:
            st.info("No prediction yet. Use **🔮 Machine Prediction** to run one.")

    with col_hist:
        st.subheader("Recent Prediction History")
        history_data = api_get("/history?limit=20")
        if history_data and history_data["records"]:
            records = history_data["records"]
            hist_df = pd.DataFrame([
                {
                    "Time": r["timestamp"][:19].replace("T", " ") if r.get("timestamp") else "",
                    "Failure": "🔴 YES" if r["predicted_failure"] else "🟢 NO",
                    "Probability": f"{r['failure_probability']:.3f}",
                    "Risk": r.get("risk_level", ""),
                    "Anomaly": "⚠️" if r.get("is_anomaly") else "✓",
                }
                for r in records
            ])
            st.dataframe(hist_df, use_container_width=True, height=280)
        else:
            st.info("No prediction history yet.")

    # Dataset distribution charts
    if df is not None and "Machine failure" in df.columns:
        st.divider()
        st.subheader("Dataset Exploration")
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            target_dist = df["Machine failure"].value_counts().reset_index()
            target_dist.columns = ["Class", "Count"]
            target_dist["Class"] = target_dist["Class"].map({0: "Normal", 1: "Failure"})
            fig = px.pie(
                target_dist, values="Count", names="Class",
                color="Class", color_discrete_map={"Normal": "#3fb950", "Failure": "#f85149"},
                title="Target Distribution",
            )
            fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font_color="#e6edf3")
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            if "Type" in df.columns:
                type_dist = df.groupby(["Type", "Machine failure"]).size().reset_index(name="Count")
                type_dist["Class"] = type_dist["Machine failure"].map({0: "Normal", 1: "Failure"})
                fig = px.bar(
                    type_dist, x="Type", y="Count", color="Class",
                    color_discrete_map={"Normal": "#3fb950", "Failure": "#f85149"},
                    title="Failures by Product Type", barmode="stack",
                )
                fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font_color="#e6edf3")
                st.plotly_chart(fig, use_container_width=True)

        with col_c:
            if "Tool wear [min]" in df.columns:
                fig = px.histogram(
                    df, x="Tool wear [min]", color="Machine failure",
                    color_discrete_map={0: "#3fb950", 1: "#f85149"},
                    nbins=40, title="Tool Wear Distribution",
                    labels={"Machine failure": "Failed"},
                )
                fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font_color="#e6edf3")
                st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: MACHINE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Machine Prediction":
    st.title("🔮 Machine Prediction")
    st.caption("Enter sensor readings to predict failure probability")

    if health and not health.get("model_ready"):
        st.error("⚠️ Model not trained. Run: `python -m src.train_pipeline`")

    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.subheader("Sensor Inputs")
        with st.form("prediction_form"):
            product_type = st.selectbox("Product Type", ["L", "M", "H"],
                help="L=Low, M=Medium, H=High grade product")
            air_temp = st.number_input(
                "Air Temperature [K]", min_value=250.0, max_value=400.0,
                value=300.0, step=0.5,
                help="Normal range: 295–305 K",
            )
            proc_temp = st.number_input(
                "Process Temperature [K]", min_value=250.0, max_value=400.0,
                value=310.0, step=0.5,
                help="Normal range: 304–315 K. Typically 8–10 K above air temp.",
            )
            rot_speed = st.number_input(
                "Rotational Speed [rpm]", min_value=0, max_value=10000,
                value=1500, step=10,
                help="Normal range: 1200–2800 rpm",
            )
            torque = st.number_input(
                "Torque [Nm]", min_value=0.0, max_value=500.0,
                value=40.0, step=0.5,
                help="Normal range: 20–65 Nm",
            )
            tool_wear = st.number_input(
                "Tool Wear [min]", min_value=0, max_value=1000,
                value=100, step=1,
                help="Cumulative tool use. Replace at ~250 min.",
            )

            col_b1, col_b2 = st.columns(2)
            with col_b1:
                predict_btn = st.form_submit_button("🔮 Predict", use_container_width=True, type="primary")
            with col_b2:
                anomaly_btn = st.form_submit_button("🔍 + Anomaly", use_container_width=True)

    # Build payload
    payload = {
        "Type": product_type,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": proc_temp,
        "Rotational speed [rpm]": rot_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
    }

    with col_result:
        st.subheader("Prediction Results")

        if predict_btn or anomaly_btn:
            with st.spinner("Running prediction..."):
                result = api_post("/predict", payload)

            if result:
                st.session_state.last_prediction = result
                st.session_state.last_sensor = payload

                risk = result.get("risk_level", "UNKNOWN")
                prob = result.get("failure_probability", 0)
                failed = result.get("predicted_failure", False)

                if failed:
                    st.markdown(
                        f'<div class="metric-card"><div class="metric-value risk-{risk}">'
                        f'🔴 FAILURE PREDICTED</div>'
                        f'<div class="metric-label">Risk Level: {risk}</div></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="metric-card"><div class="metric-value risk-LOW">'
                        f'🟢 NORMAL OPERATION</div>'
                        f'<div class="metric-label">Risk Level: {risk}</div></div>',
                        unsafe_allow_html=True,
                    )

                r1, r2, r3 = st.columns(3)
                with r1:
                    st.metric("Failure Probability", f"{prob:.1%}")
                with r2:
                    st.metric("Decision Threshold", f"{result.get('decision_threshold', 0):.3f}")
                with r3:
                    st.metric("Latency", f"{result.get('inference_latency_ms', 0):.1f} ms")

                if result.get("input_warnings"):
                    for w in result["input_warnings"]:
                        st.markdown(f'<div class="warning-box">⚠️ {w}</div>', unsafe_allow_html=True)

                st.caption(f"Model: {result.get('model_name', '—')} v{result.get('model_version', '—')}")
                st.caption(f"⚠️ {result.get('disclaimer', '')}")

                # Power calculation
                import math
                power_w = torque * (2 * math.pi * rot_speed / 60) if rot_speed > 0 else 0
                if power_w > 9000:
                    st.markdown(
                        f'<div class="warning-box">⚡ Calculated power: {power_w:.0f} W — '
                        f'exceeds 9 kW rated limit (PWF risk)</div>',
                        unsafe_allow_html=True,
                    )

            if anomaly_btn and result:
                with st.spinner("Running anomaly detection..."):
                    anomaly = api_post("/anomaly", payload)
                if anomaly:
                    st.divider()
                    st.subheader("Anomaly Detection")
                    if anomaly.get("is_anomaly"):
                        st.warning(f"⚠️ ANOMALY DETECTED — Score: {anomaly.get('anomaly_score_normalised', 0):.3f}")
                    else:
                        st.success(f"✅ Normal — Score: {anomaly.get('anomaly_score_normalised', 0):.3f}")
                    st.caption(anomaly.get("interpretation", ""))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    st.caption("Evaluation results from training and held-out test set")

    st.markdown(
        '<div class="info-box">ℹ️ Metrics are computed on the UCI AI4I 2020 <strong>synthetic</strong> dataset. '
        'Real-world performance will differ. Model selected by PR-AUC on validation set, '
        'not accuracy.</div>',
        unsafe_allow_html=True,
    )

    val_metrics = load_val_metrics()
    test_metrics = load_test_metrics()

    if not val_metrics and not test_metrics:
        st.warning("No metrics found. Run `python -m src.train_pipeline` first.")
    else:
        # Validation comparison table
        if val_metrics:
            st.subheader("🏆 Model Comparison (Validation Set)")
            st.caption(f"Best model: **{val_metrics.get('best_model', '—')}**")
            comparison = val_metrics.get("model_comparison", {})
            rows = []
            for name, m in comparison.items():
                rows.append({
                    "Model": name.replace("_", " ").title(),
                    "F1": f"{m['f1_score']:.4f}",
                    "Precision": f"{m['precision']:.4f}",
                    "Recall": f"{m['recall']:.4f}",
                    "PR-AUC": f"{m.get('pr_auc', 0):.4f}" if m.get("pr_auc") else "N/A",
                    "ROC-AUC": f"{m.get('roc_auc', 0):.4f}" if m.get("roc_auc") else "N/A",
                    "Threshold": f"{m.get('threshold', 0.5):.4f}",
                })
            st.dataframe(
                pd.DataFrame(rows).set_index("Model"),
                use_container_width=True,
            )

            # Visual comparison
            if len(rows) > 1:
                plot_data = []
                for name, m in comparison.items():
                    for metric in ["f1_score", "precision", "recall"]:
                        plot_data.append({
                            "Model": name.replace("_", " ").title(),
                            "Metric": metric.replace("_", " ").title(),
                            "Value": m.get(metric, 0),
                        })
                fig = px.bar(
                    pd.DataFrame(plot_data),
                    x="Model", y="Value", color="Metric",
                    barmode="group", title="Model Comparison",
                    color_discrete_sequence=["#58a6ff", "#3fb950", "#e3b341"],
                )
                fig.update_layout(
                    paper_bgcolor="#0d1117", plot_bgcolor="#1c2128",
                    font_color="#e6edf3", yaxis_range=[0, 1],
                )
                st.plotly_chart(fig, use_container_width=True)

        # Test set metrics
        if test_metrics:
            st.subheader("🎯 Final Test Set Evaluation (Held-Out)")
            st.caption(f"Dataset: {test_metrics.get('dataset', '—')} | N={test_metrics.get('n_test', '—')}")

            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                metric_card(f"{test_metrics.get('f1_score', 0):.4f}", "F1 Score", "#3fb950")
            with c2:
                metric_card(f"{test_metrics.get('precision', 0):.4f}", "Precision", "#58a6ff")
            with c3:
                metric_card(f"{test_metrics.get('recall', 0):.4f}", "Recall", "#e3b341")
            with c4:
                metric_card(
                    f"{test_metrics.get('pr_auc', 0):.4f}" if test_metrics.get("pr_auc") else "N/A",
                    "PR-AUC", "#d29922"
                )
            with c5:
                metric_card(
                    f"{test_metrics.get('roc_auc', 0):.4f}" if test_metrics.get("roc_auc") else "N/A",
                    "ROC-AUC", "#bc8cff"
                )

            c_a, c_b = st.columns(2)
            with c_a:
                metric_card(str(test_metrics.get("false_positives", "N/A")), "False Positives", "#e3b341")
            with c_b:
                metric_card(str(test_metrics.get("false_negatives", "N/A")), "False Negatives", "#f85149")

            # Confusion matrix
            cm = test_metrics.get("confusion_matrix")
            if cm:
                st.subheader("Confusion Matrix")
                cm_arr = np.array(cm)
                fig = px.imshow(
                    cm_arr,
                    labels={"x": "Predicted", "y": "Actual", "color": "Count"},
                    x=["Normal (0)", "Failure (1)"],
                    y=["Normal (0)", "Failure (1)"],
                    text_auto=True,
                    color_continuous_scale="Blues",
                    title=f"Confusion Matrix — {test_metrics.get('model_name', '')}",
                )
                fig.update_layout(
                    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                    font_color="#e6edf3", height=350,
                )
                st.plotly_chart(fig, use_container_width=True)

            # Limitations
            st.subheader("⚠️ Model Limitations")
            st.markdown("""
- Trained on **synthetic** data that may not capture real factory failure patterns
- Class imbalance (~3.4% failure rate) makes rare failures harder to detect
- Threshold tuned on validation data — adjust for your operational tolerance of false alarms
- Model does not account for seasonal trends, maintenance history, or equipment ageing
- Performance on different machine types or cutting conditions may vary significantly
            """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: EXPLAINABLE AI
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Explainable AI":
    st.title("🧠 Explainable AI")
    st.caption("SHAP feature contributions for model transparency")

    st.markdown(
        '<div class="warning-box">⚠️ SHAP values explain <strong>model behaviour</strong>, '
        'not physical causation. A high SHAP value for Torque means the model weighted '
        'that reading heavily — it does not prove torque caused the failure.</div>',
        unsafe_allow_html=True,
    )

    # Pre-warm the embedding model in the background.
    # Returns immediately (5s connect timeout); the server loads the model async.
    api_warmup_rag()

    pred = st.session_state.last_prediction
    sensor = st.session_state.last_sensor

    if not pred:
        st.info("Run a prediction first on **🔮 Machine Prediction** page.")
    else:
        st.success(f"Explaining prediction: {'🔴 FAILURE' if pred['predicted_failure'] else '🟢 NORMAL'} (prob={pred['failure_probability']:.3f})")

        with st.spinner("Computing SHAP explanations..."):
            explain_result = api_post("/explain", sensor)

        if explain_result:
            contribs = explain_result.get("feature_contributions", {})

            if contribs:
                # Waterfall-style bar chart
                names = list(contribs.keys())
                values = list(contribs.values())
                colors = ["#f85149" if v > 0 else "#3fb950" for v in values]

                fig = go.Figure(go.Bar(
                    x=values,
                    y=names,
                    orientation="h",
                    marker_color=colors,
                    text=[f"{v:.4f}" for v in values],
                    textposition="outside",
                ))
                fig.update_layout(
                    title="SHAP Feature Contributions (Local Explanation)",
                    xaxis_title="SHAP Value (impact on prediction)",
                    paper_bgcolor="#0d1117", plot_bgcolor="#1c2128",
                    font_color="#e6edf3", height=400,
                    xaxis={"zeroline": True, "zerolinecolor": "#8b949e", "zerolinewidth": 2},
                )
                st.plotly_chart(fig, use_container_width=True)

            # Summary
            if explain_result.get("summary"):
                st.subheader("Plain-Language Summary")
                st.text(explain_result["summary"])

            st.caption(f"Method: {explain_result.get('method', 'N/A')}")
            st.caption(explain_result.get("disclaimer", ""))

        # Global importance
        st.subheader("Global Feature Importance")
        st.caption("Average SHAP |value| across training samples")
        st.info(
            "Global importance is computed from the training dataset. "
            "Navigate to 🔮 Machine Prediction to first get a local explanation."
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: ANOMALY MONITORING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Anomaly Monitoring":
    st.title("🔍 Anomaly Monitoring")
    st.caption("Isolation Forest anomaly detection across sensor readings")

    st.markdown(
        '<div class="info-box">ℹ️ Anomaly detection is <strong>independent</strong> from failure '
        'prediction. An anomaly means sensor readings deviate from normal patterns — it does '
        '<strong>not</strong> confirm a failure.</div>',
        unsafe_allow_html=True,
    )

    if health and not health.get("anomaly_ready"):
        st.error("Anomaly detector not trained. Run: `python -m src.train_pipeline`")

    # Historical prediction history with anomaly info
    st.subheader("Recent Predictions with Anomaly Status")
    history_data = api_get("/history?limit=100")

    if history_data and history_data["records"]:
        records = history_data["records"]
        df_hist = pd.DataFrame([
            {
                "Timestamp": r.get("timestamp", "")[:19].replace("T", " "),
                "Predicted Failure": "🔴 YES" if r["predicted_failure"] else "🟢 NO",
                "Probability": round(r.get("failure_probability", 0), 4),
                "Risk": r.get("risk_level", ""),
                "Anomaly": "⚠️ YES" if r.get("is_anomaly") else "✓ NO",
                "Anomaly Score": round(r.get("anomaly_score") or 0, 4),
            }
            for r in records
        ])

        # Filters
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            show_anomalies = st.checkbox("Show only anomalies", value=False)
        with col_f2:
            show_failures = st.checkbox("Show only failures", value=False)

        display_df = df_hist.copy()
        if show_anomalies:
            display_df = display_df[display_df["Anomaly"] == "⚠️ YES"]
        if show_failures:
            display_df = display_df[display_df["Predicted Failure"] == "🔴 YES"]

        st.dataframe(display_df, use_container_width=True)

        # Download
        csv = display_df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv,
            file_name="prediction_history.csv",
            mime="text/csv",
        )

        # Scatter plot: probability vs anomaly score
        if len(df_hist) > 1:
            st.subheader("Failure Probability vs Anomaly Score")
            scatter_df = pd.DataFrame([
                {
                    "Failure Prob": r.get("failure_probability", 0),
                    "Anomaly Score": r.get("anomaly_score") or 0,
                    "Failure": "Failure" if r["predicted_failure"] else "Normal",
                    "Risk": r.get("risk_level", ""),
                }
                for r in records
            ])
            fig = px.scatter(
                scatter_df, x="Failure Prob", y="Anomaly Score",
                color="Failure",
                color_discrete_map={"Failure": "#f85149", "Normal": "#3fb950"},
                title="Failure Probability vs Anomaly Score",
                labels={"Failure Prob": "Failure Probability", "Anomaly Score": "Anomaly Score (normalised)"},
            )
            fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#1c2128", font_color="#e6edf3")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No prediction history yet. Make some predictions first.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6: GENAI ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💬 GenAI Assistant":
    st.title("💬 GenAI Maintenance Assistant")
    st.caption("RAG-based assistant with Sentence Transformers + Chroma vector database")

    st.markdown(
        '<div class="info-box">ℹ️ Answers are grounded in maintenance documentation retrieved from '
        'the vector database. Sources are cited where available. '
        'LLM is optional — the assistant returns retrieved passages even without one.</div>',
        unsafe_allow_html=True,
    )

    pred = st.session_state.last_prediction
    sensor = st.session_state.last_sensor

    if pred:
        with st.expander("Current Prediction Context (passed to assistant)", expanded=False):
            st.json({
                "predicted_failure": pred.get("predicted_failure"),
                "failure_probability": pred.get("failure_probability"),
                "risk_level": pred.get("risk_level"),
                "model": pred.get("model_name"),
            })
            if sensor:
                st.json(sensor)

    # Chat interface
    for entry in st.session_state.chat_history:
        st.markdown(
            f'<div class="chat-user"><strong>You:</strong> {entry["question"]}</div>',
            unsafe_allow_html=True,
        )
        answer_text = entry["answer"]
        st.markdown(
            f'<div class="chat-assistant"><strong>Assistant:</strong> {answer_text}</div>',
            unsafe_allow_html=True,
        )
        if entry.get("sources"):
            st.caption(f"📄 Sources: {', '.join(entry['sources'])}")
        if not entry.get("llm_available"):
            st.caption("⚠️ LLM offline — showing retrieved passages only")
        st.caption(f"⏱ {entry.get('latency_ms', 0):.0f} ms")
        st.divider()

    # Suggested questions
    st.subheader("Ask a Question")
    suggestions = [
        "What does a high torque reading indicate?",
        "When should I replace the tool?",
        "What causes heat dissipation failure?",
        "How should I interpret the failure probability?",
        "What are the normal operating temperature ranges?",
    ]
    with st.expander("💡 Suggested Questions"):
        for s in suggestions:
            if st.button(s, key=f"sug_{s[:20]}"):
                st.session_state["pending_question"] = s

    with st.form("chat_form", clear_on_submit=True):
        question = st.text_input(
            "Your question:",
            value=st.session_state.pop("pending_question", ""),
            placeholder="Ask about sensor readings, failure modes, maintenance procedures...",
        )
        send_btn = st.form_submit_button("Send ↵", type="primary", use_container_width=True)

    if send_btn and question.strip():
        with st.spinner(
            "\U0001f9e0 Generating answer\u2026 "
            "(First use may take 30\u201360 s while the AI model loads on the cloud server)"
        ):
            result = api_post_rag(
                {
                    "question": question,
                    "prediction_context": pred,
                    "sensor_context": sensor,
                },
            )
        if result:
            st.session_state.chat_history.append({
                "question": question,
                "answer": result.get("answer", "No answer"),
                "sources": result.get("sources", []),
                "llm_available": result.get("llm_available", False),
                "latency_ms": result.get("latency_ms", 0),
            })
            st.rerun()

    if st.button("🗑️ Clear Chat", type="secondary"):
        st.session_state.chat_history = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7: PROJECT INFO
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ Project Info":
    st.title("ℹ️ IndustrialAI — Project Information")

    st.markdown("""
## Project Overview
**IndustrialAI** is an end-to-end industrial predictive maintenance platform combining:
- **Machine Learning** for failure risk prediction
- **Anomaly Detection** for pattern deviation alerts
- **Explainable AI** (SHAP) for transparent model insights
- **Retrieval-Augmented Generation** for AI-powered maintenance assistance

## Dataset
- **Source:** [UCI AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
- **Size:** ~10,000 synthetic industrial machine records
- **Features:** Type, Air Temp, Process Temp, Rotational Speed, Torque, Tool Wear
- **Target:** Machine failure (binary)
- **Important:** This is a **synthetic** dataset — results do not establish real-world factory performance

## Architecture
```
UCI AI4I Dataset → Data Validation → Preprocessing (fit on train only)
                                           ↓
              Logistic Regression ─┐
              Random Forest       ─┼→ Best Model (PR-AUC) → FastAPI
              XGBoost             ─┘
                                           ↓
              Isolation Forest → Anomaly Detector → FastAPI
                                           ↓
              SHAP Explainer → Feature Contributions → FastAPI
                                           ↓
              Sentence Transformers + Chroma → RAG → LLM → FastAPI
                                           ↓
                              Streamlit Dashboard
```

## Technology Stack
| Component | Technology |
|---|---|
| Language | Python 3.11 |
| ML Models | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Anomaly Detection | Isolation Forest |
| Embeddings | Sentence Transformers |
| Vector DB | ChromaDB |
| LLM | OpenRouter / Ollama |
| API | FastAPI + Pydantic |
| Database | SQLite (SQLAlchemy) |
| Dashboard | Streamlit + Plotly |

## Evaluation Methodology
1. Stratified train/val/test split (70/15/15)
2. Preprocessor fitted only on training data
3. Class weighting for imbalanced data (~3.4% failures)
4. Model selection by PR-AUC on validation set (preferred over accuracy for imbalanced classes)
5. Decision threshold tuned on validation set
6. Final metrics reported on untouched test set

## Limitations
- **Synthetic data:** Real factory failure patterns may differ significantly
- **Imbalanced classes:** False negatives are harder to eliminate at high precision
- **LLM dependence:** RAG assistant requires external API key or local Ollama setup
- **Training data scope:** Model trained on a single synthetic dataset without domain transfer
- **No real-time data pipeline:** Designed for batch/on-demand prediction, not continuous streaming

## Ethical Considerations
- This system should be used as **decision support**, not autonomous control
- All predictions carry uncertainty — human engineering judgment is mandatory
- Do not share prediction data externally without appropriate data governance
- Model should be re-evaluated periodically as equipment or processes change
""")

    st.subheader("🚀 Getting Started")
    st.code("""
# 1. Setup
git clone <repo>
cd IndustrialAI
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure (optional — for GenAI features)
cp .env.example .env
# Edit .env with your OpenRouter key or Ollama config

# 3. Train models
python -m src.train_pipeline

# 4. Start API (terminal 1)
uvicorn src.api:app --reload --port 8000

# 5. Start Dashboard (terminal 2)
streamlit run frontend/dashboard.py

# 6. Run tests
pytest tests/ -v
    """, language="bash")
