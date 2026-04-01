import re
from typing import Dict, Any

import requests
import streamlit as st


def parse_prometheus_text(text: str) -> dict:
    out: dict = {}

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        m_plain = re.match(
            r"^([a-zA-Z_:][a-zA-Z0-9_:]*)\s+([-+]?\d+(\.\d+)?([eE][-+]?\d+)?)$",
            line,
        )
        if m_plain:
            key = m_plain.group(1)
            val = float(m_plain.group(2))
            out[key] = val
            continue

        m_labeled = re.match(
            r"^([a-zA-Z_:][a-zA-Z0-9_:]*)\{([^}]*)\}\s+([-+]?\d+(\.\d+)?([eE][-+]?\d+)?)$",
            line,
        )
        if m_labeled:
            metric = m_labeled.group(1)
            labels = m_labeled.group(2)
            val = float(m_labeled.group(3))
            out.setdefault(metric, []).append((labels, val))

    return out


def pick_labeled_value(
    items: list[tuple[str, float]], needle: str, default: float = 0.0
) -> float:
    for labels, val in items:
        if needle in labels:
            return val
    return default


def get_safe_sample() -> Dict[str, Any]:
    return {
        "X1": 200000.0,
        "X2": 2,
        "X3": 2,
        "X4": 1,
        "X5": 35.0,
        "X6": 0,
        "X7": 0,
        "X8": 0,
        "X9": 0,
        "X10": 0,
        "X11": 0,
        "X12": 20000.0,
        "X13": 18000.0,
        "X14": 15000.0,
        "X15": 12000.0,
        "X16": 10000.0,
        "X17": 9000.0,
        "X18": 5000.0,
        "X19": 4000.0,
        "X20": 3500.0,
        "X21": 3000.0,
        "X22": 3000.0,
        "X23": 2500.0,
    }


def get_risky_sample() -> Dict[str, Any]:
    return {
        "X1": 20000.0,
        "X2": 1,
        "X3": 3,
        "X4": 1,
        "X5": 25.0,
        "X6": 3,
        "X7": 4,
        "X8": 5,
        "X9": 3,
        "X10": 2,
        "X11": 2,
        "X12": 500000.0,
        "X13": 450000.0,
        "X14": 420000.0,
        "X15": 400000.0,
        "X16": 380000.0,
        "X17": 350000.0,
        "X18": 0.0,
        "X19": 0.0,
        "X20": 0.0,
        "X21": 0.0,
        "X22": 0.0,
        "X23": 0.0,
    }


def init_session_state():
    if "form_data" not in st.session_state:
        st.session_state.form_data = get_safe_sample()


def render_predict_form() -> Dict[str, Any]:
    data = st.session_state.form_data

    st.subheader("Predict Form")

    col1, col2, col3 = st.columns(3)

    with col1:
        x1 = st.number_input("X1 - LIMIT_BAL", value=float(data["X1"]), step=1000.0)
        x2 = st.number_input("X2 - SEX", value=int(data["X2"]), step=1)
        x3 = st.number_input("X3 - EDUCATION", value=int(data["X3"]), step=1)
        x4 = st.number_input("X4 - MARRIAGE", value=int(data["X4"]), step=1)
        x5 = st.number_input("X5 - AGE", value=float(data["X5"]), step=1.0)
        x6 = st.number_input("X6 - PAY_0", value=int(data["X6"]), step=1)
        x7 = st.number_input("X7 - PAY_2", value=int(data["X7"]), step=1)
        x8 = st.number_input("X8 - PAY_3", value=int(data["X8"]), step=1)

    with col2:
        x9 = st.number_input("X9 - PAY_4", value=int(data["X9"]), step=1)
        x10 = st.number_input("X10 - PAY_5", value=int(data["X10"]), step=1)
        x11 = st.number_input("X11 - PAY_6", value=int(data["X11"]), step=1)
        x12 = st.number_input("X12 - BILL_AMT1", value=float(data["X12"]), step=1000.0)
        x13 = st.number_input("X13 - BILL_AMT2", value=float(data["X13"]), step=1000.0)
        x14 = st.number_input("X14 - BILL_AMT3", value=float(data["X14"]), step=1000.0)
        x15 = st.number_input("X15 - BILL_AMT4", value=float(data["X15"]), step=1000.0)
        x16 = st.number_input("X16 - BILL_AMT5", value=float(data["X16"]), step=1000.0)

    with col3:
        x17 = st.number_input("X17 - BILL_AMT6", value=float(data["X17"]), step=1000.0)
        x18 = st.number_input("X18 - PAY_AMT1", value=float(data["X18"]), step=500.0)
        x19 = st.number_input("X19 - PAY_AMT2", value=float(data["X19"]), step=500.0)
        x20 = st.number_input("X20 - PAY_AMT3", value=float(data["X20"]), step=500.0)
        x21 = st.number_input("X21 - PAY_AMT4", value=float(data["X21"]), step=500.0)
        x22 = st.number_input("X22 - PAY_AMT5", value=float(data["X22"]), step=500.0)
        x23 = st.number_input("X23 - PAY_AMT6", value=float(data["X23"]), step=500.0)

    payload = {
        "X1": float(x1),
        "X2": int(x2),
        "X3": int(x3),
        "X4": int(x4),
        "X5": float(x5),
        "X6": int(x6),
        "X7": int(x7),
        "X8": int(x8),
        "X9": int(x9),
        "X10": int(x10),
        "X11": int(x11),
        "X12": float(x12),
        "X13": float(x13),
        "X14": float(x14),
        "X15": float(x15),
        "X16": float(x16),
        "X17": float(x17),
        "X18": float(x18),
        "X19": float(x19),
        "X20": float(x20),
        "X21": float(x21),
        "X22": float(x22),
        "X23": float(x23),
    }

    st.session_state.form_data = payload
    return payload


st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")
st.title("Credit Risk Dashboard")

init_session_state()

api_url = st.text_input("API URL", "http://api:8000")

top_col1, top_col2, top_col3 = st.columns(3)

with top_col1:
    if st.button("Health Check"):
        r = requests.get(f"{api_url}/health", timeout=5)
        st.write("Status:", r.status_code)
        st.json(r.json())

with top_col2:
    if st.button("Fetch Metrics"):
        r = requests.get(f"{api_url}/metrics", timeout=5)
        st.write("Status:", r.status_code)

        raw = r.text
        metrics = parse_prometheus_text(raw)

        total_req = metrics.get("pred_requests_total", 0.0)
        risky = metrics.get("pred_risky_total", 0.0)
        count = metrics.get("pred_latency_ms_count", 0.0)
        total_ms = metrics.get("pred_latency_ms_sum", 0.0)
        avg_ms = (total_ms / count) if count else 0.0

        resp_items = metrics.get("pred_responses_total", [])
        resp_200 = (
            pick_labeled_value(resp_items, 'status_code="200"', 0.0)
            if isinstance(resp_items, list)
            else 0.0
        )
        resp_422 = (
            pick_labeled_value(resp_items, 'status_code="422"', 0.0)
            if isinstance(resp_items, list)
            else 0.0
        )

        st.subheader("Metrics Summary")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total /predict", int(total_req))
        c2.metric("Risky (pred=1)", int(risky))
        c3.metric("Avg latency (ms)", round(avg_ms, 2))
        c4.metric("200 OK", int(resp_200))
        c5.metric("422 Errors", int(resp_422))

        with st.expander("Raw /metrics", expanded=False):
            st.code(raw[:5000])

with top_col3:
    if st.button("Fetch Model Info"):
        r = requests.get(f"{api_url}/model/info", timeout=5)
        st.write("Status:", r.status_code)
        st.json(r.json())

st.markdown("---")

sample_col1, sample_col2 = st.columns(2)

with sample_col1:
    if st.button("Load Sample Safe User"):
        st.session_state.form_data = get_safe_sample()
        st.rerun()

with sample_col2:
    if st.button("Load Sample Risky User"):
        st.session_state.form_data = get_risky_sample()
        st.rerun()

payload = render_predict_form()

st.markdown("---")

predict_col1, predict_col2 = st.columns([1, 2])

with predict_col1:
    mode = st.selectbox("Prediction Mode", ["recall80", "maxf1", "fixed_0_5"], index=0)

    if st.button("Run Prediction", type="primary"):
        try:
            r = requests.post(
                f"{api_url}/predict",
                params={"mode": mode},
                json=payload,
                timeout=10,
            )

            if r.status_code != 200:
                st.error(f"Prediction failed. Status={r.status_code}")
                try:
                    st.json(r.json())
                except Exception:
                    st.text(r.text)
            else:
                result = r.json()
                st.session_state.last_prediction = result
                st.success("Prediction completed.")
        except Exception as e:
            st.error(f"Request error: {e}")

with predict_col2:
    st.subheader("Prediction Result")

    result = st.session_state.get("last_prediction")
    if result:
        proba = float(result["proba_default"])
        pred = int(result["prediction"])
        threshold = float(result["threshold"])
        used_mode = result["mode"]

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Probability of Default", f"{proba:.4f}")
        r2.metric("Prediction", "RISKY" if pred == 1 else "SAFE")
        r3.metric("Threshold", f"{threshold:.4f}")
        r4.metric("Mode", used_mode)

        if pred == 1:
            st.error("This customer is predicted as HIGH RISK.")
        else:
            st.success("This customer is predicted as LOWER RISK.")
    else:
        st.info("Run a prediction to see results here.")

    st.markdown("---")
    st.subheader("Why this prediction? (SHAP)")

    if st.button("Explain Prediction"):
        try:
            r = requests.post(
                f"{api_url}/explain/predict/grouped",
                params={"mode": mode},
                json=payload,
                timeout=15,
            )

            if r.status_code != 200:
                st.error(f"Explanation failed. Status={r.status_code}")
                try:
                    st.json(r.json())
                except Exception:
                    st.text(r.text)
            else:
                st.session_state.last_explanation = r.json()
                st.success("Explanation loaded.")
        except Exception as e:
            st.error(f"Explanation request error: {e}")

    explanation = st.session_state.get("last_explanation")

    if explanation:
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Explained Prediction", explanation.get("prediction"))
        e2.metric(
            "Explained Probability", f"{float(explanation.get('proba_default', 0)):.4f}"
        )
        e3.metric("Threshold", f"{float(explanation.get('threshold', 0)):.4f}")
        e4.metric("Model", explanation.get("model_type", "-"))

        st.markdown(" Top Positive Drivers (increase default risk)")
        pos = explanation.get("top_positive_drivers", [])
        if pos:
            for item in pos:
                st.write(f"**{item['feature']}**: {item['shap_value']:.4f}")
        else:
            st.info("No positive drivers found.")

        st.markdown(" Top Negative Drivers (decrease default risk)")
        neg = explanation.get("top_negative_drivers", [])
        if neg:
            for item in neg:
                st.write(f"**{item['feature']}**: {item['shap_value']:.4f}")
        else:
            st.info("No negative drivers found.")

        st.markdown("Top Features Overall")
        overall = explanation.get("top_features_overall", [])
        if overall:
            for item in overall[:10]:
                st.write(
                    f"**{item['feature']}** | shap={item['shap_value']:.4f} | abs={item['abs_shap_value']:.4f}"
                )
        else:
            st.info("No feature contributions found.")

st.markdown("---")

drift_col1, drift_col2 = st.columns(2)

with drift_col1:
    st.subheader("Drift Summary")
    if st.button("Fetch Drift Summary"):
        try:
            r = requests.get(f"{api_url}/drift/summary", timeout=10)
            st.write("Status:", r.status_code)
            st.json(r.json())
        except Exception as e:
            st.error(f"Drift summary error: {e}")

with drift_col2:
    st.subheader("System Snapshot")
    if st.button("Fetch Health Snapshot"):
        try:
            r = requests.get(f"{api_url}/health", timeout=10)
            st.write("Status:", r.status_code)
            st.json(r.json())
        except Exception as e:
            st.error(f"Health fetch error: {e}")

st.markdown("---")
st.subheader("Model Comparison")

if st.button("Fetch Model Comparison"):
    try:
        compare = requests.get(f"{api_url}/models/compare", timeout=5).json()
        registry = requests.get(f"{api_url}/registry/status", timeout=5).json()

        latest = compare.get("latest", {})
        champion = compare.get("champion", {})

        latest_version = latest.get("model_version")
        champion_version = champion.get("model_version")
        active_version = compare.get("active_model_version")
        select_mode = compare.get("model_select_mode")

        latest_score = latest.get("test_pr_auc_recall80")
        champion_score = champion.get("test_pr_auc_recall80")
        improvement = compare.get("improvement")
        decision = compare.get("decision")

        top1, top2, top3, top4 = st.columns(4)
        top1.metric("Active Model", active_version if active_version else "-")
        top2.metric("Select Mode", select_mode if select_mode else "-")
        top3.metric("Latest Model", latest_version if latest_version else "-")
        top4.metric("Champion Model", champion_version if champion_version else "-")

        st.markdown("PR-AUC Comparison (test / recall80)")
        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Latest PR-AUC",
            f"{latest_score:.4f}" if latest_score is not None else "N/A",
        )
        c2.metric(
            "Champion PR-AUC",
            f"{champion_score:.4f}" if champion_score is not None else "N/A",
        )
        c3.metric(
            "Improvement", f"{improvement:.4f}" if improvement is not None else "N/A"
        )

        st.markdown(" Registry Details")
        st.json(registry)

        if decision == "promote":
            st.success("Decision: promote")
        elif decision == "keep_champion":
            st.warning("Decision: keep_champion")
        else:
            st.info(f"Decision: {decision}")

        if latest_version == champion_version:
            st.success("Latest model is already champion.")
        else:
            st.info("Latest model is different from champion.")

    except Exception as e:
        st.error(f"Model comparison error: {e}")

st.markdown("---")
st.subheader("Feature Importance")

if st.button("Fetch Feature Importance"):
    try:
        r = requests.get(f"{api_url}/explain/feature_importance", timeout=10)
        st.write("Status:", r.status_code)

        if r.status_code != 200:
            st.json(r.json())
        else:
            data = r.json()
            st.write("Model Type:", data["model_type"])

            items = data["feature_importance"][:10]
            for item in items:
                st.write(f"{item['feature']}: {item['importance']:.4f}")
    except Exception as e:
        st.error(f"Feature importance error: {e}")
