"""Streamlit app - Student Pass/Fail predictor (demo on synthetic data).

Run:  streamlit run app.py
Loads the saved scikit-learn pipeline (preprocessing + model) and predicts
from raw form inputs. No retraining happens here.
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "models" / "student_pass_fail_pipeline.joblib"
FEATURES = ["hours_studied", "attendance", "practice_tests", "sleep_hours", "study_method"]
STUDY_METHODS = ["self", "group", "video"]
LABELS = {0: "Fail", 1: "Pass"}


@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)


def predict(pipeline, records: pd.DataFrame) -> pd.DataFrame:
    """Return the records with predicted label and pass probability."""
    X = records[FEATURES]
    out = records.copy()
    out["predicted_label"] = [LABELS[int(c)] for c in pipeline.predict(X)]
    out["pass_probability"] = pipeline.predict_proba(X)[:, 1].round(3)
    return out


st.set_page_config(page_title="Student Pass/Fail Predictor", page_icon="🎓")
st.title("🎓 Student Pass/Fail Predictor")
st.warning(
    "**Demo only - synthetic classroom data.** This prediction must not be used to "
    "judge, grade, rank or penalize real students."
)

try:
    pipeline = load_pipeline()
except FileNotFoundError:
    st.error(f"Saved pipeline not found at `{MODEL_PATH}`. Run the notebook first to create it.")
    st.stop()

tab_single, tab_batch = st.tabs(["Single student", "Batch (CSV)"])

with tab_single:
    st.subheader("Enter study information (known before the final exam)")
    with st.form("student_form"):
        c1, c2 = st.columns(2)
        hours = c1.number_input("Hours studied", 0.0, 15.0, 5.0, 0.1)
        attendance = c1.slider("Attendance (%)", 0, 100, 75)
        tests = c2.number_input("Practice tests taken", 0, 15, 3, 1)
        sleep = c2.number_input("Average sleep (hours)", 0.0, 14.0, 6.5, 0.1)
        method = st.selectbox("Study method", STUDY_METHODS)
        submitted = st.form_submit_button("Predict")

    if submitted:
        row = pd.DataFrame([{
            "hours_studied": hours, "attendance": attendance,
            "practice_tests": tests, "sleep_hours": sleep, "study_method": method,
        }])
        result = predict(pipeline, row).iloc[0]
        prob = float(result["pass_probability"])
        if result["predicted_label"] == "Pass":
            st.success(f"Predicted outcome: **PASS**  (probability of passing: {prob:.0%})")
        else:
            st.error(f"Predicted outcome: **FAIL**  (probability of passing: {prob:.0%})")
        st.progress(prob)
        if 0.35 <= prob <= 0.65:
            st.info("The model is uncertain for this profile (probability close to 50%).")
        st.caption("Probabilities come from a small model trained on 224 synthetic students; "
                   "treat them as rough illustrations, not exact chances.")

with tab_batch:
    st.subheader("Predict many students from a CSV")
    st.caption(f"Required columns: {', '.join(FEATURES)}. Blank cells are imputed by the pipeline.")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded is not None:
        data = pd.read_csv(uploaded)
        missing = [c for c in FEATURES if c not in data.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            scored = predict(pipeline, data)
            st.dataframe(scored)
            st.download_button("Download predictions", scored.to_csv(index=False),
                               "predictions.csv", "text/csv")
