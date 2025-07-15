# app/streamlit_app.py
"""
Streamlit dashboard for Harvard Mental‑Health Prediction Project.
Reads engineered data directly from an Excel workbook.

Run with:
    streamlit run app/streamlit_app.py
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="College Mental‑Health Risk Dashboard",
    page_icon="🧠",
    layout="wide"
)

DATA_PATH  = "Harvard_Mental_Health_Prediction_Project.xlsx"
SHEET_NAME = "Feature Engineering"        # <— change if your sheet name differs


# --------------------------------------------------
# LOADERS (cached for speed)
# --------------------------------------------------
@st.cache_data(show_spinner=True)
def load_data(path: str, sheet: str) -> pd.DataFrame:
    """Load engineered Excel sheet."""
    return pd.read_excel(path, sheet_name=sheet)

@st.cache_resource(show_spinner=True)
def load_model(path: str):
    """Load trained Random‑Forest pipeline."""
    return joblib.load(path)

df    = load_data(DATA_PATH, SHEET_NAME)


# --------------------------------------------------
# COLUMN GROUPS (used later)
# --------------------------------------------------
NUM_COLS = [
    "Age", "GPA", "Study Hours Per Week", "Sleep Hours",
    "Stress Level", "Social Support Score",
    "Depression Score", "Anxiety Score", "Risk Score"
]

CAT_COLS = [
    "Gender", "Academic Year", "Sleep Category",
    "Stress Bin", "Study Intensity",
    "Therapy History", "Family History"
]

# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------
with st.sidebar:
    st.title("🧰 Filters")

    years = st.multiselect(
        "Academic Year",
        options=sorted(df["Academic Year"].unique()),
        default=sorted(df["Academic Year"].unique())
    )

    genders = st.multiselect(
        "Gender",
        options=sorted(df["Gender"].unique()),
        default=sorted(df["Gender"].unique())
    )

    stress_range = st.slider(
        "Stress Level (1–10)", 1, 10, (1, 10)
    )

    # Apply filters
    filtered = df[
        df["Academic Year"].isin(years) &
        df["Gender"].isin(genders) &
        df["Stress Level"].between(*stress_range)
    ]

# --------------------------------------------------
# MAIN TABS
# --------------------------------------------------
tab_explore, tab_single = st.tabs(["📊 Explore", "🔮 Predict Single Student"])

# ===== TAB 1: EXPLORE =========================================
with tab_explore:
    st.header("Cohort Overview")

    col1, col2, col3 = st.columns(3)
    total_students = len(filtered)
    at_risk_count  = (filtered["Is_AtRisk"] == "Yes").sum()
    at_risk_pct    = (at_risk_count / total_students * 100) if total_students else 0

    col1.metric("Total Students", total_students)
    col2.metric("At‑Risk Students", at_risk_count)
    col3.metric("At‑Risk %", f"{at_risk_pct:.1f}%")

    # ---------- Charts ----------
    st.subheader("Stress vs Sleep Scatter")
    fig, ax = plt.subplots()
    colors = (filtered["Is_AtRisk"] == "Yes").map({True: "red", False: "green"})
    ax.scatter(filtered["Sleep Hours"], filtered["Stress Level"], c=colors, alpha=0.7)
    ax.set_xlabel("Sleep Hours")
    ax.set_ylabel("Stress Level")
    st.pyplot(fig)

    st.subheader("Average Depression Score by Therapy History")
    bar_data = (filtered.groupby("Therapy History")["Depression Score"]
                          .mean().reindex(["No", "Yes"]))
    st.bar_chart(bar_data)

    st.subheader("Data Table (first 100 rows)")
    st.dataframe(filtered.head(100), use_container_width=True)

# ===== TAB 2: SINGLE-STUDENT PREDICTION =======================
with tab_single:
    st.header("Enter Student Details for Prediction")

    colA, colB, colC = st.columns(3)

    age   = colA.number_input("Age", 17, 30, 20)
    gpa   = colA.slider("GPA", 0.0, 4.0, 3.0, 0.01)
    study = colA.slider("Study Hours / Week", 0, 60, 20)

    sleep  = colB.slider("Sleep Hours (avg)", 0.0, 10.0, 6.0, 0.1)
    stress = colB.slider("Stress Level (1‑10)", 1, 10, 5)
    social = colB.slider("Social Support (1‑10)", 1, 10, 5)

    depression = colC.slider("Depression Score", 0, 27, 5)
    anxiety    = colC.slider("Anxiety Score", 0, 21, 5)

    gender        = st.selectbox("Gender", sorted(df["Gender"].unique()))
    year          = st.selectbox("Academic Year", sorted(df["Academic Year"].unique()))
    therapy_hist  = st.selectbox("Therapy History", ["No", "Yes"])
    family_hist   = st.selectbox("Family History of MH Issues", ["No", "Yes"])

    # Derived features
    sleep_cat = "<5" if sleep < 5 else "5-7" if sleep <= 7 else ">7"
    stress_bin = "Low" if stress <= 3 else "Med" if stress <= 6 else "High"
    study_intensity = ("Light" if study <= 15 else
                       "Moderate" if study <= 30 else "Heavy")
    risk_score = sum([
        sleep < 5,
        stress >= 8,
        depression >= 10,
        anxiety >= 11
    ])

    # Build input row
    sample = pd.DataFrame([{
        "Age": age,
        "GPA": gpa,
        "Study Hours Per Week": study,
        "Sleep Hours": sleep,
        "Stress Level": stress,
        "Social Support Score": social,
        "Depression Score": depression,
        "Anxiety Score": anxiety,
        "Risk Score": risk_score,
        "Gender": gender,
        "Academic Year": year,
        "Sleep Category": sleep_cat,
        "Stress Bin": stress_bin,
        "Study Intensity": study_intensity,
        "Therapy History": therapy_hist,
        "Family History": family_hist,
    }])

    if st.button("Predict Risk"):
        prob = model.predict_proba(sample)[0, 1]
        pred_flag = prob >= 0.5
        st.success(
            f"Risk probability: **{prob:.2%}** → "
            f"{'⚠️ At Risk' if pred_flag else '✅ Not at Immediate Risk'}"
        )

        st.write("Input features:")
        st.json(sample.to_dict(orient="records")[0])

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    """
    ---
    *Built with ❤️ by Aayush Tiwari – Excel → Python → Streamlit*  
    """
)