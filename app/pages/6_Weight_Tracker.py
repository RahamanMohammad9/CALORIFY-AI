import os
import sqlite3
import sys
import json
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils import CHART_SURFACE, FAVICON_PATH, apply_glass_style, render_page_header

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Weight Tracker",
    page_icon=FAVICON_PATH,
    layout="wide"
)

apply_glass_style(st)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "meal_history.db")
PROFILE_PATH = os.path.join(BASE_DIR, "user_profile.json")

# -----------------------------
# Database helpers
# -----------------------------
def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def create_weight_table():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS weight_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            weight_kg REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()

def add_weight(weight_kg):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO weight_logs (weight_kg, created_at)
        VALUES (?, ?)
    """, (
        float(weight_kg),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()

def get_weight_history():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, weight_kg, created_at
        FROM weight_logs
        ORDER BY created_at DESC
    """)

    rows = cursor.fetchall()
    conn.close()
    return rows

def get_weight_history_ascending():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, weight_kg, created_at
        FROM weight_logs
        ORDER BY created_at ASC
    """)

    rows = cursor.fetchall()
    conn.close()
    return rows

def weight_log_exists(log_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM weight_logs WHERE id = ?", (int(log_id),))
    count = cursor.fetchone()[0]

    conn.close()
    return count > 0

def delete_weight_log(log_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM weight_logs WHERE id = ?", (int(log_id),))

    conn.commit()
    conn.close()

def load_profile():
    if os.path.exists(PROFILE_PATH):
        with open(PROFILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "height_cm": 170.0
    }

def calculate_bmi(weight_kg, height_cm):
    height_m = height_cm / 100
    if height_m <= 0:
        return 0
    return weight_kg / (height_m ** 2)

def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    return "Obese"

create_weight_table()
profile = load_profile()

render_page_header(
    st,
    "Weight Tracker",
    "Track your body weight changes and monitor your progress over time",
    kicker="Progress",
)

# -----------------------------
# Load data
# -----------------------------
weight_logs_desc = get_weight_history()
weight_logs_asc = get_weight_history_ascending()

current_weight = weight_logs_desc[0][1] if weight_logs_desc else 0
starting_weight = weight_logs_asc[0][1] if weight_logs_asc else 0
weight_change = current_weight - starting_weight if weight_logs_desc else 0

height_cm = float(profile.get("height_cm", 170.0))
current_bmi = calculate_bmi(current_weight, height_cm) if current_weight > 0 else 0
current_bmi_status = bmi_category(current_bmi) if current_weight > 0 else "N/A"

# -----------------------------
# Top section
# -----------------------------
top_col1, top_col2 = st.columns([1, 1.4])

with top_col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Current Weight Summary")

    if current_weight > 0:
        c1, c2 = st.columns(2)
        c3, c4 = st.columns(2)

        c1.metric("Current Weight", f"{current_weight:.1f} kg")
        c2.metric("Starting Weight", f"{starting_weight:.1f} kg")
        c3.metric("Change", f"{weight_change:+.1f} kg")
        c4.metric("BMI", f"{current_bmi:.1f}")

        if current_bmi_status == "Normal":
            st.success(f"BMI Category: {current_bmi_status}")
        elif current_bmi_status == "Underweight":
            st.warning(f"BMI Category: {current_bmi_status}")
        else:
            st.warning(f"BMI Category: {current_bmi_status}")
    else:
        st.info("No weight entries available yet.")
    st.markdown('</div>', unsafe_allow_html=True)

with top_col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Add Weight Entry")

    new_weight = st.number_input(
        "Enter your weight (kg)",
        min_value=20.0,
        max_value=300.0,
        value=70.0,
        step=0.1
    )

    if st.button("➕ Save Weight", width="stretch"):
        add_weight(new_weight)
        st.success(f"Saved weight: {new_weight:.1f} kg")
        st.rerun()

    st.write("**Tip:** Save your weight regularly to visualize long-term trends.")
    st.write(f"Using profile height: **{height_cm:.1f} cm**")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Trend graph
# -----------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Weight Trend")

if weight_logs_asc:
    weight_df = pd.DataFrame(weight_logs_asc, columns=["ID", "Weight", "Created At"])
    weight_df["Date"] = pd.to_datetime(weight_df["Created At"]).dt.date.astype(str)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=weight_df["Date"],
        y=weight_df["Weight"],
        mode="lines+markers",
        name="Weight",
        line=dict(color="#A855F7", width=4, shape="spline"),
        marker=dict(size=8, color="#A855F7", line=dict(width=2, color="#FFFFFF")),
        fill="tozeroy",
        fillcolor="rgba(168, 85, 247, 0.14)",
        hovertemplate="<b>Date:</b> %{x}<br><b>Weight:</b> %{y:.1f} kg<extra></extra>"
    ))

    fig.update_layout(
        title="Weight Over Time",
        template="plotly_dark",
        plot_bgcolor=CHART_SURFACE,
        paper_bgcolor=CHART_SURFACE,
        font=dict(color="white"),
        xaxis_title="Date",
        yaxis_title="Weight (kg)",
        margin=dict(l=20, r=20, t=50, b=20),
        height=420
    )

    st.plotly_chart(fig, width="stretch")
else:
    st.info("No weight history available yet.")

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Weight history table
# -----------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Weight History")

if weight_logs_desc:
    history_df = pd.DataFrame(weight_logs_desc, columns=["ID", "Weight (kg)", "Created At"])
    st.dataframe(history_df, width="stretch", height=320)

    st.markdown("### Delete Weight Entry")
    delete_id = st.number_input(
        "Enter Weight Log ID",
        min_value=1,
        step=1,
        key="delete_weight_id"
    )

    if st.button("🗑️ Delete Weight Entry", width="stretch"):
        if weight_log_exists(delete_id):
            delete_weight_log(delete_id)
            st.success(f"Weight log ID {delete_id} deleted.")
            st.rerun()
        else:
            st.warning("That Weight Log ID does not exist.")
else:
    st.info("No weight entries saved yet.")

st.markdown('</div>', unsafe_allow_html=True)