import os
import sqlite3
import sys
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
    page_title="Water Tracker",
    page_icon=FAVICON_PATH,
    layout="wide"
)

apply_glass_style(st)

# -----------------------------
# Database setup
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "meal_history.db")


def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def create_water_table():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS water_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            amount_ml REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def add_water(amount_ml):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO water_logs (amount_ml, created_at)
        VALUES (?, ?)
    """, (
        float(amount_ml),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()


def get_today_water_total():
    conn = get_connection()
    cursor = conn.cursor()

    today = datetime.now().strftime("%Y-%m-%d")
    cursor.execute("""
        SELECT COALESCE(SUM(amount_ml), 0)
        FROM water_logs
        WHERE substr(created_at, 1, 10) = ?
    """, (today,))

    total = cursor.fetchone()[0]
    conn.close()
    return total


def get_water_history():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, amount_ml, created_at
        FROM water_logs
        ORDER BY created_at DESC
    """)

    rows = cursor.fetchall()
    conn.close()
    return rows


def get_daily_water_history():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT substr(created_at, 1, 10) AS log_date,
               COALESCE(SUM(amount_ml), 0) AS total_ml
        FROM water_logs
        GROUP BY substr(created_at, 1, 10)
        ORDER BY log_date ASC
    """)

    rows = cursor.fetchall()
    conn.close()
    return rows


def water_log_exists(log_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM water_logs WHERE id = ?", (int(log_id),))
    count = cursor.fetchone()[0]

    conn.close()
    return count > 0


def delete_water_log(log_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM water_logs WHERE id = ?", (int(log_id),))

    conn.commit()
    conn.close()


create_water_table()

render_page_header(
    st,
    "Water Tracker",
    "Track your daily hydration and monitor water intake trends",
    kicker="Hydration",
)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Hydration Settings")
    water_goal_ml = st.number_input(
        "Daily water goal (ml)",
        min_value=500,
        max_value=6000,
        value=2500,
        step=100
    )

# -----------------------------
# Today's summary
# -----------------------------
today_water = get_today_water_total()
remaining = water_goal_ml - today_water
progress_value = min(today_water / water_goal_ml, 1.0) if water_goal_ml > 0 else 0
progress_percent = progress_value * 100

top_col1, top_col2 = st.columns([1, 1.4])

with top_col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    ring_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=today_water,
        number={"suffix": " ml", "font": {"size": 28, "color": "white"}},
        title={"text": "Today's Water Intake", "font": {"size": 20, "color": "white"}},
        gauge={
            "axis": {"range": [0, water_goal_ml], "tickcolor": "white"},
            "bar": {"color": "#38BDF8"},
            "bgcolor": "#1A1F2B",
            "borderwidth": 2,
            "bordercolor": "#444",
            "steps": [
                {"range": [0, water_goal_ml * 0.5], "color": "rgba(56, 189, 248, 0.18)"},
                {"range": [water_goal_ml * 0.5, water_goal_ml * 0.8], "color": "rgba(56, 189, 248, 0.35)"},
                {"range": [water_goal_ml * 0.8, water_goal_ml], "color": "rgba(56, 189, 248, 0.55)"}
            ],
            "threshold": {
                "line": {"color": "#22C55E", "width": 4},
                "thickness": 0.8,
                "value": water_goal_ml
            }
        }
    ))

    ring_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=CHART_SURFACE,
        plot_bgcolor=CHART_SURFACE,
        font={"color": "white"},
        margin=dict(l=20, r=20, t=60, b=20),
        height=330
    )

    st.plotly_chart(ring_fig, width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)

with top_col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Hydration Summary")

    m1, m2, m3 = st.columns(3)
    m1.metric("Water Today", f"{today_water:.0f} ml")
    m2.metric("Goal", f"{water_goal_ml} ml")
    m3.metric("Progress", f"{progress_percent:.1f}%")

    if remaining > 0:
        st.info(f"You need {remaining:.0f} ml more to reach today’s goal.")
    else:
        st.success(f"You reached your goal and exceeded it by {abs(remaining):.0f} ml.")

    st.progress(progress_value, text=f"{today_water:.0f} / {water_goal_ml} ml")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Add water
# -----------------------------
left_col, right_col = st.columns([1, 1])

with left_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Add Water Intake")

    amount_ml = st.number_input(
        "Enter water amount (ml)",
        min_value=50,
        max_value=2000,
        value=250,
        step=50
    )

    if st.button("➕ Add Water", width="stretch"):
        add_water(amount_ml)
        st.success(f"Added {amount_ml} ml of water.")
        st.rerun()

    st.markdown("### Quick Add")
    q1, q2, q3, q4 = st.columns(4)

    with q1:
        if st.button("250 ml"):
            add_water(250)
            st.rerun()
    with q2:
        if st.button("500 ml"):
            add_water(500)
            st.rerun()
    with q3:
        if st.button("750 ml"):
            add_water(750)
            st.rerun()
    with q4:
        if st.button("1000 ml"):
            add_water(1000)
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Daily Water Trend")

    history = get_daily_water_history()

    if history:
        history_df = pd.DataFrame(history, columns=["Date", "Water"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history_df["Date"],
            y=history_df["Water"],
            mode="lines+markers",
            line=dict(color="#38BDF8", width=4, shape="spline"),
            marker=dict(size=8, color="#38BDF8", line=dict(width=2, color="#FFFFFF")),
            fill="tozeroy",
            fillcolor="rgba(56, 189, 248, 0.16)",
            hovertemplate="<b>Date:</b> %{x}<br><b>Water:</b> %{y:.0f} ml<extra></extra>"
        ))

        fig.add_hline(
            y=water_goal_ml,
            line_dash="dash",
            line_color="#22C55E",
            annotation_text="Water Goal",
            annotation_position="top left"
        )

        fig.update_layout(
            title="Daily Water Intake",
            template="plotly_dark",
            plot_bgcolor=CHART_SURFACE,
            paper_bgcolor=CHART_SURFACE,
            font=dict(color="white"),
            xaxis_title="Date",
            yaxis_title="Water (ml)",
            margin=dict(l=20, r=20, t=50, b=20),
            height=380
        )

        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No water history available yet.")

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# History
# -----------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Water Log History")

logs = get_water_history()

if logs:
    history_df = pd.DataFrame(logs, columns=["ID", "Amount (ml)", "Created At"])
    st.dataframe(history_df, width="stretch", height=300)

    st.markdown("### Delete Water Entry")
    delete_id = st.number_input(
        "Enter Water Log ID",
        min_value=1,
        step=1,
        key="delete_water_id"
    )

    if st.button("🗑️ Delete Water Entry", width="stretch"):
        if water_log_exists(delete_id):
            delete_water_log(delete_id)
            st.success(f"Water log ID {delete_id} deleted.")
            st.rerun()
        else:
            st.warning("That Water Log ID does not exist.")
else:
    st.info("No water entries saved yet.")

st.markdown('</div>', unsafe_allow_html=True)