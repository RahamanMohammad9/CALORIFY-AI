import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import CHART_SURFACE, FAVICON_PATH, apply_glass_style, render_page_header

st.set_page_config(
    page_title="Activity Tracker",
    page_icon=FAVICON_PATH,
    layout="wide",
)

apply_glass_style(st)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "meal_history.db")


def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def create_activity_table():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS activity_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            steps INTEGER NOT NULL,
            workout_minutes INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def add_activity_log(steps, workout_minutes):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO activity_logs (steps, workout_minutes, created_at)
        VALUES (?, ?, ?)
    """, (int(steps), int(workout_minutes), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()


def get_activity_logs(desc=True):
    conn = get_connection()
    cursor = conn.cursor()
    order = "DESC" if desc else "ASC"
    cursor.execute(f"""
        SELECT id, steps, workout_minutes, created_at
        FROM activity_logs
        ORDER BY created_at {order}
    """)
    rows = cursor.fetchall()
    conn.close()
    return rows


def activity_log_exists(log_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM activity_logs WHERE id = ?", (int(log_id),))
    exists = cursor.fetchone()[0] > 0
    conn.close()
    return exists


def delete_activity_log(log_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM activity_logs WHERE id = ?", (int(log_id),))
    conn.commit()
    conn.close()


create_activity_table()

render_page_header(
    st,
    "Activity Tracker",
    "Log daily movement and workouts to monitor lifestyle consistency",
    kicker="Movement",
)

activity_desc = get_activity_logs(desc=True)
activity_asc = get_activity_logs(desc=False)

top_left, top_right = st.columns([1, 1.4])

with top_left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Activity Summary")
    if activity_desc:
        act_df = pd.DataFrame(activity_desc, columns=["ID", "Steps", "Workout", "Created At"])
        avg_steps = float(act_df["Steps"].mean())
        avg_workout = float(act_df["Workout"].mean())
        last_steps = int(act_df.iloc[0]["Steps"])
        a1, a2, a3 = st.columns(3)
        a1.metric("Last Steps", f"{last_steps:,}")
        a2.metric("Avg Steps", f"{avg_steps:.0f}")
        a3.metric("Avg Workout", f"{avg_workout:.0f} min")
        if avg_steps < 7000:
            st.warning("Average steps are low. Try adding a short walk routine.")
        else:
            st.success("Great movement consistency.")
    else:
        st.info("No activity logs yet.")
    st.markdown("</div>", unsafe_allow_html=True)

with top_right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Add Activity Entry")
    steps = st.number_input("Steps", min_value=0, max_value=60000, value=8000, step=500)
    workout_minutes = st.number_input("Workout minutes", min_value=0, max_value=300, value=30, step=5)
    if st.button("💾 Save Activity Log", width="stretch"):
        add_activity_log(steps, workout_minutes)
        st.success("Activity log saved.")
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Activity Trend")
if activity_asc:
    trend_df = pd.DataFrame(activity_asc, columns=["ID", "Steps", "Workout", "Created At"])
    trend_df["Date"] = pd.to_datetime(trend_df["Created At"]).dt.date.astype(str)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=trend_df["Date"],
        y=trend_df["Steps"],
        name="Steps",
        marker_color="#22d3ee",
        opacity=0.85,
    ))
    fig.add_trace(go.Scatter(
        x=trend_df["Date"],
        y=trend_df["Workout"],
        name="Workout (min)",
        mode="lines+markers",
        line=dict(color="#f59e0b", width=3),
        yaxis="y2",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=CHART_SURFACE,
        plot_bgcolor=CHART_SURFACE,
        font=dict(color="white"),
        xaxis_title="Date",
        yaxis=dict(title="Steps"),
        yaxis2=dict(title="Workout (min)", overlaying="y", side="right"),
        height=420,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, width="stretch")
else:
    st.info("No activity trend data yet.")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Activity History")
if activity_desc:
    history_df = pd.DataFrame(activity_desc, columns=["ID", "Steps", "Workout (min)", "Created At"])
    st.dataframe(history_df, width="stretch", height=300)
    delete_id = st.number_input("Enter Activity Log ID to delete", min_value=1, step=1, key="activity_delete_id")
    if st.button("🗑️ Delete Activity Log", width="stretch"):
        if activity_log_exists(delete_id):
            delete_activity_log(delete_id)
            st.success("Activity log deleted.")
            st.rerun()
        else:
            st.warning("That activity log ID does not exist.")
else:
    st.info("No activity history yet.")
st.markdown("</div>", unsafe_allow_html=True)
