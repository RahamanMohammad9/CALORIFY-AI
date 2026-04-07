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
    page_title="Sleep Tracker",
    page_icon=FAVICON_PATH,
    layout="wide",
)

apply_glass_style(st)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "meal_history.db")


def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def create_sleep_table():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sleep_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sleep_hours REAL NOT NULL,
            sleep_quality INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def add_sleep_log(sleep_hours, sleep_quality):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO sleep_logs (sleep_hours, sleep_quality, created_at)
        VALUES (?, ?, ?)
    """, (float(sleep_hours), int(sleep_quality), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()


def get_sleep_logs(desc=True):
    conn = get_connection()
    cursor = conn.cursor()
    order = "DESC" if desc else "ASC"
    cursor.execute(f"""
        SELECT id, sleep_hours, sleep_quality, created_at
        FROM sleep_logs
        ORDER BY created_at {order}
    """)
    rows = cursor.fetchall()
    conn.close()
    return rows


def sleep_log_exists(log_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM sleep_logs WHERE id = ?", (int(log_id),))
    exists = cursor.fetchone()[0] > 0
    conn.close()
    return exists


def delete_sleep_log(log_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM sleep_logs WHERE id = ?", (int(log_id),))
    conn.commit()
    conn.close()


create_sleep_table()

render_page_header(
    st,
    "Sleep Tracker",
    "Track sleep duration and quality to improve recovery and health",
    kicker="Recovery",
)

sleep_logs_desc = get_sleep_logs(desc=True)
sleep_logs_asc = get_sleep_logs(desc=False)

top_left, top_right = st.columns([1, 1.4])

with top_left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Sleep Summary")
    if sleep_logs_desc:
        sleep_df = pd.DataFrame(sleep_logs_desc, columns=["ID", "Hours", "Quality", "Created At"])
        avg_hours = float(sleep_df["Hours"].mean())
        avg_quality = float(sleep_df["Quality"].mean())
        last_hours = float(sleep_df.iloc[0]["Hours"])
        s1, s2, s3 = st.columns(3)
        s1.metric("Last Night", f"{last_hours:.1f} h")
        s2.metric("Avg Hours", f"{avg_hours:.1f} h")
        s3.metric("Avg Quality", f"{avg_quality:.1f}/10")
        if avg_hours < 7:
            st.warning("Average sleep is below 7h. Try improving bedtime consistency.")
        else:
            st.success("Sleep duration is in a healthy range.")
    else:
        st.info("No sleep logs yet.")
    st.markdown("</div>", unsafe_allow_html=True)

with top_right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Add Sleep Entry")
    sleep_hours = st.number_input("Sleep hours", min_value=1.0, max_value=14.0, value=7.5, step=0.5)
    sleep_quality = st.slider("Sleep quality (1-10)", min_value=1, max_value=10, value=7)
    if st.button("💾 Save Sleep Log", width="stretch"):
        add_sleep_log(sleep_hours, sleep_quality)
        st.success("Sleep log saved.")
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Sleep Trend")
if sleep_logs_asc:
    trend_df = pd.DataFrame(sleep_logs_asc, columns=["ID", "Hours", "Quality", "Created At"])
    trend_df["Date"] = pd.to_datetime(trend_df["Created At"]).dt.date.astype(str)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend_df["Date"],
        y=trend_df["Hours"],
        mode="lines+markers",
        name="Hours",
        line=dict(color="#60a5fa", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=trend_df["Date"],
        y=trend_df["Quality"],
        mode="lines+markers",
        name="Quality (1-10)",
        line=dict(color="#34d399", width=3),
        yaxis="y2",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=CHART_SURFACE,
        plot_bgcolor=CHART_SURFACE,
        font=dict(color="white"),
        xaxis_title="Date",
        yaxis=dict(title="Hours"),
        yaxis2=dict(title="Quality", overlaying="y", side="right"),
        height=420,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, width="stretch")
else:
    st.info("No sleep trend data yet.")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Sleep History")
if sleep_logs_desc:
    history_df = pd.DataFrame(sleep_logs_desc, columns=["ID", "Sleep Hours", "Quality (1-10)", "Created At"])
    st.dataframe(history_df, width="stretch", height=300)
    delete_id = st.number_input("Enter Sleep Log ID to delete", min_value=1, step=1, key="sleep_delete_id")
    if st.button("🗑️ Delete Sleep Log", width="stretch"):
        if sleep_log_exists(delete_id):
            delete_sleep_log(delete_id)
            st.success("Sleep log deleted.")
            st.rerun()
        else:
            st.warning("That sleep log ID does not exist.")
else:
    st.info("No sleep history yet.")
st.markdown("</div>", unsafe_allow_html=True)
