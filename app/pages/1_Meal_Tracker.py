import sys
import sqlite3
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ai_insights import build_daily_insights
from database import (
    create_table,
    delete_meal,
    get_all_meals,
    get_daily_calorie_history,
    get_daily_macro_history,
    get_today_totals,
    get_weekly_summary,
    meal_exists,
)
from profile_utils import calculate_daily_calories, load_profile, macro_targets
from utils import (
    ACCENT,
    ACCENT_FILL_A,
    ACCENT_FILL_B,
    ACCENT_FILL_C,
    CHART_SURFACE,
    FAVICON_PATH,
    GOAL_LINE,
    apply_glass_style,
    render_page_header,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "meal_history.db"


def get_health_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def get_latest_sleep_activity_water_weight():
    conn = get_health_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sleep_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sleep_hours REAL NOT NULL,
            sleep_quality INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS activity_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            steps INTEGER NOT NULL,
            workout_minutes INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS water_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            amount_ml REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS weight_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            weight_kg REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        SELECT sleep_hours, sleep_quality, created_at
        FROM sleep_logs
        ORDER BY created_at DESC
        LIMIT 1
    """)
    latest_sleep = cursor.fetchone()

    cursor.execute("""
        SELECT steps, workout_minutes, created_at
        FROM activity_logs
        ORDER BY created_at DESC
        LIMIT 1
    """)
    latest_activity = cursor.fetchone()

    cursor.execute("""
        SELECT COALESCE(SUM(amount_ml), 0)
        FROM water_logs
        WHERE substr(created_at, 1, 10) = date('now', 'localtime')
    """)
    latest_water_today = cursor.fetchone()

    cursor.execute("""
        SELECT weight_kg, created_at
        FROM weight_logs
        ORDER BY created_at DESC
        LIMIT 1
    """)
    latest_weight = cursor.fetchone()

    conn.close()
    return latest_sleep, latest_activity, latest_water_today, latest_weight


def get_recent_health_series(limit_days: int = 7):
    conn = get_health_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sleep_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sleep_hours REAL NOT NULL,
            sleep_quality INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS activity_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            steps INTEGER NOT NULL,
            workout_minutes INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS weight_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            weight_kg REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute(f"""
        SELECT avg_sleep
        FROM (
            SELECT substr(created_at, 1, 10) AS day, AVG(sleep_hours) AS avg_sleep
            FROM sleep_logs
            GROUP BY substr(created_at, 1, 10)
            ORDER BY day DESC
            LIMIT {int(limit_days)}
        )
    """)
    recent_sleep = [row[0] for row in cursor.fetchall()][::-1]

    cursor.execute(f"""
        SELECT avg_steps
        FROM (
            SELECT substr(created_at, 1, 10) AS day, AVG(steps) AS avg_steps
            FROM activity_logs
            GROUP BY substr(created_at, 1, 10)
            ORDER BY day DESC
            LIMIT {int(limit_days)}
        )
    """)
    recent_steps = [row[0] for row in cursor.fetchall()][::-1]

    cursor.execute(f"""
        SELECT avg_weight
        FROM (
            SELECT substr(created_at, 1, 10) AS day, AVG(weight_kg) AS avg_weight
            FROM weight_logs
            GROUP BY substr(created_at, 1, 10)
            ORDER BY day DESC
            LIMIT {int(limit_days)}
        )
    """)
    recent_weights = [row[0] for row in cursor.fetchall()][::-1]

    conn.close()
    return recent_sleep, recent_steps, recent_weights


def _priority_badge(priority: str) -> str:
    p = str(priority or "").lower()
    if p == "high":
        bg = "rgba(239, 68, 68, 0.18)"
        border = "rgba(239, 68, 68, 0.35)"
        text = "#fecaca"
        label = "HIGH PRIORITY"
    elif p == "medium":
        bg = "rgba(245, 158, 11, 0.16)"
        border = "rgba(245, 158, 11, 0.35)"
        text = "#fde68a"
        label = "MEDIUM PRIORITY"
    else:
        bg = "rgba(20, 184, 166, 0.16)"
        border = "rgba(20, 184, 166, 0.35)"
        text = "#ccfbf1"
        label = "ON TRACK"

    return f"""
    <div style="
        display:inline-block;
        padding:0.42rem 0.72rem;
        border-radius:999px;
        background:{bg};
        border:1px solid {border};
        color:{text};
        font-size:0.72rem;
        font-weight:700;
        letter-spacing:0.08em;
        text-transform:uppercase;
        margin-bottom:0.8rem;
    ">
        {label}
    </div>
    """


def _render_bullets(title: str, items: list[str], kind: str = "neutral"):
    if not items:
        return

    if kind == "critical":
        icon = "🚨"
    elif kind == "warning":
        icon = "⚠️"
    elif kind == "success":
        icon = "✅"
    elif kind == "pattern":
        icon = "📊"
    else:
        icon = "•"

    st.markdown(f"##### {title}")
    for item in items:
        st.write(f"{icon} {item}")


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Meal Tracker",
    page_icon=FAVICON_PATH,
    layout="wide"
)

create_table()
apply_glass_style(st)

render_page_header(
    st,
    "Meal Tracker",
    "Monitor today's intake, compare it with targets, and review recent meal activity.",
    kicker="Tracking",
)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Tracker Settings")
    profile = load_profile()
    personalized_goal = int(
        round(
            calculate_daily_calories(
                profile["weight_kg"],
                profile["height_cm"],
                profile["age"],
                profile["gender"],
                profile["activity_level"],
                profile["goal"],
            )
        )
    )

    calorie_goal = st.number_input(
        "Daily calorie goal",
        min_value=1000,
        max_value=5000,
        value=max(1000, min(5000, personalized_goal)),
        step=50
    )

    water_goal_ml = st.number_input(
        "Daily water goal (ml)",
        min_value=500,
        max_value=6000,
        value=2500,
        step=100
    )

    st.caption(f"Profile-based target: **{personalized_goal} kcal/day**")

# -----------------------------
# Data load
# -----------------------------
today_calories, today_protein, today_carbs, today_fat = get_today_totals()
week_calories, week_protein, week_carbs, week_fat = get_weekly_summary()
history = get_daily_calorie_history()
macro_history = get_daily_macro_history()
meals = get_all_meals()

latest_sleep, latest_activity, latest_water_today, latest_weight = get_latest_sleep_activity_water_weight()
recent_sleep, recent_steps, recent_weights = get_recent_health_series(limit_days=7)

history_df = None
macro_df = None
meals_df = None

if history:
    history_df = pd.DataFrame(history, columns=["Date", "Calories"])

if macro_history:
    macro_df = pd.DataFrame(macro_history, columns=["Date", "Protein", "Carbs", "Fat"])

if meals:
    meals_df = pd.DataFrame(meals, columns=[
        "ID", "Food", "Grams", "Calories", "Protein", "Carbs", "Fat", "Confidence", "Created At"
    ])

recent_calories = history_df["Calories"].tail(7).tolist() if history_df is not None and len(history_df) > 0 else []
recent_protein = macro_df["Protein"].tail(7).tolist() if macro_df is not None and len(macro_df) > 0 else []
recent_carbs = macro_df["Carbs"].tail(7).tolist() if macro_df is not None and len(macro_df) > 0 else []

targets = macro_targets(calorie_goal, profile["weight_kg"], profile["goal"])

coach = build_daily_insights(
    today_calories=today_calories,
    today_protein=today_protein,
    today_carbs=today_carbs,
    calorie_goal=calorie_goal,
    protein_goal=targets["protein_g"],
    carbs_goal=targets["carbs_g"],
    goal=profile["goal"],
    latest_sleep_hours=float(latest_sleep[0]) if latest_sleep else None,
    latest_steps=int(latest_activity[0]) if latest_activity else None,
    latest_water_ml=float(latest_water_today[0]) if latest_water_today else None,
    water_goal_ml=water_goal_ml,
    recent_calories=recent_calories,
    recent_protein=recent_protein,
    recent_carbs=recent_carbs,
    recent_sleep_hours=recent_sleep,
    recent_steps=recent_steps,
    recent_weights=recent_weights,
)

# -----------------------------
# Today's summary
# -----------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Today's Nutrition Summary")

remaining = calorie_goal - today_calories
progress_value = min(today_calories / calorie_goal, 1.0) if calorie_goal > 0 else 0
progress_percent = progress_value * 100

summary_col1, summary_col2 = st.columns([1, 1.4])

with summary_col1:
    ring_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=today_calories,
        number={"suffix": " kcal", "font": {"size": 28, "color": "white"}},
        title={"text": "Daily Calorie Goal", "font": {"size": 20, "color": "white"}},
        gauge={
            "axis": {"range": [0, calorie_goal], "tickcolor": "white"},
            "bar": {"color": ACCENT},
            "bgcolor": "#1A1F2B",
            "borderwidth": 2,
            "bordercolor": "rgba(148, 163, 184, 0.2)",
            "steps": [
                {"range": [0, calorie_goal * 0.5], "color": ACCENT_FILL_A},
                {"range": [calorie_goal * 0.5, calorie_goal * 0.8], "color": ACCENT_FILL_B},
                {"range": [calorie_goal * 0.8, calorie_goal], "color": ACCENT_FILL_C}
            ],
            "threshold": {
                "line": {"color": GOAL_LINE, "width": 4},
                "thickness": 0.8,
                "value": calorie_goal
            }
        }
    ))

    ring_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=CHART_SURFACE,
        plot_bgcolor=CHART_SURFACE,
        font={"color": "white"},
        margin=dict(l=20, r=20, t=60, b=20),
        height=350
    )

    st.plotly_chart(ring_fig, width="stretch")

with summary_col2:
    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)

    c1.metric("Calories Today", f"{today_calories:.1f} kcal")
    c2.metric("Protein Today", f"{today_protein:.1f} g")
    c3.metric("Carbs Today", f"{today_carbs:.1f} g")
    c4.metric("Fat Today", f"{today_fat:.1f} g")

    t1, t2, t3 = st.columns(3)
    t1.metric("Protein Target", f"{targets['protein_g']:.0f} g")
    t2.metric("Carb Target", f"{targets['carbs_g']:.0f} g")
    t3.metric("Fat Guide", f"{targets['fat_g']:.0f} g")

    if remaining > 0:
        st.success(f"You have {remaining:.1f} kcal remaining for today.")
    else:
        st.warning(f"You exceeded your goal by {abs(remaining):.1f} kcal today.")

    st.progress(
        progress_value,
        text=f"Goal progress: {today_calories:.1f} / {calorie_goal} kcal ({progress_percent:.1f}%)"
    )

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# AI coach tracker summary
# -----------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Tracker AI Coach")

st.markdown(_priority_badge(coach.get("priority", "low")), unsafe_allow_html=True)

a1, a2 = st.columns(2)
with a1:
    st.markdown("##### Main issue")
    st.write(coach.get("main_issue", "No major issue detected today."))

with a2:
    st.markdown("##### Best action")
    st.write(coach.get("best_action", "Stay consistent with your current routine."))

st.markdown("##### Summary")
st.write(coach.get("summary", "No summary available."))

if coach.get("critical"):
    _render_bullets("Critical alerts", coach["critical"], kind="critical")
if coach.get("warnings"):
    _render_bullets("Warnings", coach["warnings"], kind="warning")
if coach.get("patterns"):
    _render_bullets("Detected patterns", coach["patterns"], kind="pattern")
if coach.get("wins"):
    _render_bullets("What is going well", coach["wins"], kind="success")
if coach.get("suggestions"):
    _render_bullets("Smart suggestions", coach["suggestions"], kind="neutral")

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Charts
# -----------------------------
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Daily Calorie Trend")

    if history_df is not None and len(history_df) > 0:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=history_df["Date"],
            y=history_df["Calories"],
            mode="lines+markers",
            name="Calories",
            line=dict(
                color=ACCENT,
                width=3,
                shape="spline"
            ),
            marker=dict(
                size=7,
                color=ACCENT,
                line=dict(width=2, color="#FFFFFF")
            ),
            fill="tozeroy",
            fillcolor=ACCENT_FILL_A,
            hovertemplate="<b>Date:</b> %{x}<br><b>Calories:</b> %{y:.1f} kcal<extra></extra>"
        ))

        fig.add_hline(
            y=calorie_goal,
            line_dash="dash",
            line_color=GOAL_LINE,
            annotation_text="Daily Goal",
            annotation_position="top left"
        )

        fig.update_layout(
            title="Daily Calorie Intake",
            template="plotly_dark",
            plot_bgcolor=CHART_SURFACE,
            paper_bgcolor=CHART_SURFACE,
            font=dict(color="white"),
            xaxis_title="Date",
            yaxis_title="Calories",
            margin=dict(l=20, r=20, t=50, b=20),
            height=420
        )

        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No calorie history available yet.")

    st.markdown('</div>', unsafe_allow_html=True)

with chart_col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Weekly Summary")

    w1, w2 = st.columns(2)
    w3, w4 = st.columns(2)

    w1.metric("Weekly Calories", f"{week_calories:.1f} kcal")
    w2.metric("Weekly Protein", f"{week_protein:.1f} g")
    w3.metric("Weekly Carbs", f"{week_carbs:.1f} g")
    w4.metric("Weekly Fat", f"{week_fat:.1f} g")

    if history_df is not None and len(history_df) > 0:
        avg_daily = float(history_df["Calories"].mean())
        above_goal_days = int((history_df["Calories"] > calorie_goal).sum())
        near_goal_days = int(((history_df["Calories"] >= calorie_goal * 0.9) & (history_df["Calories"] <= calorie_goal * 1.1)).sum())

        s1, s2, s3 = st.columns(3)
        s1.metric("Average Daily", f"{avg_daily:.1f} kcal")
        s2.metric("Days Above Goal", above_goal_days)
        s3.metric("Days Near Goal", near_goal_days)
    else:
        st.info("Log meals on multiple days to unlock weekly tracking insights.")

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Divider
# -----------------------------
st.markdown("""
<div style="
height:1px;
background: linear-gradient(90deg, rgba(20,184,166,0), rgba(20,184,166,0.45), rgba(20,184,166,0));
margin: 18px 0 24px 0;">
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Meal history
# -----------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Meal History")

if meals_df is not None and len(meals_df) > 0:
    filt_col1, filt_col2 = st.columns([1.2, 1])
    with filt_col1:
        meal_search = st.text_input(
            "Search meals",
            value="",
            placeholder="Filter by food name...",
            key="meal_history_search",
        )
    with filt_col2:
        row_choice = st.select_slider(
            "Rows to show",
            options=["10", "25", "50", "100", "All"],
            value="All",
            key="meal_history_rows",
        )

    display_df = meals_df.copy()

    if meal_search.strip():
        mask = display_df["Food"].astype(str).str.contains(
            meal_search.strip(), case=False, na=False
        )
        display_df = display_df.loc[mask]

    if row_choice != "All":
        display_df = display_df.head(int(row_choice))

    if display_df.empty and meal_search.strip():
        st.info("No meals match that search. Clear the filter to see all meals.")
    else:
        st.dataframe(display_df, width="stretch", height=350)
        if meal_search.strip() or row_choice != "All":
            st.caption(f"Showing **{len(display_df)}** of **{len(meals_df)}** meals.")

    st.markdown("### Delete Meal Record")
    delete_col1, delete_col2 = st.columns([1, 1])

    with delete_col1:
        meal_id_to_delete = st.number_input(
            "Enter Meal ID",
            min_value=1,
            step=1,
            key="delete_meal_id"
        )

    with delete_col2:
        st.write("")
        st.write("")
        if st.button("🗑️ Delete Meal", width="stretch"):
            if meal_exists(meal_id_to_delete):
                delete_meal(meal_id_to_delete)
                st.success(f"Meal ID {meal_id_to_delete} deleted.")
                st.rerun()
            else:
                st.warning("That Meal ID does not exist.")
else:
    st.info("No meals saved yet.")

st.markdown('</div>', unsafe_allow_html=True)