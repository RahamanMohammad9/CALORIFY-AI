import os
import sqlite3

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ai_insights import build_daily_insights
from database import (
    create_table,
    get_all_meals,
    get_daily_calorie_history,
    get_daily_macro_history,
    get_today_totals,
    get_weekly_summary,
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
    MACRO_PIE_COLORS,
    apply_glass_style,
    render_page_header,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(os.path.dirname(BASE_DIR), "meal_history.db")


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
    page_title="Calorify AI",
    page_icon=FAVICON_PATH,
    layout="wide"
)

create_table()
apply_glass_style(st)

render_page_header(
    st,
    "Dashboard",
    "AI Nutrition Coach: tracking, decisions, and personalized guidance",
    kicker="Overview",
)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
with st.expander("Project Highlights", expanded=False):
    h1, h2 = st.columns(2)
    with h1:
        st.write("**USP:** AI Nutrition Coach, not just calorie tracking.")
        st.write("**Core stack:** Computer Vision + Streamlit + SQLite + Analytics.")
        st.write("**Personalization:** BMI, BMR, activity-aware targets, goal-based macros.")
    with h2:
        st.write("**Actionable guidance:** over/under alerts and smart meal suggestions.")
        st.write("**Data storytelling:** trends, projections, and behavior insights.")
        st.write("**Health ecosystem:** food, water, weight, sleep, and activity tracking.")
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Dashboard navigation hub
# -----------------------------
DASHBOARD_NAV = [
    ("pages/2_Add_Meal.py", "🍽️", "Add Meal", "Photo or name, save meals"),
    ("pages/1_Meal_Tracker.py", "📊", "Meal Tracker", "Totals, trends, history"),
    ("pages/3_Meal_History.py", "📜", "Meal History", "Search and manage entries"),
    ("pages/4_Analytics.py", "📈", "Analytics", "Charts and insights"),
    ("pages/5_Water_Tracker.py", "💧", "Water Tracker", "Hydration trends"),
    ("pages/6_Weight_Tracker.py", "⚖️", "Weight Tracker", "Progress over time"),
    ("pages/7_Sleep_Tracker.py", "😴", "Sleep Tracker", "Sleep duration trends"),
    ("pages/8_Activity_Tracker.py", "🏃", "Activity Tracker", "Steps and workouts"),
    ("pages/1_Profile.py", "👤", "Profile", "Goals and targets"),
    ("pages/9_Model_Comparison.py", "🧪", "Model Comparison", "Compare deep learning models"),
]

st.markdown(
    """
    <style> 
    .dash-hub-card {
        border-radius: 14px;
        border: 1px solid rgba(148, 163, 184, 0.14);
        background: rgba(15, 23, 42, 0.45);
        padding: 1.1rem 0.9rem 0.75rem;
        text-align: center;
        margin-bottom: 0.3rem;
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }
    .dash-hub-card:hover {
        border-color: rgba(45, 212, 191, 0.35);
        box-shadow: 0 0 0 1px rgba(20, 184, 166, 0.12);
    }
    .dash-hub-emoji {
        font-family: "Segoe UI Emoji", "Segoe UI Symbol", "Apple Color Emoji",
            "Noto Color Emoji", "Android Emoji", "EmojiSymbols", sans-serif !important;
        font-size: 2.5rem;
        line-height: 1.1;
        margin-bottom: 0.3rem;
        display: block;
        filter: none !important;
    }
    .dash-hub-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 0.15rem;
    }
    .dash-hub-desc {
        font-size: 0.7rem;
        color: #94a3b8;
        line-height: 1.35;
        min-height: 2.2rem;
        margin-bottom: 0.5rem;
    }
    div[data-testid="column"] .stButton > button { margin-top: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("##### Go to")
cols_per_row = 4
for i in range(0, len(DASHBOARD_NAV), cols_per_row):
    row = st.columns(cols_per_row, gap="small")
    for j, col in enumerate(row):
        idx = i + j
        if idx >= len(DASHBOARD_NAV):
            break
        path, emoji, label, desc = DASHBOARD_NAV[idx]
        with col:
            st.markdown(
                f"""<div class="dash-hub-card">
<div class="dash-hub-emoji" role="img" aria-label="{label}">{emoji}</div>
<div class="dash-hub-label">{label}</div>
<div class="dash-hub-desc">{desc}</div>
</div>""",
                unsafe_allow_html=True,
            )
            if st.button(
                f"Open {label}",
                key=f"dash_hub_{idx}",
                width="stretch",
            ):
                st.switch_page(path)

st.divider()

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Dashboard Settings")
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

    st.caption(f"Personalized calorie target: **{personalized_goal} kcal/day**")
    st.markdown("---")
    st.caption("Use the pages list above to switch sections.")

# -----------------------------
# Load data
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

if history:
    history_df = pd.DataFrame(history, columns=["Date", "Calories"])
if macro_history:
    macro_df = pd.DataFrame(macro_history, columns=["Date", "Protein", "Carbs", "Fat"])

recent_calories = []
recent_protein = []
recent_carbs = []

if history_df is not None and len(history_df) > 0:
    recent_calories = history_df["Calories"].tail(7).tolist()

if macro_df is not None and len(macro_df) > 0:
    recent_protein = macro_df["Protein"].tail(7).tolist()
    recent_carbs = macro_df["Carbs"].tail(7).tolist()

remaining = calorie_goal - today_calories
goal_progress = min(today_calories / calorie_goal, 1.0) if calorie_goal > 0 else 0
goal_percent = goal_progress * 100

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
# Top section
# -----------------------------
top_col1, top_col2 = st.columns([1, 1.4])

with top_col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    ring_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=today_calories,
        number={"suffix": " kcal", "font": {"size": 30, "color": "white"}},
        title={"text": "Today's Calories", "font": {"size": 20, "color": "white"}},
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
        height=340
    )

    st.plotly_chart(ring_fig, width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)

with top_col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Today's Overview")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Calories", f"{today_calories:.1f} kcal")
    m2.metric("Protein", f"{today_protein:.1f} g")
    m3.metric("Carbs", f"{today_carbs:.1f} g")
    m4.metric("Fat", f"{today_fat:.1f} g")

    if remaining > 0:
        st.success(f"You have {remaining:.1f} kcal remaining for today.")
    else:
        st.warning(f"You exceeded your goal by {abs(remaining):.1f} kcal today.")

    st.progress(
        goal_progress,
        text=f"Goal progress: {today_calories:.1f} / {calorie_goal} kcal ({goal_percent:.1f}%)"
    )

    st.markdown("### Weekly Snapshot")
    w1, w2, w3, w4 = st.columns(4)
    w1.metric("Weekly Calories", f"{week_calories:.1f} kcal")
    w2.metric("Weekly Protein", f"{week_protein:.1f} g")
    w3.metric("Weekly Carbs", f"{week_carbs:.1f} g")
    w4.metric("Weekly Fat", f"{week_fat:.1f} g")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Lifestyle KPIs
# -----------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Lifestyle Snapshot")

l1, l2, l3, l4, l5 = st.columns(5)

if latest_sleep:
    l1.metric("Last Sleep", f"{float(latest_sleep[0]):.1f} h")
else:
    l1.metric("Last Sleep", "N/A")

if latest_sleep:
    l2.metric("Sleep Quality", f"{int(latest_sleep[1])}/10")
else:
    l2.metric("Sleep Quality", "N/A")

if latest_activity:
    l3.metric("Last Steps", f"{int(latest_activity[0]):,}")
else:
    l3.metric("Last Steps", "N/A")

if latest_activity:
    l4.metric("Workout", f"{int(latest_activity[1])} min")
else:
    l4.metric("Workout", "N/A")

water_today_value = float(latest_water_today[0]) if latest_water_today else 0.0
l5.metric("Water Today", f"{water_today_value:.0f} ml")

st.caption("Track sleep, water, activity, and weight daily to unlock stronger AI coaching patterns.")
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# AI coach feedback
# -----------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("AI Nutrition Coach")

st.markdown(_priority_badge(coach.get("priority", "low")), unsafe_allow_html=True)

a1, a2 = st.columns([1, 1])

with a1:
    st.markdown("##### Main issue today")
    st.write(coach.get("main_issue", "No major issue detected today."))

with a2:
    st.markdown("##### Best action")
    st.write(coach.get("best_action", "Stay consistent with your current routine."))

st.markdown("##### Coach summary")
st.write(coach.get("summary", "No summary available."))

if coach.get("critical"):
    _render_bullets("Critical alerts", coach["critical"], kind="critical")

if coach.get("warnings"):
    _render_bullets("Warnings", coach["warnings"], kind="warning")

if coach.get("patterns"):
    _render_bullets("Detected patterns", coach["patterns"], kind="pattern")

if coach.get("wins"):
    _render_bullets("What is going well", coach["wins"], kind="success")

if coach.get("info"):
    _render_bullets("Additional guidance", coach["info"], kind="neutral")

if coach.get("suggestions"):
    _render_bullets("Smart suggestions", coach["suggestions"], kind="neutral")

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Middle section
# -----------------------------
mid_col1, mid_col2 = st.columns([1.2, 1])

with mid_col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Daily Calorie Trend")

    if history:
        trend_df = pd.DataFrame(history, columns=["Date", "Calories"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trend_df["Date"],
            y=trend_df["Calories"],
            mode="lines+markers",
            name="Calories",
            line=dict(color=ACCENT, width=3, shape="spline"),
            marker=dict(size=7, color=ACCENT, line=dict(width=2, color="#FFFFFF")),
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

with mid_col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Macro Split Today")

    macro_values = [today_protein, today_carbs, today_fat]
    macro_labels = ["Protein", "Carbs", "Fat"]

    if sum(macro_values) > 0:
        pie_fig = go.Figure(
            data=[
                go.Pie(
                    labels=macro_labels,
                    values=macro_values,
                    hole=0.55,
                    marker=dict(colors=MACRO_PIE_COLORS),
                    textinfo="label+percent"
                )
            ]
        )

        pie_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=CHART_SURFACE,
            plot_bgcolor=CHART_SURFACE,
            font=dict(color="white"),
            margin=dict(l=20, r=20, t=40, b=20),
            height=420
        )

        st.plotly_chart(pie_fig, width="stretch")
    else:
        st.info("No macro data available for today yet.")

    st.markdown("### Target vs Actual")
    t1, t2, t3 = st.columns(3)
    t1.metric("Protein Target", f"{targets['protein_g']:.0f} g")
    t2.metric("Carb Target", f"{targets['carbs_g']:.0f} g")
    t3.metric("Fat Guide", f"{targets['fat_g']:.0f} g")

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Bottom section
# -----------------------------
bottom_col1, bottom_col2 = st.columns([1.2, 1])

with bottom_col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Recent Meals")

    if meals:
        recent_df = pd.DataFrame(meals, columns=[
            "ID", "Food", "Grams", "Calories", "Protein", "Carbs", "Fat", "Confidence", "Created At"
        ])
        recent_df = recent_df.head(5)
        st.dataframe(recent_df, width="stretch", height=220)
    else:
        st.info("No meals saved yet.")

    st.markdown('</div>', unsafe_allow_html=True)

with bottom_col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Quick Insights")

    total_meals = len(meals)
    avg_daily = 0.0
    consistency_score = 0

    if history_df is not None and len(history_df) > 0:
        avg_daily = float(history_df["Calories"].mean())
        days_near_goal = ((history_df["Calories"] >= calorie_goal * 0.9) & (history_df["Calories"] <= calorie_goal * 1.1)).sum()
        consistency_score = int(round((days_near_goal / len(history_df)) * 100))

    last_meal_text = "No meals logged yet."
    if meals:
        last_meal = meals[0]
        last_meal_text = f"{last_meal[1]} ({last_meal[2]:.0f} g)"

    st.write(f"**Meals logged:** {total_meals}")
    st.write(f"**Average daily calories:** {avg_daily:.1f} kcal")
    st.write(f"**Last saved meal:** {last_meal_text}")
    st.write(f"**Goal consistency:** {consistency_score}% of logged days near target")

    st.markdown("### Tips")
    st.write("- Use **Add Meal** to scan and save food.")
    st.write("- Use **Meal History** to review saved meals.")
    st.write("- Use **Analytics** for trends and deeper insights.")
    st.write("- Use all trackers regularly for stronger AI coaching.")

    st.markdown('</div>', unsafe_allow_html=True)