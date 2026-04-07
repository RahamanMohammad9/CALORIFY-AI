import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sqlite3
import os

from database import (
    create_table,
    get_all_meals,
    get_today_totals,
    get_daily_calorie_history,
    get_weekly_summary
)
from ai_insights import build_daily_insights
from profile_utils import calculate_daily_calories, load_profile, macro_targets

# If you already added this in utils.py, keep this import.
# Otherwise you can remove these 2 lines and paste your CSS directly.
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


def get_latest_sleep_activity():
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
    conn.close()
    return latest_sleep, latest_activity

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
st.subheader("Project Highlights")
h1, h2 = st.columns(2)
with h1:
    st.write("**USP:** AI Nutrition Coach (not just calorie tracking).")
    st.write("**Core stack:** Computer Vision + Streamlit + SQLite + Analytics.")
    st.write("**Personalization:** BMI, BMR, activity-aware targets, goal-based macros.")
with h2:
    st.write("**Actionable guidance:** over/under alerts and smart meal suggestions.")
    st.write("**Data storytelling:** trends, projections, and behavior insights.")
    st.write("**Health ecosystem:** food, water, weight, sleep, and activity tracking.")
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Icon shortcuts (main dashboard only)
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
    st.caption(f"Personalized target from profile: **{personalized_goal} kcal/day**")

    st.markdown("---")
    st.caption("Use the **pages** list above to switch sections.")

# -----------------------------
# Load data
# -----------------------------
today_calories, today_protein, today_carbs, today_fat = get_today_totals()
week_calories, week_protein, week_carbs, week_fat = get_weekly_summary()
history = get_daily_calorie_history()
meals = get_all_meals()
latest_sleep, latest_activity = get_latest_sleep_activity()

remaining = calorie_goal - today_calories
goal_progress = min(today_calories / calorie_goal, 1.0) if calorie_goal > 0 else 0
goal_percent = goal_progress * 100

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
l1, l2, l3, l4 = st.columns(4)
if latest_sleep:
    l1.metric("Last Sleep", f"{float(latest_sleep[0]):.1f} h")
    l2.metric("Sleep Quality", f"{int(latest_sleep[1])}/10")
else:
    l1.metric("Last Sleep", "N/A")
    l2.metric("Sleep Quality", "N/A")
if latest_activity:
    l3.metric("Last Steps", f"{int(latest_activity[0]):,}")
    l4.metric("Workout", f"{int(latest_activity[1])} min")
else:
    l3.metric("Last Steps", "N/A")
    l4.metric("Workout", "N/A")
st.caption("Track sleep and activity daily to unlock stronger AI coaching patterns.")
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# AI coach feedback
# -----------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("AI Nutrition Coach")
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
)

for msg in coach["warnings"]:
    st.warning(msg)
for msg in coach["info"]:
    st.info(msg)

st.markdown("##### Smart suggestions")
for s in coach["suggestions"]:
    st.write(f"- {s}")
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Middle section
# -----------------------------
mid_col1, mid_col2 = st.columns([1.2, 1])

with mid_col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Daily Calorie Trend")

    if history:
        history_df = pd.DataFrame(history, columns=["Date", "Calories"])

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=history_df["Date"],
            y=history_df["Calories"],
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
    avg_daily = 0
    if history:
        history_df = pd.DataFrame(history, columns=["Date", "Calories"])
        if len(history_df) > 0:
            avg_daily = history_df["Calories"].mean()

    last_meal_text = "No meals logged yet."
    if meals:
        last_meal = meals[0]
        last_meal_text = f"{last_meal[1]} ({last_meal[2]:.0f} g)"

    st.write(f"**Meals logged:** {total_meals}")
    st.write(f"**Average daily calories:** {avg_daily:.1f} kcal")
    st.write(f"**Last saved meal:** {last_meal_text}")

    st.markdown("### Tips")
    st.write("- Use **Add Meal** to scan and save food.")
    st.write("- Use **Meal History** to review saved meals.")
    st.write("- Use **Analytics** for trends and deeper insights.")
    st.write("- Use **Water Tracker** and **Weight Tracker** for a full health log.")
    st.markdown('</div>', unsafe_allow_html=True)