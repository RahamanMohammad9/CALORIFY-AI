import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sqlite3
import os

from database import (
    create_table,
    get_all_meals,
    get_today_totals,
    get_daily_calorie_history,
    get_weekly_summary
)
from utils import (
    ACCENT,
    ACCENT_FILL_A,
    CHART_SURFACE,
    FAVICON_PATH,
    GOAL_LINE,
    MACRO_PIE_COLORS,
    apply_glass_style,
    render_page_header,
)
from profile_utils import calculate_daily_calories, load_profile, macro_targets

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(os.path.dirname(BASE_DIR), "meal_history.db")


def get_health_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def get_sleep_activity_daily():
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
        SELECT substr(created_at, 1, 10) AS day, AVG(sleep_hours) AS sleep_hours
        FROM sleep_logs
        GROUP BY substr(created_at, 1, 10)
        ORDER BY day ASC
    """)
    sleep_rows = cursor.fetchall()
    cursor.execute("""
        SELECT substr(created_at, 1, 10) AS day, AVG(steps) AS steps, AVG(workout_minutes) AS workout_minutes
        FROM activity_logs
        GROUP BY substr(created_at, 1, 10)
        ORDER BY day ASC
    """)
    activity_rows = cursor.fetchall()
    conn.close()
    sleep_df = pd.DataFrame(sleep_rows, columns=["Date", "SleepHours"]) if sleep_rows else pd.DataFrame(columns=["Date", "SleepHours"])
    activity_df = pd.DataFrame(activity_rows, columns=["Date", "Steps", "WorkoutMinutes"]) if activity_rows else pd.DataFrame(columns=["Date", "Steps", "WorkoutMinutes"])
    if not sleep_df.empty:
        sleep_df["Date"] = pd.to_datetime(sleep_df["Date"])
    if not activity_df.empty:
        activity_df["Date"] = pd.to_datetime(activity_df["Date"])
    return sleep_df, activity_df

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Analytics",
    page_icon=FAVICON_PATH,
    layout="wide"
)

create_table()
apply_glass_style(st)

render_page_header(
    st,
    "Analytics",
    "Explore nutrition trends, meal patterns, and progress insights",
    kicker="Insights",
)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Analytics Settings")
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
    st.caption(f"Profile-based target: **{personalized_goal} kcal/day**")

# -----------------------------
# Load data
# -----------------------------
meals = get_all_meals()
today_calories, today_protein, today_carbs, today_fat = get_today_totals()
week_calories, week_protein, week_carbs, week_fat = get_weekly_summary()
history = get_daily_calorie_history()

meals_df = None
history_df = None

if meals:
    meals_df = pd.DataFrame(meals, columns=[
        "ID", "Food", "Grams", "Calories", "Protein", "Carbs", "Fat", "Confidence", "Created At"
    ])
    meals_df["Created At"] = pd.to_datetime(meals_df["Created At"])
    meals_df["Date"] = meals_df["Created At"].dt.date.astype(str)
    meals_df["DateDT"] = pd.to_datetime(meals_df["Date"])
    meals_df["Hour"] = meals_df["Created At"].dt.hour
    meals_df["Weekday"] = meals_df["Created At"].dt.day_name()

if history:
    history_df = pd.DataFrame(history, columns=["Date", "Calories"])
    history_df["Date"] = pd.to_datetime(history_df["Date"])
sleep_daily_df, activity_daily_df = get_sleep_activity_daily()

# -----------------------------
# KPI row
# -----------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Key Performance Indicators")

total_meals = len(meals) if meals else 0
avg_daily_calories = history_df["Calories"].mean() if history_df is not None and len(history_df) > 0 else 0
best_day = history_df.loc[history_df["Calories"].idxmax(), "Date"] if history_df is not None and len(history_df) > 0 else "N/A"
best_day_cal = history_df["Calories"].max() if history_df is not None and len(history_df) > 0 else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Meals Logged", total_meals)
k2.metric("Average Daily Calories", f"{avg_daily_calories:.1f} kcal")
k3.metric("Today's Calories", f"{today_calories:.1f} kcal")
k4.metric("Highest Intake Day", f"{best_day_cal:.1f} kcal" if best_day != "N/A" else "N/A")

if best_day != "N/A":
    st.caption(f"Highest calorie day recorded: {best_day}")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Nutrition vs Recovery & Activity")
cx1, cx2 = st.columns(2)

with cx1:
    st.markdown("##### Calories vs sleep (by day)")
    if history_df is not None and len(history_df) > 0 and not sleep_daily_df.empty:
        cal_sleep = history_df.merge(sleep_daily_df, on="Date", how="inner")
        if len(cal_sleep) > 0:
            cs_fig = go.Figure()
            cs_fig.add_trace(go.Scatter(
                x=cal_sleep["SleepHours"],
                y=cal_sleep["Calories"],
                mode="markers+text",
                text=cal_sleep["Date"].dt.strftime("%m-%d"),
                textposition="top center",
                marker=dict(size=10, color="#60a5fa"),
                name="Day",
            ))
            cs_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor=CHART_SURFACE,
                plot_bgcolor=CHART_SURFACE,
                font=dict(color="white"),
                height=340,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title="Sleep hours",
                yaxis_title="Calories",
            )
            st.plotly_chart(cs_fig, width="stretch")
        else:
            st.info("No overlapping sleep and calorie dates yet.")
    else:
        st.info("Add both sleep logs and meal logs to view this.")

with cx2:
    st.markdown("##### Calories vs steps (by day)")
    if history_df is not None and len(history_df) > 0 and not activity_daily_df.empty:
        cal_act = history_df.merge(activity_daily_df, on="Date", how="inner")
        if len(cal_act) > 0:
            ca_fig = go.Figure()
            ca_fig.add_trace(go.Scatter(
                x=cal_act["Steps"],
                y=cal_act["Calories"],
                mode="markers+text",
                text=cal_act["Date"].dt.strftime("%m-%d"),
                textposition="top center",
                marker=dict(size=10, color="#34d399"),
                name="Day",
            ))
            ca_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor=CHART_SURFACE,
                plot_bgcolor=CHART_SURFACE,
                font=dict(color="white"),
                height=340,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title="Steps",
                yaxis_title="Calories",
            )
            st.plotly_chart(ca_fig, width="stretch")
        else:
            st.info("No overlapping activity and calorie dates yet.")
    else:
        st.info("Add both activity logs and meal logs to view this.")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Data Export")
if meals_df is not None and len(meals_df) > 0:
    export_df = meals_df[["ID", "Food", "Grams", "Calories", "Protein", "Carbs", "Fat", "Confidence", "Created At"]]
    st.download_button(
        label="Download meals as CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="calorify_meals_export.csv",
        mime="text/csv",
        width="stretch",
    )
else:
    st.info("No meal data available to export yet.")
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Charts row 1
# -----------------------------
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Daily Calorie Trend")

    if history_df is not None and len(history_df) > 0:
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
            annotation_text="Goal",
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

with row1_col2:
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
# Charts row 2
# -----------------------------
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Weekly Macro Totals")

    week_macro_df = pd.DataFrame({
        "Macro": ["Protein", "Carbs", "Fat"],
        "Amount": [week_protein, week_carbs, week_fat]
    })

    if week_macro_df["Amount"].sum() > 0:
        bar_fig = px.bar(
            week_macro_df,
            x="Macro",
            y="Amount",
            color="Macro",
            color_discrete_map={
                "Protein": MACRO_PIE_COLORS[0],
                "Carbs": MACRO_PIE_COLORS[1],
                "Fat": MACRO_PIE_COLORS[2],
            }
        )

        bar_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=CHART_SURFACE,
            plot_bgcolor=CHART_SURFACE,
            font=dict(color="white"),
            margin=dict(l=20, r=20, t=40, b=20),
            height=380,
            showlegend=False
        )

        st.plotly_chart(bar_fig, width="stretch")
    else:
        st.info("No weekly macro data available yet.")
    st.markdown('</div>', unsafe_allow_html=True)

with row2_col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Top Foods Logged")

    if meals_df is not None and len(meals_df) > 0:
        top_foods = meals_df["Food"].value_counts().head(8).reset_index()
        top_foods.columns = ["Food", "Count"]

        food_fig = px.bar(
            top_foods,
            x="Count",
            y="Food",
            orientation="h",
            color="Count",
            color_continuous_scale=[
                [0.0, "#1e293b"],
                [0.45, "#0f766e"],
                [1.0, "#5eead4"],
            ]
        )

        food_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=CHART_SURFACE,
            plot_bgcolor=CHART_SURFACE,
            font=dict(color="white"),
            margin=dict(l=20, r=20, t=40, b=20),
            height=380,
            yaxis=dict(categoryorder="total ascending")
        )

        st.plotly_chart(food_fig, width="stretch")
    else:
        st.info("No meal data available yet.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Advanced Analytics")
adv1, adv2 = st.columns(2)

with adv1:
    st.markdown("##### Weekly calorie trend")
    if history_df is not None and len(history_df) > 0:
        weekly = history_df.set_index("Date").resample("W")["Calories"].mean().reset_index()
        wfig = go.Figure()
        wfig.add_trace(
            go.Scatter(
                x=weekly["Date"],
                y=weekly["Calories"],
                mode="lines+markers",
                line=dict(color=ACCENT, width=3),
                name="Weekly Avg Calories",
            )
        )
        wfig.add_hline(y=calorie_goal, line_dash="dash", line_color=GOAL_LINE)
        wfig.update_layout(
            template="plotly_dark",
            paper_bgcolor=CHART_SURFACE,
            plot_bgcolor=CHART_SURFACE,
            font=dict(color="white"),
            height=340,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Week",
            yaxis_title="Avg kcal/day",
        )
        st.plotly_chart(wfig, width="stretch")
    else:
        st.info("Log meals for at least several days to see weekly trends.")

with adv2:
    st.markdown("##### Goal vs actual calories")
    if history_df is not None and len(history_df) > 0:
        compare_df = history_df.copy()
        compare_df["Goal"] = calorie_goal
        compare_df["Actual"] = compare_df["Calories"]
        gfig = go.Figure()
        gfig.add_trace(go.Bar(x=compare_df["Date"], y=compare_df["Goal"], name="Goal"))
        gfig.add_trace(go.Bar(x=compare_df["Date"], y=compare_df["Actual"], name="Actual"))
        gfig.update_layout(
            barmode="group",
            template="plotly_dark",
            paper_bgcolor=CHART_SURFACE,
            plot_bgcolor=CHART_SURFACE,
            font=dict(color="white"),
            height=340,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Date",
            yaxis_title="Calories",
        )
        st.plotly_chart(gfig, width="stretch")
    else:
        st.info("No data yet for goal-vs-actual comparison.")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Macro Distribution Over Time")
if meals_df is not None and len(meals_df) > 0:
    macro_day = (
        meals_df.groupby("DateDT", as_index=False)[["Protein", "Carbs", "Fat"]]
        .sum()
        .sort_values("DateDT")
    )
    mfig = go.Figure()
    mfig.add_trace(go.Scatter(x=macro_day["DateDT"], y=macro_day["Protein"], mode="lines", name="Protein"))
    mfig.add_trace(go.Scatter(x=macro_day["DateDT"], y=macro_day["Carbs"], mode="lines", name="Carbs"))
    mfig.add_trace(go.Scatter(x=macro_day["DateDT"], y=macro_day["Fat"], mode="lines", name="Fat"))
    mfig.update_layout(
        template="plotly_dark",
        paper_bgcolor=CHART_SURFACE,
        plot_bgcolor=CHART_SURFACE,
        font=dict(color="white"),
        height=360,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Date",
        yaxis_title="Grams/day",
    )
    st.plotly_chart(mfig, width="stretch")
else:
    st.info("No macro history available yet.")
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Insights section
# -----------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("AI Insights")

if meals_df is not None and len(meals_df) > 0:
    avg_protein = meals_df["Protein"].mean()
    avg_carbs = meals_df["Carbs"].mean()
    avg_fat = meals_df["Fat"].mean()

    i1, i2, i3 = st.columns(3)
    i1.metric("Avg Protein per Meal", f"{avg_protein:.1f} g")
    i2.metric("Avg Carbs per Meal", f"{avg_carbs:.1f} g")
    i3.metric("Avg Fat per Meal", f"{avg_fat:.1f} g")

    targets = macro_targets(calorie_goal, profile["weight_kg"], profile["goal"])

    if history_df is not None and len(history_df) > 0:
        above_goal_days = (history_df["Calories"] > calorie_goal).sum()
        below_goal_days = (history_df["Calories"] <= calorie_goal).sum()

        st.write(f"**Days above calorie goal:** {above_goal_days}")
        st.write(f"**Days within or below calorie goal:** {below_goal_days}")
        avg_cal = float(history_df["Calories"].mean())
        daily_delta = avg_cal - calorie_goal
        projected_weight_kg = profile["weight_kg"] + ((daily_delta * 30) / 7700.0)
        direction = "gain" if projected_weight_kg >= profile["weight_kg"] else "loss"
        st.write(
            f"**30-day projection:** If you continue like this, estimated weight {direction} "
            f"is **{abs(projected_weight_kg - profile['weight_kg']):.2f} kg** "
            f"(~{projected_weight_kg:.2f} kg total)."
        )

    most_common_food = meals_df["Food"].mode()[0]
    st.write(f"**Most frequently logged food:** {most_common_food}")
    st.write(
        f"**Daily macro targets:** Protein {targets['protein_g']:.0f}g, "
        f"Carbs {targets['carbs_g']:.0f}g, Fat {targets['fat_g']:.0f}g."
    )
    if today_carbs > targets["carbs_g"] + 10:
        st.warning(f"You are over carbs by about {today_carbs - targets['carbs_g']:.0f}g today.")
    if today_protein < targets["protein_g"] - 5:
        st.info(f"Increase protein intake by about {targets['protein_g'] - today_protein:.0f}g today.")
    night_cal = float(meals_df.loc[meals_df["Hour"] >= 20, "Calories"].sum())
    total_cal = float(meals_df["Calories"].sum())
    if total_cal > 0 and (night_cal / total_cal) >= 0.4:
        st.write(f"**Pattern:** You consume about **{(night_cal / total_cal) * 100:.0f}%** of calories at night.")
    weekend_mask = meals_df["Weekday"].isin(["Saturday", "Sunday"])
    weekend_avg = float(meals_df.loc[weekend_mask].groupby("DateDT")["Calories"].sum().mean()) if weekend_mask.any() else 0.0
    weekday_avg = float(meals_df.loc[~weekend_mask].groupby("DateDT")["Calories"].sum().mean()) if (~weekend_mask).any() else 0.0
    if weekend_avg > 0 and weekday_avg > 0:
        diff_pct = ((weekend_avg - weekday_avg) / weekday_avg) * 100.0
        if diff_pct >= 15:
            st.write(f"**Pattern:** Weekend calories are about **{diff_pct:.0f}% higher** than weekdays.")
    if history_df is not None and len(history_df) > 0 and not sleep_daily_df.empty:
        cal_sleep = history_df.merge(sleep_daily_df, on="Date", how="inner")
        if len(cal_sleep) > 2:
            low_sleep_mean = float(cal_sleep.loc[cal_sleep["SleepHours"] < 7, "Calories"].mean()) if (cal_sleep["SleepHours"] < 7).any() else 0.0
            high_sleep_mean = float(cal_sleep.loc[cal_sleep["SleepHours"] >= 7, "Calories"].mean()) if (cal_sleep["SleepHours"] >= 7).any() else 0.0
            if low_sleep_mean > 0 and high_sleep_mean > 0 and low_sleep_mean > high_sleep_mean * 1.08:
                st.write("**Pattern:** Calorie intake tends to be higher on low-sleep days (<7h).")
    if history_df is not None and len(history_df) > 0 and not activity_daily_df.empty:
        cal_act = history_df.merge(activity_daily_df, on="Date", how="inner")
        if len(cal_act) > 2:
            low_steps_mean = float(cal_act.loc[cal_act["Steps"] < 7000, "Calories"].mean()) if (cal_act["Steps"] < 7000).any() else 0.0
            high_steps_mean = float(cal_act.loc[cal_act["Steps"] >= 7000, "Calories"].mean()) if (cal_act["Steps"] >= 7000).any() else 0.0
            if low_steps_mean > 0 and high_steps_mean > 0 and low_steps_mean > high_steps_mean * 1.08:
                st.write("**Pattern:** Calories are higher on lower-activity days (<7k steps).")
else:
    st.info("Not enough data yet to generate insights.")
st.markdown('</div>', unsafe_allow_html=True)