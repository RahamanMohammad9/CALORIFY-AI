import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from database import (
    create_table,
    get_all_meals,
    get_today_totals,
    get_daily_calorie_history,
    get_weekly_summary,
    delete_meal,
    meal_exists
)
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
    "View saved meals, calorie progress, weekly nutrition, and meal history.",
    kicker="Tracking",
)

with st.sidebar:
    st.header("Tracker Settings")
    calorie_goal = st.number_input(
        "Daily calorie goal",
        min_value=1000,
        max_value=5000,
        value=2000,
        step=50
    )

# -----------------------------
# Today's summary
# -----------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Today's Nutrition Summary")

today_calories, today_protein, today_carbs, today_fat = get_today_totals()

summary_col1, summary_col2 = st.columns([1, 1.4])

with summary_col1:
    progress_value = min(today_calories / calorie_goal, 1.0) if calorie_goal > 0 else 0
    progress_percent = progress_value * 100

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

    remaining = calorie_goal - today_calories

    if remaining > 0:
        st.success(f"You have {remaining:.1f} kcal remaining for today.")
    else:
        st.warning(f"You exceeded your goal by {abs(remaining):.1f} kcal today.")

    st.progress(
        min(progress_value, 1.0),
        text=f"Goal progress: {today_calories:.1f} / {calorie_goal} kcal ({progress_percent:.1f}%)"
    )

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Charts
# -----------------------------
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Daily Calorie Trend")
    history = get_daily_calorie_history()

    if history:
        history_df = pd.DataFrame(history, columns=["Date", "Calories"])

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
    week_calories, week_protein, week_carbs, week_fat = get_weekly_summary()

    w1, w2 = st.columns(2)
    w3, w4 = st.columns(2)

    w1.metric("Weekly Calories", f"{week_calories:.1f} kcal")
    w2.metric("Weekly Protein", f"{week_protein:.1f} g")
    w3.metric("Weekly Carbs", f"{week_carbs:.1f} g")
    w4.metric("Weekly Fat", f"{week_fat:.1f} g")

    st.write("**Weekly Overview**")
    st.write("Use this page to track saved meals, compare intake against your calorie goal, and manage meal history.")
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

meals = get_all_meals()

if meals:
    history_df = pd.DataFrame(meals, columns=[
        "ID", "Food", "Grams", "Calories", "Protein", "Carbs", "Fat", "Confidence", "Created At"
    ])
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

    display_df = history_df
    if meal_search.strip():
        mask = display_df["Food"].astype(str).str.contains(
            meal_search.strip(), case=False, na=False
        )
        display_df = display_df.loc[mask]

    if row_choice != "All":
        n = int(row_choice)
        display_df = display_df.tail(n)

    if display_df.empty and meal_search.strip():
        st.info("No meals match that search. Clear the filter to see all meals.")
    else:
        st.dataframe(display_df, width="stretch", height=350)
        if meal_search.strip() or row_choice != "All":
            st.caption(f"Showing **{len(display_df)}** of **{len(history_df)}** meals.")

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