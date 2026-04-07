import sys
from pathlib import Path
from datetime import date, timedelta

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import pandas as pd

from database import (
    create_table,
    get_all_meals,
    delete_meal,
    meal_exists
)
from utils import FAVICON_PATH, apply_glass_style, render_page_header

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Meal History",
    page_icon=FAVICON_PATH,
    layout="wide"
)

create_table()
apply_glass_style(st)

render_page_header(
    st,
    "Meal History",
    "Browse, filter, and manage all saved meal entries",
    kicker="Records",
)

# -----------------------------
# Load data
# -----------------------------
meals = get_all_meals()

if meals:
    meals_df = pd.DataFrame(meals, columns=[
        "ID", "Food", "Grams", "Calories", "Protein", "Carbs", "Fat", "Confidence", "Created At"
    ])
    meals_df["Created At"] = pd.to_datetime(meals_df["Created At"])
    meals_df["Date"] = meals_df["Created At"].dt.date
else:
    meals_df = pd.DataFrame(columns=[
        "ID", "Food", "Grams", "Calories", "Protein", "Carbs", "Fat", "Confidence", "Created At", "Date"
    ])

# -----------------------------
# Filters
# -----------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Filters")

f1, f2, f3, f4 = st.columns(4)

with f1:
    search_food = st.text_input("Search food name")

with f2:
    preset = st.selectbox(
        "Quick date filter",
        ["Custom", "Today", "Yesterday", "Last 7 days"],
        index=0,
    )

with f3:
    if not meals_df.empty:
        min_date = meals_df["Date"].min()
        max_date = meals_df["Date"].max()
        selected_dates = st.date_input(
            "Select date range",
            value=(min_date, max_date)
        )
    else:
        selected_dates = ()

with f4:
    sort_option = st.selectbox(
        "Sort by",
        ["Newest first", "Oldest first", "Highest calories", "Lowest calories"]
    )

filtered_df = meals_df.copy()

if search_food:
    filtered_df = filtered_df[
        filtered_df["Food"].str.contains(search_food, case=False, na=False)
    ]

if not filtered_df.empty:
    if preset == "Today":
        start_date = end_date = date.today()
    elif preset == "Yesterday":
        start_date = end_date = date.today() - timedelta(days=1)
    elif preset == "Last 7 days":
        end_date = date.today()
        start_date = end_date - timedelta(days=6)
    elif isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date = end_date = None

    if start_date is not None and end_date is not None:
        filtered_df = filtered_df[
            (filtered_df["Date"] >= start_date) &
            (filtered_df["Date"] <= end_date)
        ]

if sort_option == "Newest first":
    filtered_df = filtered_df.sort_values("Created At", ascending=False)
elif sort_option == "Oldest first":
    filtered_df = filtered_df.sort_values("Created At", ascending=True)
elif sort_option == "Highest calories":
    filtered_df = filtered_df.sort_values("Calories", ascending=False)
elif sort_option == "Lowest calories":
    filtered_df = filtered_df.sort_values("Calories", ascending=True)

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Summary
# -----------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("History Summary")

if not filtered_df.empty:
    total_meals = len(filtered_df)
    total_calories = filtered_df["Calories"].sum()
    avg_calories = filtered_df["Calories"].mean()
    avg_confidence = filtered_df["Confidence"].mean()

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Meals Shown", total_meals)
    s2.metric("Total Calories", f"{total_calories:.1f} kcal")
    s3.metric("Avg Calories/Meal", f"{avg_calories:.1f} kcal")
    s4.metric("Avg Confidence", f"{avg_confidence:.1f}%")
else:
    st.info("No meals match the selected filters.")
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Table
# -----------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Meal Entries")

if not filtered_df.empty:
    display_df = filtered_df.copy()
    display_df["Created At"] = display_df["Created At"].dt.strftime("%Y-%m-%d %H:%M:%S")
    display_df = display_df[[
        "ID", "Food", "Grams", "Calories", "Protein", "Carbs", "Fat", "Confidence", "Created At"
    ]]
    st.dataframe(display_df, width="stretch", height=360)
else:
    st.info("No meal entries available.")
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Quick insights
# -----------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Quick Insights")

if not filtered_df.empty:
    most_common_food = filtered_df["Food"].mode()[0]
    highest_calorie_meal = filtered_df.loc[filtered_df["Calories"].idxmax()]
    lowest_calorie_meal = filtered_df.loc[filtered_df["Calories"].idxmin()]

    i1, i2, i3 = st.columns(3)
    i1.metric("Most Common Food", most_common_food)
    i2.metric("Highest Calorie Meal", f'{highest_calorie_meal["Calories"]:.1f} kcal')
    i3.metric("Lowest Calorie Meal", f'{lowest_calorie_meal["Calories"]:.1f} kcal')

    st.write(
        f'Highest calorie meal shown: **{highest_calorie_meal["Food"]}** '
        f'({highest_calorie_meal["Calories"]:.1f} kcal)'
    )
    st.write(
        f'Lowest calorie meal shown: **{lowest_calorie_meal["Food"]}** '
        f'({lowest_calorie_meal["Calories"]:.1f} kcal)'
    )
else:
    st.info("Not enough data to generate insights.")
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Delete section
# -----------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Delete Meal Entry")

delete_col1, delete_col2 = st.columns([1, 1])

with delete_col1:
    meal_id_to_delete = st.number_input(
        "Enter Meal ID to delete",
        min_value=1,
        step=1,
        key="delete_meal_history_id"
    )

with delete_col2:
    st.write("")
    st.write("")
    if st.button("🗑️ Delete Selected Meal", width="stretch"):
        if meal_exists(meal_id_to_delete):
            delete_meal(meal_id_to_delete)
            st.success(f"Meal ID {meal_id_to_delete} deleted.")
            st.rerun()
        else:
            st.warning("That Meal ID does not exist.")

st.markdown('</div>', unsafe_allow_html=True)