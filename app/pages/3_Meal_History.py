import sys
from datetime import date, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

from database import (
    create_table,
    delete_meal,
    get_all_meals,
    meal_exists,
)
from utils import FAVICON_PATH, apply_glass_style, render_page_header


def _history_status_badge(total_meals: int, avg_confidence: float) -> str:
    if total_meals == 0:
        label = "NO DATA"
        bg = "rgba(100, 116, 139, 0.18)"
        border = "rgba(148, 163, 184, 0.28)"
        text = "#cbd5e1"
    elif avg_confidence >= 85:
        label = "STRONG DATA QUALITY"
        bg = "rgba(20, 184, 166, 0.16)"
        border = "rgba(20, 184, 166, 0.35)"
        text = "#ccfbf1"
    elif avg_confidence >= 70:
        label = "MODERATE DATA QUALITY"
        bg = "rgba(245, 158, 11, 0.16)"
        border = "rgba(245, 158, 11, 0.35)"
        text = "#fde68a"
    else:
        label = "LOWER DATA CONFIDENCE"
        bg = "rgba(239, 68, 68, 0.18)"
        border = "rgba(239, 68, 68, 0.35)"
        text = "#fecaca"

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
    "Browse, filter, review, and manage all saved meal records in one place.",
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
    meals_df["Time"] = meals_df["Created At"].dt.strftime("%H:%M:%S")
else:
    meals_df = pd.DataFrame(columns=[
        "ID", "Food", "Grams", "Calories", "Protein", "Carbs", "Fat", "Confidence", "Created At", "Date", "Time"
    ])

# -----------------------------
# Filters
# -----------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Filters")

f1, f2, f3, f4 = st.columns(4)

with f1:
    search_food = st.text_input(
        "Search food name",
        placeholder="e.g. pizza, rice, chicken"
    )

with f2:
    preset = st.selectbox(
        "Quick date filter",
        ["Custom", "Today", "Yesterday", "Last 7 days", "Last 30 days"],
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
        [
            "Newest first",
            "Oldest first",
            "Highest calories",
            "Lowest calories",
            "Highest confidence",
            "Lowest confidence",
        ]
    )

f5, f6, f7 = st.columns(3)

with f5:
    min_conf = st.slider(
        "Minimum confidence (%)",
        min_value=0,
        max_value=100,
        value=0,
        step=5
    )

with f6:
    min_cal = st.number_input(
        "Minimum calories",
        min_value=0.0,
        value=0.0,
        step=10.0
    )

with f7:
    max_cal = st.number_input(
        "Maximum calories",
        min_value=0.0,
        value=5000.0,
        step=50.0
    )

filtered_df = meals_df.copy()

if search_food:
    filtered_df = filtered_df[
        filtered_df["Food"].astype(str).str.contains(search_food, case=False, na=False)
    ]

if not filtered_df.empty:
    if preset == "Today":
        start_date = end_date = date.today()
    elif preset == "Yesterday":
        start_date = end_date = date.today() - timedelta(days=1)
    elif preset == "Last 7 days":
        end_date = date.today()
        start_date = end_date - timedelta(days=6)
    elif preset == "Last 30 days":
        end_date = date.today()
        start_date = end_date - timedelta(days=29)
    elif isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date = end_date = None

    if start_date is not None and end_date is not None:
        filtered_df = filtered_df[
            (filtered_df["Date"] >= start_date) &
            (filtered_df["Date"] <= end_date)
        ]

if not filtered_df.empty:
    filtered_df = filtered_df[
        (filtered_df["Confidence"] >= float(min_conf)) &
        (filtered_df["Calories"] >= float(min_cal)) &
        (filtered_df["Calories"] <= float(max_cal))
    ]

if sort_option == "Newest first":
    filtered_df = filtered_df.sort_values("Created At", ascending=False)
elif sort_option == "Oldest first":
    filtered_df = filtered_df.sort_values("Created At", ascending=True)
elif sort_option == "Highest calories":
    filtered_df = filtered_df.sort_values("Calories", ascending=False)
elif sort_option == "Lowest calories":
    filtered_df = filtered_df.sort_values("Calories", ascending=True)
elif sort_option == "Highest confidence":
    filtered_df = filtered_df.sort_values("Confidence", ascending=False)
elif sort_option == "Lowest confidence":
    filtered_df = filtered_df.sort_values("Confidence", ascending=True)

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Summary
# -----------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("History Summary")

if not filtered_df.empty:
    total_meals = len(filtered_df)
    total_calories = float(filtered_df["Calories"].sum())
    avg_calories = float(filtered_df["Calories"].mean())
    avg_confidence = float(filtered_df["Confidence"].mean())
    total_protein = float(filtered_df["Protein"].sum())
    total_carbs = float(filtered_df["Carbs"].sum())
    total_fat = float(filtered_df["Fat"].sum())

    st.markdown(_history_status_badge(total_meals, avg_confidence), unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Meals Shown", total_meals)
    s2.metric("Total Calories", f"{total_calories:.1f} kcal")
    s3.metric("Avg Calories/Meal", f"{avg_calories:.1f} kcal")
    s4.metric("Avg Confidence", f"{avg_confidence:.1f}%")

    s5, s6, s7 = st.columns(3)
    s5.metric("Total Protein", f"{total_protein:.1f} g")
    s6.metric("Total Carbs", f"{total_carbs:.1f} g")
    s7.metric("Total Fat", f"{total_fat:.1f} g")
else:
    st.info("No meals match the selected filters.")

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
    highest_conf_meal = filtered_df.loc[filtered_df["Confidence"].idxmax()]
    lowest_conf_meal = filtered_df.loc[filtered_df["Confidence"].idxmin()]

    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Most Common Food", most_common_food)
    i2.metric("Highest Calorie", f'{highest_calorie_meal["Calories"]:.1f} kcal')
    i3.metric("Lowest Calorie", f'{lowest_calorie_meal["Calories"]:.1f} kcal')
    i4.metric("Best Confidence", f'{highest_conf_meal["Confidence"]:.1f}%')

    st.write(
        f'Highest calorie meal shown: **{highest_calorie_meal["Food"]}** '
        f'({highest_calorie_meal["Calories"]:.1f} kcal)'
    )
    st.write(
        f'Lowest calorie meal shown: **{lowest_calorie_meal["Food"]}** '
        f'({lowest_calorie_meal["Calories"]:.1f} kcal)'
    )
    st.write(
        f'Lowest-confidence meal shown: **{lowest_conf_meal["Food"]}** '
        f'({lowest_conf_meal["Confidence"]:.1f}% confidence)'
    )

    if avg_confidence < 70:
        st.warning("Your filtered history contains several lower-confidence predictions. Review uncertain meals when using the data for analysis.")
    elif avg_confidence >= 85:
        st.success("This filtered history has strong average prediction confidence.")
else:
    st.info("Not enough data to generate insights.")

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Table
# -----------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Meal Entries")

if not filtered_df.empty:
    row_choice_col1, row_choice_col2 = st.columns([1, 2])

    with row_choice_col1:
        rows_to_show = st.select_slider(
            "Rows to display",
            options=["10", "25", "50", "100", "All"],
            value="25",
        )

    display_df = filtered_df.copy()

    if rows_to_show != "All":
        display_df = display_df.head(int(rows_to_show))

    display_df["Created At"] = display_df["Created At"].dt.strftime("%Y-%m-%d %H:%M:%S")
    display_df = display_df[[
        "ID", "Food", "Grams", "Calories", "Protein", "Carbs", "Fat", "Confidence", "Created At"
    ]]

    st.dataframe(display_df, width="stretch", height=380)
    st.caption(f"Showing **{len(display_df)}** of **{len(filtered_df)}** filtered meals.")
else:
    st.info("No meal entries available.")

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Drill-down lookup
# -----------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Meal Record Lookup")

if not meals_df.empty:
    lookup_id = st.number_input(
        "Enter Meal ID to inspect",
        min_value=1,
        step=1,
        key="history_lookup_id"
    )

    lookup_row = meals_df[meals_df["ID"] == int(lookup_id)]

    if len(lookup_row) > 0:
        row = lookup_row.iloc[0]

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Food", str(row["Food"]))
        d2.metric("Calories", f'{float(row["Calories"]):.1f} kcal')
        d3.metric("Confidence", f'{float(row["Confidence"]):.1f}%')
        d4.metric("Portion", f'{float(row["Grams"]):.1f} g')

        d5, d6, d7 = st.columns(3)
        d5.metric("Protein", f'{float(row["Protein"]):.1f} g')
        d6.metric("Carbs", f'{float(row["Carbs"]):.1f} g')
        d7.metric("Fat", f'{float(row["Fat"]):.1f} g')

        st.write(f'**Recorded at:** {pd.to_datetime(row["Created At"]).strftime("%Y-%m-%d %H:%M:%S")}')

        if float(row["Confidence"]) < 60:
            st.warning("This meal has low confidence. Consider checking whether the prediction and macros are accurate.")
        elif float(row["Confidence"]) >= 85:
            st.success("This meal has strong prediction confidence.")
else:
    st.info("No saved meals yet.")

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