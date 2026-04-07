import streamlit as st

from database import create_table
from utils import FAVICON_PATH, apply_glass_style, render_page_header

st.set_page_config(
    page_title="Calorify AI",
    page_icon=FAVICON_PATH,
    layout="wide",
)

create_table()
apply_glass_style(st)

render_page_header(
    st,
    "Menu",
    "Open **Dashboard** in the sidebar for your overview, charts, and icon shortcuts to every section.",
    kicker="Start here",
)

st.info(
    "This app provides AI-based nutrition estimates and is not medical advice."
)

st.markdown("### Pages")
st.caption("Pick a page from the sidebar, or go to the dashboard first.")
st.page_link("app.py", label="Dashboard (overview & shortcuts)")
st.page_link("pages/2_Add_Meal.py", label="Add Meal")
st.page_link("pages/1_Meal_Tracker.py", label="Meal Tracker")
st.page_link("pages/3_Meal_History.py", label="Meal History")
st.page_link("pages/4_Analytics.py", label="Analytics")
st.page_link("pages/5_Water_Tracker.py", label="Water Tracker")
st.page_link("pages/6_Weight_Tracker.py", label="Weight Tracker")
st.page_link("pages/7_Sleep_Tracker.py", label="Sleep Tracker")
st.page_link("pages/8_Activity_Tracker.py", label="Activity Tracker")
st.page_link("pages/1_Profile.py", label="Profile")
