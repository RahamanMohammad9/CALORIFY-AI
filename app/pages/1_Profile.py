import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import plotly.graph_objects as go

from utils import ACCENT, FAVICON_PATH, apply_glass_style, render_page_header
from profile_utils import (
    activity_multiplier,
    calculate_bmi,
    calculate_bmr,
    calculate_daily_calories,
    goal_calorie_delta,
    load_profile,
    normalize_goal_label,
    save_profile,
)

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Profile",
    page_icon=FAVICON_PATH,
    layout="wide"
)

apply_glass_style(st)

def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    return "Obese"

def bmi_gauge_figure(bmi):
    bmi_clamped = min(max(bmi, 12.0), 45.0)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=bmi_clamped,
            number={"font": {"size": 26, "color": "white"}, "valueformat": ".1f"},
            title={"text": "BMI (live)", "font": {"size": 15, "color": "#94A3B8"}},
            gauge={
                "axis": {"range": [15, 40], "tickcolor": "#94A3B8"},
                "bar": {"color": ACCENT},
                "bgcolor": "#1A1F2B",
                "borderwidth": 1,
                "bordercolor": "rgba(255,255,255,0.12)",
                "steps": [
                    {"range": [15, 18.5], "color": "rgba(59, 130, 246, 0.35)"},
                    {"range": [18.5, 25], "color": "rgba(34, 197, 94, 0.35)"},
                    {"range": [25, 30], "color": "rgba(250, 204, 21, 0.35)"},
                    {"range": [30, 40], "color": "rgba(239, 68, 68, 0.35)"},
                ],
            },
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        margin=dict(l=24, r=24, t=40, b=24),
        height=260,
    )
    return fig

# -----------------------------
# Load profile
# -----------------------------
profile = load_profile()

render_page_header(
    st,
    "Profile",
    "Manage your personal details and estimate your daily calorie target",
    kicker="Account",
)

# -----------------------------
# Layout
# -----------------------------
left_col, right_col = st.columns([1.1, 1])

with left_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Personal Information")

    name = st.text_input("Name", value=profile["name"])
    age = st.number_input("Age", min_value=10, max_value=100, value=int(profile["age"]), step=1)
    gender = st.selectbox("Gender", ["Male", "Female"], index=0 if profile["gender"] == "Male" else 1)

    height_cm = st.number_input(
        "Height (cm)",
        min_value=100.0,
        max_value=250.0,
        value=float(profile["height_cm"]),
        step=0.5
    )

    weight_kg = st.number_input(
        "Weight (kg)",
        min_value=20.0,
        max_value=300.0,
        value=float(profile["weight_kg"]),
        step=0.5
    )

    activity_options = [
        "Sedentary",
        "Lightly active",
        "Moderately active",
        "Very active",
        "Extra active"
    ]
    activity_index = activity_options.index(profile["activity_level"]) if profile["activity_level"] in activity_options else 2

    activity_level = st.selectbox(
        "Activity Level",
        activity_options,
        index=activity_index
    )

    goal_options = ["Fat Loss", "Maintain", "Muscle Gain"]
    normalized_goal = normalize_goal_label(profile.get("goal", "Maintain"))
    goal_index = goal_options.index(normalized_goal) if normalized_goal in goal_options else 1

    goal = st.selectbox("Goal", goal_options, index=goal_index)

    if st.button("💾 Save Profile", width="stretch"):
        updated_profile = {
            "name": name,
            "age": int(age),
            "gender": gender,
            "height_cm": float(height_cm),
            "weight_kg": float(weight_kg),
            "activity_level": activity_level,
            "goal": goal
        }
        save_profile(updated_profile)
        st.success("Profile saved successfully.")
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Health Summary")

    bmi = calculate_bmi(weight_kg, height_cm)
    bmi_status = bmi_category(bmi)
    daily_calories = calculate_daily_calories(
        weight_kg, height_cm, age, gender, activity_level, goal
    )
    bmr_val = calculate_bmr(weight_kg, height_cm, age, gender)
    mult = activity_multiplier(activity_level)
    maintenance_val = bmr_val * mult
    goal_delta = goal_calorie_delta(goal)

    m1, m2 = st.columns(2)
    m3, m4 = st.columns(2)

    m1.metric("BMI", f"{bmi:.1f}")
    m2.metric("BMI Category", bmi_status)
    m3.metric("Weight", f"{weight_kg:.1f} kg")
    m4.metric("Height", f"{height_cm:.1f} cm")

    st.plotly_chart(bmi_gauge_figure(bmi), width="stretch")

    st.markdown("### Estimated Daily Calories")
    st.metric("Suggested Daily Goal", f"{daily_calories:.0f} kcal")

    if bmi_status == "Normal":
        st.success("Your BMI is in the normal range.")
    elif bmi_status == "Underweight":
        st.warning("Your BMI is below the normal range.")
    else:
        st.warning("Your BMI is above the normal range.")

    with st.expander("How your calorie target is calculated", expanded=False):
        st.write("Values update as you change age, weight, activity, or goal on the left.")
        c1, c2, c3 = st.columns(3)
        c1.metric("BMR (Mifflin-St Jeor)", f"{bmr_val:.0f} kcal")
        c2.metric("Activity multiplier", f"× {mult:.3f}")
        c3.metric("Maintenance (approx.)", f"{maintenance_val:.0f} kcal")
        if goal_delta != 0:
            st.caption(
                f"Goal adjustment: **{goal_delta:+d} kcal/day** vs maintenance → **{daily_calories:.0f} kcal** suggested."
            )
        else:
            st.caption("Goal is maintenance — suggested intake matches estimated TDEE.")

    st.markdown("### Notes")
    st.write("- Daily calories are estimated using the Mifflin-St Jeor equation.")
    st.write("- This is only a general estimate, not medical advice.")
    st.write("- You can use this value as your calorie goal in the tracker.")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Quick goal helper
# -----------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Quick Recommendation")

if goal == "Fat Loss":
    st.write(f"A reasonable calorie target for weight loss is around **{daily_calories:.0f} kcal/day**.")
elif goal == "Muscle Gain":
    st.write(f"A reasonable calorie target for weight gain is around **{daily_calories:.0f} kcal/day**.")
else:
    st.write(f"A reasonable calorie target for weight maintenance is around **{daily_calories:.0f} kcal/day**.")

st.write("You can copy this calorie target into your dashboard or tracker settings.")
st.markdown('</div>', unsafe_allow_html=True)