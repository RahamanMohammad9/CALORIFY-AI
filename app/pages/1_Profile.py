import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import plotly.graph_objects as go
import streamlit as st

from profile_utils import (
    activity_multiplier,
    calculate_bmi,
    calculate_bmr,
    calculate_daily_calories,
    goal_calorie_delta,
    load_profile,
    macro_targets,
    normalize_goal_label,
    save_profile,
)
from utils import ACCENT, FAVICON_PATH, apply_glass_style, render_page_header


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Profile",
    page_icon=FAVICON_PATH,
    layout="wide"
)

apply_glass_style(st)


def bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def bmi_gauge_figure(bmi: float):
    bmi_clamped = min(max(float(bmi), 12.0), 45.0)
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


def goal_badge(goal: str) -> str:
    goal = normalize_goal_label(goal)
    if goal == "Fat Loss":
        bg = "rgba(245, 158, 11, 0.16)"
        border = "rgba(245, 158, 11, 0.35)"
        text = "#fde68a"
        label = "FAT LOSS PROFILE"
    elif goal == "Muscle Gain":
        bg = "rgba(59, 130, 246, 0.16)"
        border = "rgba(59, 130, 246, 0.35)"
        text = "#bfdbfe"
        label = "MUSCLE GAIN PROFILE"
    else:
        bg = "rgba(20, 184, 166, 0.16)"
        border = "rgba(20, 184, 166, 0.35)"
        text = "#ccfbf1"
        label = "MAINTENANCE PROFILE"

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


def build_profile_summary(goal: str, bmi: float, activity_level: str) -> tuple[str, str]:
    goal = normalize_goal_label(goal)
    bmi_status = bmi_category(bmi)

    if goal == "Fat Loss":
        action = "Focus on a small sustainable calorie deficit with high protein intake."
    elif goal == "Muscle Gain":
        action = "Support muscle gain with a modest calorie surplus and consistent protein."
    else:
        action = "Aim for stable intake and balanced macros to maintain current weight."

    if bmi_status == "Underweight":
        summary = "Your BMI is below the normal range, so aggressive calorie restriction would not be ideal."
    elif bmi_status == "Normal":
        summary = "Your BMI is currently in the normal range, which gives you flexibility to fine-tune your goal."
    elif bmi_status == "Overweight":
        summary = "Your BMI is above the normal range, so gradual calorie control and consistency would be useful."
    else:
        summary = "Your BMI is well above the normal range, so a steady and sustainable approach is more important than extreme restriction."

    activity_note = f"Current activity level is **{activity_level}**, which directly affects your maintenance calories."
    full_summary = f"{summary} {activity_note}"

    return full_summary, action


# -----------------------------
# Load profile
# -----------------------------
profile = load_profile()

render_page_header(
    st,
    "Profile",
    "Manage your personal details, calorie targets, and macro guidance for AI coaching.",
    kicker="Account",
)

# -----------------------------
# Live values
# -----------------------------
name_default = profile["name"]
age_default = int(profile["age"])
gender_default = profile["gender"]
height_default = float(profile["height_cm"])
weight_default = float(profile["weight_kg"])
activity_default = profile["activity_level"]
goal_default = normalize_goal_label(profile.get("goal", "Maintain"))

activity_options = [
    "Sedentary",
    "Lightly active",
    "Moderately active",
    "Very active",
    "Extra active"
]
goal_options = ["Fat Loss", "Maintain", "Muscle Gain"]

# -----------------------------
# Layout
# -----------------------------
left_col, right_col = st.columns([1.08, 1])

with left_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Personal Information")

    name = st.text_input("Name", value=name_default)

    a1, a2 = st.columns(2)
    with a1:
        age = st.number_input("Age", min_value=10, max_value=100, value=age_default, step=1)
    with a2:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"],
            index=0 if gender_default == "Male" else 1
        )

    h1, h2 = st.columns(2)
    with h1:
        height_cm = st.number_input(
            "Height (cm)",
            min_value=100.0,
            max_value=250.0,
            value=height_default,
            step=0.5
        )
    with h2:
        weight_kg = st.number_input(
            "Weight (kg)",
            min_value=20.0,
            max_value=300.0,
            value=weight_default,
            step=0.5
        )

    activity_index = activity_options.index(activity_default) if activity_default in activity_options else 2
    activity_level = st.selectbox(
        "Activity Level",
        activity_options,
        index=activity_index
    )

    goal_index = goal_options.index(goal_default) if goal_default in goal_options else 1
    goal = st.selectbox("Goal", goal_options, index=goal_index)

    st.markdown("### Save Profile")
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
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Calculations
# -----------------------------
bmi = calculate_bmi(weight_kg, height_cm)
bmi_status = bmi_category(bmi)
daily_calories = calculate_daily_calories(
    weight_kg, height_cm, age, gender, activity_level, goal
)
bmr_val = calculate_bmr(weight_kg, height_cm, age, gender)
mult = activity_multiplier(activity_level)
maintenance_val = bmr_val * mult
goal_delta = goal_calorie_delta(goal)
targets = macro_targets(daily_calories, weight_kg, goal)
profile_summary, best_action = build_profile_summary(goal, bmi, activity_level)

with right_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Health Summary")

    st.markdown(goal_badge(goal), unsafe_allow_html=True)

    m1, m2 = st.columns(2)
    m3, m4 = st.columns(2)

    m1.metric("BMI", f"{bmi:.1f}")
    m2.metric("BMI Category", bmi_status)
    m3.metric("Weight", f"{weight_kg:.1f} kg")
    m4.metric("Height", f"{height_cm:.1f} cm")

    st.plotly_chart(bmi_gauge_figure(bmi), width="stretch")

    st.markdown("### Personalized Daily Calories")
    st.metric("Suggested Daily Goal", f"{daily_calories:.0f} kcal")

    if bmi_status == "Normal":
        st.success("Your BMI is in the normal range.")
    elif bmi_status == "Underweight":
        st.warning("Your BMI is below the normal range.")
    else:
        st.warning("Your BMI is above the normal range.")

    st.markdown("### Profile AI Summary")
    st.write(profile_summary)

    st.markdown("##### Best direction")
    st.write(best_action)

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Macro targets
# -----------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Daily Macro Targets")

mc1, mc2, mc3 = st.columns(3)
mc1.metric("Protein Target", f"{targets['protein_g']:.0f} g")
mc2.metric("Carb Target", f"{targets['carbs_g']:.0f} g")
mc3.metric("Fat Guide", f"{targets['fat_g']:.0f} g")

st.caption("These targets are estimated from your calorie goal, body weight, and selected goal type.")
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Methodology
# -----------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("How Your Targets Are Calculated")

c1, c2, c3 = st.columns(3)
c1.metric("BMR", f"{bmr_val:.0f} kcal")
c2.metric("Activity Multiplier", f"× {mult:.3f}")
c3.metric("Maintenance Estimate", f"{maintenance_val:.0f} kcal")

if goal_delta != 0:
    st.write(
        f"Your selected goal applies a **{goal_delta:+d} kcal/day** adjustment to maintenance, "
        f"which gives a suggested daily target of **{daily_calories:.0f} kcal**."
    )
else:
    st.write(
        f"Your selected goal is **maintenance**, so your suggested intake stays close to "
        f"your estimated maintenance calories at **{daily_calories:.0f} kcal**."
    )

st.markdown("### Notes")
st.write("- BMR is estimated using the **Mifflin-St Jeor equation**.")
st.write("- Maintenance calories are estimated using your selected activity level.")
st.write("- Macro targets are coaching estimates, not medical prescriptions.")
st.write("- This page supports the AI coach across the rest of the app.")

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Quick recommendation
# -----------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Quick Recommendation")

if goal == "Fat Loss":
    st.write(
        f"A reasonable fat-loss target for your current profile is around **{daily_calories:.0f} kcal/day**, "
        "with a strong focus on protein and consistency."
    )
elif goal == "Muscle Gain":
    st.write(
        f"A reasonable muscle-gain target for your current profile is around **{daily_calories:.0f} kcal/day**, "
        "with enough protein and a modest calorie surplus."
    )
else:
    st.write(
        f"A reasonable maintenance target for your current profile is around **{daily_calories:.0f} kcal/day**, "
        "with balanced macros and stable daily intake."
    )

st.write("You can use this calorie target throughout the dashboard, tracker, and analytics pages.")
st.markdown('</div>', unsafe_allow_html=True)