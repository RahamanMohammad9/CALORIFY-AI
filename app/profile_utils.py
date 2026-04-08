import json
import os
from typing import Any


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILE_PATH = os.path.join(BASE_DIR, "user_profile.json")


DEFAULT_PROFILE = {
    "name": "",
    "age": 21,
    "gender": "Male",
    "height_cm": 170.0,
    "weight_kg": 70.0,
    "activity_level": "Moderately active",
    "goal": "Maintain",
}


VALID_GENDERS = {"Male", "Female"}
VALID_ACTIVITY_LEVELS = {
    "Sedentary",
    "Lightly active",
    "Moderately active",
    "Very active",
    "Extra active",
}
VALID_GOALS = {"Fat Loss", "Maintain", "Muscle Gain"}


def _safe_float(value: Any, default: float) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def normalize_goal_label(goal: str) -> str:
    g = str(goal or "").strip().lower()
    if g in {"lose weight", "fat loss", "weight loss", "cut", "cutting"}:
        return "Fat Loss"
    if g in {"gain weight", "muscle gain", "bulk", "bulking"}:
        return "Muscle Gain"
    return "Maintain"


def normalize_gender(gender: str) -> str:
    g = str(gender or "").strip().title()
    return g if g in VALID_GENDERS else DEFAULT_PROFILE["gender"]


def normalize_activity_level(activity_level: str) -> str:
    a = str(activity_level or "").strip()
    return a if a in VALID_ACTIVITY_LEVELS else DEFAULT_PROFILE["activity_level"]


def sanitize_profile(profile_data: dict | None) -> dict:
    raw = {**DEFAULT_PROFILE, **(profile_data or {})}

    cleaned = {
        "name": str(raw.get("name", DEFAULT_PROFILE["name"])).strip(),
        "age": _safe_int(raw.get("age"), DEFAULT_PROFILE["age"]),
        "gender": normalize_gender(raw.get("gender", DEFAULT_PROFILE["gender"])),
        "height_cm": _safe_float(raw.get("height_cm"), DEFAULT_PROFILE["height_cm"]),
        "weight_kg": _safe_float(raw.get("weight_kg"), DEFAULT_PROFILE["weight_kg"]),
        "activity_level": normalize_activity_level(
            raw.get("activity_level", DEFAULT_PROFILE["activity_level"])
        ),
        "goal": normalize_goal_label(raw.get("goal", DEFAULT_PROFILE["goal"])),
    }

    cleaned["age"] = min(max(cleaned["age"], 10), 100)
    cleaned["height_cm"] = min(max(cleaned["height_cm"], 100.0), 250.0)
    cleaned["weight_kg"] = min(max(cleaned["weight_kg"], 20.0), 300.0)

    return cleaned


def load_profile() -> dict:
    if os.path.exists(PROFILE_PATH):
        try:
            with open(PROFILE_PATH, "r", encoding="utf-8") as f:
                profile = json.load(f)
                return sanitize_profile(profile)
        except (json.JSONDecodeError, OSError, TypeError):
            pass
    return DEFAULT_PROFILE.copy()


def save_profile(profile_data: dict) -> None:
    cleaned = sanitize_profile(profile_data)
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=4)


def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    weight_kg = _safe_float(weight_kg, 0.0)
    height_cm = _safe_float(height_cm, 0.0)

    height_m = height_cm / 100.0
    if height_m <= 0:
        return 0.0
    return weight_kg / (height_m ** 2)


def bmi_category(bmi: float) -> str:
    bmi = _safe_float(bmi, 0.0)
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def activity_multiplier(activity_level: str) -> float:
    mapping = {
        "Sedentary": 1.20,
        "Lightly active": 1.375,
        "Moderately active": 1.55,
        "Very active": 1.725,
        "Extra active": 1.90,
    }
    return mapping.get(normalize_activity_level(activity_level), 1.55)


def calculate_bmr(weight_kg: float, height_cm: float, age: int, gender: str) -> float:
    weight_kg = _safe_float(weight_kg, DEFAULT_PROFILE["weight_kg"])
    height_cm = _safe_float(height_cm, DEFAULT_PROFILE["height_cm"])
    age = _safe_int(age, DEFAULT_PROFILE["age"])
    gender = normalize_gender(gender)

    if gender == "Male":
        return 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    return 10 * weight_kg + 6.25 * height_cm - 5 * age - 161


def goal_calorie_delta(goal: str) -> int:
    g = normalize_goal_label(goal)
    if g == "Fat Loss":
        return -300
    if g == "Muscle Gain":
        return 300
    return 0


def calculate_daily_calories(
    weight_kg: float,
    height_cm: float,
    age: int,
    gender: str,
    activity_level: str,
    goal: str,
) -> float:
    bmr = calculate_bmr(weight_kg, height_cm, age, gender)
    maintenance = bmr * activity_multiplier(activity_level)
    return maintenance + goal_calorie_delta(goal)


def macro_targets(calorie_target: float, weight_kg: float, goal: str) -> dict:
    calorie_target = max(_safe_float(calorie_target, 0.0), 0.0)
    weight_kg = max(_safe_float(weight_kg, DEFAULT_PROFILE["weight_kg"]), 1.0)
    g = normalize_goal_label(goal)

    if g == "Fat Loss":
        protein_g = weight_kg * 2.0
        fat_ratio = 0.25
    elif g == "Muscle Gain":
        protein_g = weight_kg * 1.8
        fat_ratio = 0.28
    else:
        protein_g = weight_kg * 1.6
        fat_ratio = 0.27

    fat_g = (calorie_target * fat_ratio) / 9.0
    protein_kcal = protein_g * 4.0
    fat_kcal = fat_g * 9.0
    carbs_kcal = max(calorie_target - protein_kcal - fat_kcal, 0.0)
    carbs_g = carbs_kcal / 4.0

    return {
        "protein_g": protein_g,
        "carbs_g": carbs_g,
        "fat_g": fat_g,
    }


def maintenance_calories(
    weight_kg: float,
    height_cm: float,
    age: int,
    gender: str,
    activity_level: str,
) -> float:
    bmr = calculate_bmr(weight_kg, height_cm, age, gender)
    return bmr * activity_multiplier(activity_level)


def profile_summary(profile: dict | None = None) -> dict:
    p = sanitize_profile(profile if profile is not None else load_profile())

    bmi = calculate_bmi(p["weight_kg"], p["height_cm"])
    maintenance = maintenance_calories(
        p["weight_kg"],
        p["height_cm"],
        p["age"],
        p["gender"],
        p["activity_level"],
    )
    daily_target = calculate_daily_calories(
        p["weight_kg"],
        p["height_cm"],
        p["age"],
        p["gender"],
        p["activity_level"],
        p["goal"],
    )
    targets = macro_targets(daily_target, p["weight_kg"], p["goal"])

    return {
        "profile": p,
        "bmi": bmi,
        "bmi_category": bmi_category(bmi),
        "maintenance_calories": maintenance,
        "daily_target_calories": daily_target,
        "macro_targets": targets,
    }