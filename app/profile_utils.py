import json
import os


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


def normalize_goal_label(goal: str) -> str:
    g = str(goal or "").strip().lower()
    if g in {"lose weight", "fat loss"}:
        return "Fat Loss"
    if g in {"gain weight", "muscle gain"}:
        return "Muscle Gain"
    return "Maintain"


def load_profile() -> dict:
    if os.path.exists(PROFILE_PATH):
        try:
            with open(PROFILE_PATH, "r", encoding="utf-8") as f:
                profile = json.load(f)
                profile["goal"] = normalize_goal_label(profile.get("goal", "Maintain"))
                return {**DEFAULT_PROFILE, **profile}
        except (json.JSONDecodeError, OSError):
            pass
    return DEFAULT_PROFILE.copy()


def save_profile(profile_data: dict) -> None:
    cleaned = {**DEFAULT_PROFILE, **profile_data}
    cleaned["goal"] = normalize_goal_label(cleaned.get("goal", "Maintain"))
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=4)


def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    height_m = float(height_cm) / 100.0
    if height_m <= 0:
        return 0.0
    return float(weight_kg) / (height_m ** 2)


def activity_multiplier(activity_level: str) -> float:
    mapping = {
        "Sedentary": 1.2,
        "Lightly active": 1.375,
        "Moderately active": 1.55,
        "Very active": 1.725,
        "Extra active": 1.9,
    }
    return mapping.get(activity_level, 1.55)


def calculate_bmr(weight_kg: float, height_cm: float, age: int, gender: str) -> float:
    if gender == "Male":
        return 10 * float(weight_kg) + 6.25 * float(height_cm) - 5 * int(age) + 5
    return 10 * float(weight_kg) + 6.25 * float(height_cm) - 5 * int(age) - 161


def goal_calorie_delta(goal: str) -> int:
    g = normalize_goal_label(goal)
    if g == "Fat Loss":
        return -300
    if g == "Muscle Gain":
        return 300
    return 0


def calculate_daily_calories(
    weight_kg: float, height_cm: float, age: int, gender: str, activity_level: str, goal: str
) -> float:
    bmr = calculate_bmr(weight_kg, height_cm, age, gender)
    maintenance = bmr * activity_multiplier(activity_level)
    return maintenance + goal_calorie_delta(goal)


def macro_targets(calorie_target: float, weight_kg: float, goal: str) -> dict:
    g = normalize_goal_label(goal)
    if g == "Fat Loss":
        protein_g = float(weight_kg) * 2.0
        fat_ratio = 0.25
    elif g == "Muscle Gain":
        protein_g = float(weight_kg) * 1.8
        fat_ratio = 0.28
    else:
        protein_g = float(weight_kg) * 1.6
        fat_ratio = 0.27

    fat_g = (float(calorie_target) * fat_ratio) / 9.0
    protein_kcal = protein_g * 4.0
    fat_kcal = fat_g * 9.0
    carbs_kcal = max(float(calorie_target) - protein_kcal - fat_kcal, 0.0)
    carbs_g = carbs_kcal / 4.0
    return {"protein_g": protein_g, "carbs_g": carbs_g, "fat_g": fat_g}
