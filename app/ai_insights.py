from typing import List

from settings import (
    CARB_ALERT_GRAMS,
    LOW_SLEEP_HOURS,
    LOW_STEPS_THRESHOLD,
    PROTEIN_GAP_ALERT_GRAMS,
)


def build_daily_insights(
    *,
    today_calories: float,
    today_protein: float,
    today_carbs: float,
    calorie_goal: float,
    protein_goal: float,
    carbs_goal: float,
    goal: str,
    latest_sleep_hours: float | None = None,
    latest_steps: int | None = None,
) -> dict:
    info: List[str] = []
    warnings: List[str] = []
    suggestions: List[str] = []

    calorie_delta = float(today_calories) - float(calorie_goal)
    carb_delta = float(today_carbs) - float(carbs_goal)
    protein_gap = float(protein_goal) - float(today_protein)

    if calorie_delta > 0:
        warnings.append(f"You are over calories by {calorie_delta:.0f} kcal today.")
    else:
        info.append(f"You are within goal with {abs(calorie_delta):.0f} kcal remaining.")

    if carb_delta > CARB_ALERT_GRAMS:
        warnings.append(f"You are over carbs by {carb_delta:.0f} g today.")
    if protein_gap > PROTEIN_GAP_ALERT_GRAMS:
        info.append(f"Increase protein intake by {protein_gap:.0f} g to match your goal.")

    if goal == "Fat Loss":
        suggestions.extend([
            "Try grilled chicken + salad instead of pizza for better deficit control.",
            "Use high-protein, lower-oil options for your next meal.",
        ])
    elif goal == "Muscle Gain":
        suggestions.extend([
            "Add a lean protein snack to close your protein gap.",
            "Pair carbs with protein after training for recovery support.",
        ])
    else:
        suggestions.extend([
            "Balance each meal with protein + fiber for stable appetite.",
            "If late-night calories are high, shift intake earlier in the day.",
        ])

    if latest_sleep_hours is not None and latest_sleep_hours < LOW_SLEEP_HOURS:
        warnings.append("Low sleep detected. Poor sleep may increase cravings and reduce recovery.")
    if latest_steps is not None and latest_steps < LOW_STEPS_THRESHOLD:
        info.append("Activity is low today. A short walk can improve energy balance.")

    return {"info": info, "warnings": warnings, "suggestions": suggestions}
