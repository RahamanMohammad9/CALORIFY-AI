from __future__ import annotations

from typing import Any

from settings import (
    CARB_ALERT_GRAMS,
    HIGH_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    LOW_SLEEP_HOURS,
    LOW_STEPS_THRESHOLD,
    LOW_WATER_THRESHOLD_ML,
    PROTEIN_GAP_ALERT_GRAMS,
    RECENT_DAYS_LOOKBACK,
    WEIGHT_PROJECTION_DAYS,
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_goal(goal: str) -> str:
    g = str(goal or "").strip().lower()
    if g in {"fat loss", "lose weight", "weight loss", "cut", "cutting"}:
        return "Fat Loss"
    if g in {"muscle gain", "gain weight", "bulk", "bulking"}:
        return "Muscle Gain"
    return "Maintain"


def _mean(values: list[float]) -> float:
    nums = [_safe_float(v) for v in values if v is not None]
    return sum(nums) / len(nums) if nums else 0.0


def _count_if(values: list[Any], fn) -> int:
    return sum(1 for v in values if v is not None and fn(v))


def _last_n(values: list[Any], n: int) -> list[Any]:
    return list(values[-max(1, int(n)):]) if values else []


def _unique(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        cleaned = str(item).strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            out.append(cleaned)
    return out


def _confidence_band(confidence_pct: float) -> str:
    conf = _safe_float(confidence_pct)
    if conf >= HIGH_CONFIDENCE_THRESHOLD:
        return "high"
    if conf >= LOW_CONFIDENCE_THRESHOLD:
        return "medium"
    return "low"


def _priority_label(score: int) -> str:
    if score >= 8:
        return "high"
    if score >= 4:
        return "medium"
    return "low"


def _weight_projection_from_calorie_delta(
    current_weight_kg: float,
    avg_daily_calorie_delta: float,
    projection_days: int = WEIGHT_PROJECTION_DAYS,
) -> dict:
    current_weight_kg = _safe_float(current_weight_kg)
    avg_daily_calorie_delta = _safe_float(avg_daily_calorie_delta)
    projection_days = max(1, _safe_int(projection_days, 30))

    projected_change = (avg_daily_calorie_delta * projection_days) / 7700.0
    projected_weight = current_weight_kg + projected_change

    return {
        "projected_change_kg": projected_change,
        "projected_weight_kg": projected_weight,
        "days": projection_days,
    }


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
    latest_water_ml: float | None = None,
    water_goal_ml: float | None = None,
    recent_calories: list[float] | None = None,
    recent_protein: list[float] | None = None,
    recent_carbs: list[float] | None = None,
    recent_sleep_hours: list[float] | None = None,
    recent_steps: list[int] | None = None,
    recent_weights: list[float] | None = None,
    current_weight_kg: float | None = None,
    latest_prediction_confidence: float | None = None,
) -> dict:
    """
    AI coaching engine for Calorify AI.

    Backward-compatible keys:
        - warnings
        - info
        - suggestions

    Richer keys:
        - critical
        - patterns
        - wins
        - summary
        - main_issue
        - best_action
        - priority
        - confidence_band
        - projections
        - scores
    """

    goal = _normalize_goal(goal)

    today_calories = _safe_float(today_calories)
    today_protein = _safe_float(today_protein)
    today_carbs = _safe_float(today_carbs)
    calorie_goal = max(_safe_float(calorie_goal), 1.0)
    protein_goal = max(_safe_float(protein_goal), 1.0)
    carbs_goal = max(_safe_float(carbs_goal), 1.0)

    latest_sleep_hours = None if latest_sleep_hours is None else _safe_float(latest_sleep_hours)
    latest_steps = None if latest_steps is None else _safe_int(latest_steps)
    latest_water_ml = None if latest_water_ml is None else _safe_float(latest_water_ml)
    water_goal_ml = None if water_goal_ml is None else max(_safe_float(water_goal_ml), 1.0)

    recent_calories = _last_n(recent_calories or [], RECENT_DAYS_LOOKBACK)
    recent_protein = _last_n(recent_protein or [], RECENT_DAYS_LOOKBACK)
    recent_carbs = _last_n(recent_carbs or [], RECENT_DAYS_LOOKBACK)
    recent_sleep_hours = _last_n(recent_sleep_hours or [], RECENT_DAYS_LOOKBACK)
    recent_steps = _last_n(recent_steps or [], RECENT_DAYS_LOOKBACK)
    recent_weights = _last_n(recent_weights or [], RECENT_DAYS_LOOKBACK)

    critical: list[str] = []
    warnings: list[str] = []
    info: list[str] = []
    suggestions: list[str] = []
    patterns: list[str] = []
    wins: list[str] = []

    severity_score = 0

    calorie_delta = today_calories - calorie_goal
    calorie_delta_pct = (calorie_delta / calorie_goal) * 100.0

    protein_gap = protein_goal - today_protein
    protein_gap_pct = (protein_gap / protein_goal) * 100.0

    carb_delta = today_carbs - carbs_goal
    carb_delta_pct = (carb_delta / carbs_goal) * 100.0 if carbs_goal > 0 else 0.0

    # --------------------------------------------------
    # Core daily assessment
    # --------------------------------------------------
    if calorie_delta > calorie_goal * 0.15:
        critical.append(f"You are significantly over your calorie target by {calorie_delta:.0f} kcal today.")
        severity_score += 4
    elif calorie_delta > 0:
        warnings.append(f"You are over calories by {calorie_delta:.0f} kcal today.")
        severity_score += 2
    else:
        info.append(f"You are within goal with {abs(calorie_delta):.0f} kcal remaining.")
        wins.append("Calorie intake is currently under control.")

    if protein_gap > max(PROTEIN_GAP_ALERT_GRAMS * 2.5, protein_goal * 0.25):
        critical.append(f"Protein intake is far below target. You still need about {protein_gap:.0f} g.")
        severity_score += 3
    elif protein_gap > PROTEIN_GAP_ALERT_GRAMS:
        info.append(f"Increase protein intake by {protein_gap:.0f} g to match your goal.")
        severity_score += 1
    else:
        wins.append("Protein intake is close to target.")

    if carb_delta > max(CARB_ALERT_GRAMS * 2.5, carbs_goal * 0.18):
        warnings.append(f"Carbohydrates are well above target by about {carb_delta:.0f} g today.")
        severity_score += 2
    elif carb_delta > CARB_ALERT_GRAMS:
        warnings.append(f"You are over carbs by {carb_delta:.0f} g today.")
        severity_score += 1
    elif today_carbs > 0 and carb_delta <= 0:
        wins.append("Carbohydrate intake is within target range.")

    # --------------------------------------------------
    # Lifestyle modifiers
    # --------------------------------------------------
    if latest_sleep_hours is not None:
        if latest_sleep_hours < max(0.0, LOW_SLEEP_HOURS - 0.75):
            critical.append("Very low sleep detected. Recovery and appetite regulation may be affected today.")
            severity_score += 2
        elif latest_sleep_hours < LOW_SLEEP_HOURS:
            warnings.append("Low sleep detected. Poor sleep may increase cravings and reduce recovery.")
            severity_score += 1
        else:
            wins.append("Sleep duration is in a supportive range.")

    if latest_steps is not None:
        if latest_steps < max(3000, LOW_STEPS_THRESHOLD - 2500):
            warnings.append("Activity is very low today. Energy balance may be harder to manage.")
            severity_score += 1
        elif latest_steps < LOW_STEPS_THRESHOLD:
            info.append("Activity is low today. A short walk can improve energy balance.")
        else:
            wins.append("Movement level looks supportive today.")

    if latest_water_ml is not None:
        if water_goal_ml is not None:
            water_gap = water_goal_ml - latest_water_ml
            if water_gap > water_goal_ml * 0.4:
                info.append(f"Hydration is low. Try drinking about {water_gap:.0f} ml more.")
            elif water_gap <= 0:
                wins.append("Hydration goal has been reached.")
        elif latest_water_ml < LOW_WATER_THRESHOLD_ML:
            info.append("Hydration appears low today. Drinking more water may support appetite control and recovery.")

    # --------------------------------------------------
    # Recent trend detection
    # --------------------------------------------------
    if recent_calories:
        avg_recent_calories = _mean(recent_calories)
        days_over_goal = _count_if(recent_calories, lambda x: _safe_float(x) > calorie_goal)
        days_far_over = _count_if(recent_calories, lambda x: _safe_float(x) > calorie_goal * 1.10)
        days_under_goal = _count_if(recent_calories, lambda x: _safe_float(x) < calorie_goal * 0.90)

        if len(recent_calories) >= 3 and days_far_over >= 2:
            patterns.append(
                f"Pattern detected: calorie intake has been high on {days_far_over} of the last {len(recent_calories)} tracked days."
            )
            severity_score += 2
        elif len(recent_calories) >= 3 and days_over_goal >= 2:
            patterns.append(
                f"Pattern detected: you were above calorie goal on {days_over_goal} of the last {len(recent_calories)} days."
            )
            severity_score += 1

        if goal == "Fat Loss":
            if avg_recent_calories < calorie_goal * 0.92:
                wins.append("Recent calorie trend supports fat-loss progress.")
            elif avg_recent_calories > calorie_goal * 1.08:
                warnings.append("Your recent calorie trend is above your fat-loss target.")
                severity_score += 2

        elif goal == "Muscle Gain":
            if avg_recent_calories < calorie_goal * 0.90:
                warnings.append("Recent calorie intake may be too low to support muscle gain consistently.")
                severity_score += 1
            elif avg_recent_calories >= calorie_goal * 0.97:
                wins.append("Recent calorie trend is supportive of muscle gain.")

        elif goal == "Maintain":
            if len(recent_calories) >= 3 and days_under_goal >= 2 and days_over_goal >= 2:
                patterns.append("Your recent calorie intake looks inconsistent for a maintenance goal.")
                severity_score += 1
            elif abs(avg_recent_calories - calorie_goal) <= calorie_goal * 0.08:
                wins.append("Recent calorie trend is stable for maintenance.")

    if recent_protein:
        avg_recent_protein = _mean(recent_protein)
        if len(recent_protein) >= 3 and avg_recent_protein < protein_goal * 0.85:
            patterns.append("Protein intake has been below target across recent days.")
            severity_score += 1
        elif len(recent_protein) >= 3 and avg_recent_protein >= protein_goal * 0.95:
            wins.append("Recent protein consistency is good.")

    if recent_carbs:
        avg_recent_carbs = _mean(recent_carbs)
        if len(recent_carbs) >= 3 and avg_recent_carbs > carbs_goal * 1.10:
            patterns.append("Carbohydrate intake has been consistently above target recently.")
            severity_score += 1

    if recent_sleep_hours:
        low_sleep_days = _count_if(recent_sleep_hours, lambda x: _safe_float(x) < LOW_SLEEP_HOURS)
        if len(recent_sleep_hours) >= 3 and low_sleep_days >= 2:
            patterns.append("Sleep has been below target on multiple recent days.")
            severity_score += 1

    if recent_steps:
        low_step_days = _count_if(recent_steps, lambda x: _safe_int(x) < LOW_STEPS_THRESHOLD)
        if len(recent_steps) >= 3 and low_step_days >= 2:
            patterns.append("Daily movement has been lower than target on multiple recent days.")

    if recent_weights and len(recent_weights) >= 2:
        first_weight = _safe_float(recent_weights[0])
        last_weight = _safe_float(recent_weights[-1])
        weight_change = last_weight - first_weight

        if goal == "Fat Loss" and weight_change > 0.4:
            patterns.append("Weight trend is rising despite a fat-loss goal.")
            severity_score += 2
        elif goal == "Muscle Gain" and weight_change < -0.4:
            patterns.append("Weight trend is dropping despite a muscle-gain goal.")
            severity_score += 2
        elif goal == "Maintain" and abs(weight_change) <= 0.3:
            wins.append("Weight trend is stable, which matches a maintenance goal.")

    # --------------------------------------------------
    # Goal-specific suggestions
    # --------------------------------------------------
    if goal == "Fat Loss":
        if protein_gap > PROTEIN_GAP_ALERT_GRAMS:
            suggestions.append("Choose a lean protein meal next, such as chicken, eggs, Greek yogurt, tofu, or fish.")
        if carb_delta > CARB_ALERT_GRAMS:
            suggestions.append("For your next meal, reduce refined carbs and add vegetables or salad for volume.")
        suggestions.append("Keep your final meal lighter and protein-focused if you are already near your calorie limit.")

    elif goal == "Muscle Gain":
        if protein_gap > PROTEIN_GAP_ALERT_GRAMS:
            suggestions.append("Add a high-protein snack like milk, yogurt, eggs, chicken, paneer, or a shake.")
        if calorie_delta < -200:
            suggestions.append("You may need an extra balanced meal or snack to support muscle gain.")
        suggestions.append("Pair carbs with protein around training for better recovery and performance.")

    else:
        if calorie_delta > 0:
            suggestions.append("Keep the next meal lighter and prioritize protein plus fiber.")
        else:
            suggestions.append("Aim for balanced meals with protein, fiber, and moderate portions to maintain consistency.")
        suggestions.append("Try keeping calorie intake more even across the day rather than late-night catch-up eating.")

    # --------------------------------------------------
    # Main issue and best action
    # --------------------------------------------------
    if critical:
        main_issue = critical[0]
    elif warnings:
        main_issue = warnings[0]
    elif info:
        main_issue = info[0]
    else:
        main_issue = "No major issue detected today."

    if protein_gap > PROTEIN_GAP_ALERT_GRAMS:
        best_action = "Prioritize a protein-rich next meal."
    elif calorie_delta > 0:
        best_action = "Keep the next meal lighter and lower in calorie density."
    elif latest_sleep_hours is not None and latest_sleep_hours < LOW_SLEEP_HOURS:
        best_action = "Focus on recovery today and avoid impulsive snacking."
    elif latest_steps is not None and latest_steps < LOW_STEPS_THRESHOLD:
        best_action = "Add a short walk or light movement session today."
    elif latest_water_ml is not None and water_goal_ml is not None and latest_water_ml < water_goal_ml * 0.6:
        best_action = "Increase hydration steadily through the rest of the day."
    else:
        best_action = "Stay consistent with your current routine."

    priority = _priority_label(severity_score)

    if severity_score >= 8:
        summary = "Today needs correction: intake and recovery signals suggest a higher risk of missing your goal."
    elif severity_score >= 4:
        summary = "Today is manageable, but a few adjustments would improve alignment with your goal."
    else:
        summary = "Today is broadly on track. Small consistent actions will keep progress moving."

    # --------------------------------------------------
    # Optional model confidence explanation
    # --------------------------------------------------
    confidence_band = None
    confidence_message = None
    if latest_prediction_confidence is not None:
        confidence_band = _confidence_band(latest_prediction_confidence)
        if confidence_band == "high":
            confidence_message = "The latest food prediction was made with high confidence."
        elif confidence_band == "medium":
            confidence_message = "The latest food prediction had moderate confidence and may benefit from user confirmation."
        else:
            confidence_message = "The latest food prediction had low confidence and should be interpreted carefully."

    # --------------------------------------------------
    # Projection helper
    # --------------------------------------------------
    projections = {}
    if recent_calories and current_weight_kg is not None:
        avg_recent_calories = _mean(recent_calories)
        avg_daily_delta = avg_recent_calories - calorie_goal
        projections["weight_projection"] = _weight_projection_from_calorie_delta(
            current_weight_kg=current_weight_kg,
            avg_daily_calorie_delta=avg_daily_delta,
            projection_days=WEIGHT_PROJECTION_DAYS,
        )

    # --------------------------------------------------
    # Final cleanup
    # --------------------------------------------------
    critical = _unique(critical)
    warnings = _unique(warnings)
    info = _unique(info)
    suggestions = _unique(suggestions)
    patterns = _unique(patterns)
    wins = _unique(wins)

    if confidence_message:
        info = _unique(info + [confidence_message])

    return {
        "critical": critical,
        "warnings": warnings,
        "info": info,
        "suggestions": suggestions,
        "patterns": patterns,
        "wins": wins,
        "summary": summary,
        "main_issue": main_issue,
        "best_action": best_action,
        "priority": priority,
        "confidence_band": confidence_band,
        "projections": projections,
        "scores": {
            "severity": severity_score,
            "calorie_delta": round(calorie_delta, 2),
            "calorie_delta_pct": round(calorie_delta_pct, 2),
            "protein_gap": round(protein_gap, 2),
            "protein_gap_pct": round(protein_gap_pct, 2),
            "carb_delta": round(carb_delta, 2),
            "carb_delta_pct": round(carb_delta_pct, 2),
        },
    }