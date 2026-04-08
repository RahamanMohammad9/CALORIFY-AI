import os
from typing import Any


def _get_env(name: str, default: Any) -> str:
    raw = os.getenv(name)
    if raw is None:
        return str(default)
    return str(raw).strip()


def _get_float(name: str, default: float, *, min_value: float | None = None, max_value: float | None = None) -> float:
    raw = _get_env(name, default)
    try:
        value = float(raw)
    except ValueError:
        value = float(default)

    if min_value is not None:
        value = max(value, float(min_value))
    if max_value is not None:
        value = min(value, float(max_value))

    return float(value)


def _get_int(name: str, default: int, *, min_value: int | None = None, max_value: int | None = None) -> int:
    raw = _get_env(name, default)
    try:
        value = int(raw)
    except ValueError:
        value = int(default)

    if min_value is not None:
        value = max(value, int(min_value))
    if max_value is not None:
        value = min(value, int(max_value))

    return int(value)


def _get_bool(name: str, default: bool) -> bool:
    raw = _get_env(name, default).lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


# --------------------------------------------------
# Inference / prediction settings
# --------------------------------------------------
CONFIDENCE_THRESHOLD_DEFAULT = _get_int(
    "CALORIFY_CONFIDENCE_THRESHOLD",
    60,
    min_value=40,
    max_value=95,
)

LOW_CONFIDENCE_THRESHOLD = _get_int(
    "CALORIFY_LOW_CONFIDENCE_THRESHOLD",
    65,
    min_value=0,
    max_value=100,
)

HIGH_CONFIDENCE_THRESHOLD = _get_int(
    "CALORIFY_HIGH_CONFIDENCE_THRESHOLD",
    85,
    min_value=0,
    max_value=100,
)

ENABLE_TTA_DEFAULT = _get_bool("CALORIFY_ENABLE_TTA", True)
DEFAULT_TOP_K_PREDICTIONS = _get_int(
    "CALORIFY_TOP_K_PREDICTIONS",
    3,
    min_value=1,
    max_value=10,
)

# --------------------------------------------------
# AI coaching thresholds
# --------------------------------------------------
CARB_ALERT_GRAMS = _get_float(
    "CALORIFY_CARB_ALERT_GRAMS",
    10.0,
    min_value=0.0,
    max_value=200.0,
)

PROTEIN_GAP_ALERT_GRAMS = _get_float(
    "CALORIFY_PROTEIN_GAP_ALERT_GRAMS",
    5.0,
    min_value=0.0,
    max_value=100.0,
)

LOW_SLEEP_HOURS = _get_float(
    "CALORIFY_LOW_SLEEP_HOURS",
    6.5,
    min_value=0.0,
    max_value=24.0,
)

LOW_STEPS_THRESHOLD = _get_int(
    "CALORIFY_LOW_STEPS_THRESHOLD",
    6000,
    min_value=0,
    max_value=50000,
)

LOW_WATER_THRESHOLD_ML = _get_int(
    "CALORIFY_LOW_WATER_THRESHOLD_ML",
    1500,
    min_value=0,
    max_value=10000,
)

# --------------------------------------------------
# Portion estimation defaults
# --------------------------------------------------
DEFAULT_PORTION_GRAMS = _get_float(
    "CALORIFY_DEFAULT_PORTION_GRAMS",
    120.0,
    min_value=20.0,
    max_value=1000.0,
)

MIN_PORTION_GRAMS = _get_float(
    "CALORIFY_MIN_PORTION_GRAMS",
    40.0,
    min_value=1.0,
    max_value=1000.0,
)

MAX_PORTION_GRAMS = _get_float(
    "CALORIFY_MAX_PORTION_GRAMS",
    300.0,
    min_value=1.0,
    max_value=2000.0,
)

# --------------------------------------------------
# Nutrition defaults / fallback
# --------------------------------------------------
FALLBACK_CALORIES_PER_100G = _get_float(
    "CALORIFY_FALLBACK_CALORIES_PER_100G",
    250.0,
    min_value=0.0,
    max_value=2000.0,
)

FALLBACK_PROTEIN_PER_100G = _get_float(
    "CALORIFY_FALLBACK_PROTEIN_PER_100G",
    5.0,
    min_value=0.0,
    max_value=200.0,
)

FALLBACK_CARBS_PER_100G = _get_float(
    "CALORIFY_FALLBACK_CARBS_PER_100G",
    30.0,
    min_value=0.0,
    max_value=300.0,
)

FALLBACK_FAT_PER_100G = _get_float(
    "CALORIFY_FALLBACK_FAT_PER_100G",
    10.0,
    min_value=0.0,
    max_value=200.0,
)

# --------------------------------------------------
# Tracking / analytics defaults
# --------------------------------------------------
DEFAULT_WATER_GOAL_ML = _get_int(
    "CALORIFY_DEFAULT_WATER_GOAL_ML",
    2500,
    min_value=500,
    max_value=10000,
)

DEFAULT_CALORIE_GOAL = _get_int(
    "CALORIFY_DEFAULT_CALORIE_GOAL",
    2000,
    min_value=1000,
    max_value=5000,
)

RECENT_DAYS_LOOKBACK = _get_int(
    "CALORIFY_RECENT_DAYS_LOOKBACK",
    7,
    min_value=3,
    max_value=60,
)

WEIGHT_PROJECTION_DAYS = _get_int(
    "CALORIFY_WEIGHT_PROJECTION_DAYS",
    30,
    min_value=1,
    max_value=365,
)

# --------------------------------------------------
# Optional debug / development flags
# --------------------------------------------------
DEBUG_MODE = _get_bool("CALORIFY_DEBUG_MODE", False)


def as_dict() -> dict:
    return {
        "CONFIDENCE_THRESHOLD_DEFAULT": CONFIDENCE_THRESHOLD_DEFAULT,
        "LOW_CONFIDENCE_THRESHOLD": LOW_CONFIDENCE_THRESHOLD,
        "HIGH_CONFIDENCE_THRESHOLD": HIGH_CONFIDENCE_THRESHOLD,
        "ENABLE_TTA_DEFAULT": ENABLE_TTA_DEFAULT,
        "DEFAULT_TOP_K_PREDICTIONS": DEFAULT_TOP_K_PREDICTIONS,
        "CARB_ALERT_GRAMS": CARB_ALERT_GRAMS,
        "PROTEIN_GAP_ALERT_GRAMS": PROTEIN_GAP_ALERT_GRAMS,
        "LOW_SLEEP_HOURS": LOW_SLEEP_HOURS,
        "LOW_STEPS_THRESHOLD": LOW_STEPS_THRESHOLD,
        "LOW_WATER_THRESHOLD_ML": LOW_WATER_THRESHOLD_ML,
        "DEFAULT_PORTION_GRAMS": DEFAULT_PORTION_GRAMS,
        "MIN_PORTION_GRAMS": MIN_PORTION_GRAMS,
        "MAX_PORTION_GRAMS": MAX_PORTION_GRAMS,
        "FALLBACK_CALORIES_PER_100G": FALLBACK_CALORIES_PER_100G,
        "FALLBACK_PROTEIN_PER_100G": FALLBACK_PROTEIN_PER_100G,
        "FALLBACK_CARBS_PER_100G": FALLBACK_CARBS_PER_100G,
        "FALLBACK_FAT_PER_100G": FALLBACK_FAT_PER_100G,
        "DEFAULT_WATER_GOAL_ML": DEFAULT_WATER_GOAL_ML,
        "DEFAULT_CALORIE_GOAL": DEFAULT_CALORIE_GOAL,
        "RECENT_DAYS_LOOKBACK": RECENT_DAYS_LOOKBACK,
        "WEIGHT_PROJECTION_DAYS": WEIGHT_PROJECTION_DAYS,
        "DEBUG_MODE": DEBUG_MODE,
    }