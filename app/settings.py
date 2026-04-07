import os


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


CONFIDENCE_THRESHOLD_DEFAULT = _get_int("CALORIFY_CONFIDENCE_THRESHOLD", 60)
CARB_ALERT_GRAMS = _get_float("CALORIFY_CARB_ALERT_GRAMS", 10.0)
PROTEIN_GAP_ALERT_GRAMS = _get_float("CALORIFY_PROTEIN_GAP_ALERT_GRAMS", 5.0)
LOW_SLEEP_HOURS = _get_float("CALORIFY_LOW_SLEEP_HOURS", 6.5)
LOW_STEPS_THRESHOLD = _get_int("CALORIFY_LOW_STEPS_THRESHOLD", 6000)
