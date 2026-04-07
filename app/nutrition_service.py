"""
Resolve nutrition per 100g from local Food-101-style CSV, then Open Food Facts.
"""

from __future__ import annotations

import csv
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NUTRITION_CSV = PROJECT_ROOT / "nutrition_data.csv"

OFF_HEADERS = {
    "User-Agent": "Calorify/1.0 (meal lookup; respectful use of Open Food Facts)",
    "Accept": "application/json",
}

# v2 search is usually more reliable than legacy CGI; CGI kept as fallback.
OFF_ENDPOINTS: list[tuple[str, dict[str, str]]] = [
    (
        "https://world.openfoodfacts.org/api/v2/search",
        {"search_terms": "", "page_size": "10"},
    ),
    (
        "https://world.openfoodfacts.org/cgi/search.pl",
        {
            "search_terms": "",
            "search_simple": "1",
            "action": "process",
            "json": "1",
            "page_size": "10",
        },
    ),
    (
        "https://en.openfoodfacts.org/cgi/search.pl",
        {
            "search_terms": "",
            "search_simple": "1",
            "action": "process",
            "json": "1",
            "page_size": "10",
        },
    ),
]


class OpenFoodFactsUnavailableError(Exception):
    """Raised when OFF returns repeated errors (e.g. 503) or non-JSON responses."""

    pass

_nutrition_rows: list[dict[str, Any]] | None = None

# Normalized keys → phrase that matches local CSV / fuzzy search
_TYPO_ALIASES: dict[str, str] = {
    "biriyani": "chicken curry",
    "biryani": "chicken curry",
    "egg": "omelette",
    "eggs": "omelette",
    "boiled_egg": "omelette",
    "boiled_eggs": "omelette",
    "fried_egg": "omelette",
    "fried_eggs": "omelette",
    "scrambled_egg": "omelette",
    "scrambled_eggs": "omelette",
    "toast": "garlic_bread",
}


def _normalize_label(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = s.replace(" ", "_").replace("-", "_")
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _token_set(s: str) -> set[str]:
    n = _normalize_label(s).replace("_", " ")
    return {t for t in n.split() if t}


def _looks_like_reasonable_match(query: str, candidate: str) -> bool:
    """
    Guardrail against unrelated Open Food Facts hits.
    Accept only if token overlap is meaningful.
    """
    q = _token_set(query)
    c = _token_set(candidate)
    if not q or not c:
        return False
    overlap = q.intersection(c)
    # Exact token overlap is best.
    if overlap:
        return True
    # Lightweight singular/plural relaxation (egg/eggs).
    q_relaxed = {t[:-1] if t.endswith("s") and len(t) > 3 else t for t in q}
    c_relaxed = {t[:-1] if t.endswith("s") and len(t) > 3 else t for t in c}
    return bool(q_relaxed.intersection(c_relaxed))


def _load_local_table() -> list[dict[str, Any]]:
    global _nutrition_rows
    if _nutrition_rows is not None:
        return _nutrition_rows
    rows: list[dict[str, Any]] = []
    if NUTRITION_CSV.is_file():
        with open(NUTRITION_CSV, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    rows.append(
                        {
                            "food": str(r["food"]).strip(),
                            "calories_per_100g": float(r["calories_per_100g"]),
                            "protein": float(r["protein"]),
                            "carbs": float(r["carbs"]),
                            "fat": float(r["fat"]),
                        }
                    )
                except (KeyError, ValueError, TypeError):
                    continue
    _nutrition_rows = rows
    return rows


def lookup_local(food_name: str) -> tuple[dict[str, Any], str, float] | None:
    """
    Returns (per_100g dict, matched_key, confidence) or None.
    """
    q = _normalize_label(food_name)
    if not q:
        return None
    rows = _load_local_table()

    for row in rows:
        key = row["food"]
        nk = _normalize_label(key.replace("_", " "))
        if q == nk or q == key:
            return (
                {
                    "calories_per_100g": row["calories_per_100g"],
                    "protein": row["protein"],
                    "carbs": row["carbs"],
                    "fat": row["fat"],
                },
                key,
                1.0,
            )

    best: tuple[dict[str, Any], str, float] | None = None
    for row in rows:
        key = row["food"]
        nk = _normalize_label(key.replace("_", " "))
        if q in nk or nk in q:
            cand = (
                {
                    "calories_per_100g": row["calories_per_100g"],
                    "protein": row["protein"],
                    "carbs": row["carbs"],
                    "fat": row["fat"],
                },
                key,
                0.82,
            )
            if best is None or len(nk) > len(best[1]):
                best = cand
    return best


def _json_body(r: httpx.Response) -> dict[str, Any] | None:
    ct = (r.headers.get("content-type") or "").lower()
    if "application/json" not in ct and "json" not in ct:
        return None
    try:
        out = r.json()
        return out if isinstance(out, dict) else None
    except ValueError:
        return None


def _fetch_off_search(
    client: httpx.Client, url: str, params: dict[str, str], *, attempts: int = 4
) -> dict[str, Any] | None:
    """GET with retries; returns parsed JSON dict or None if all tries fail."""
    base_delay = 0.85
    for attempt in range(attempts):
        try:
            resp = client.get(url, params=params)
        except httpx.RequestError:
            time.sleep(base_delay * (attempt + 1))
            continue

        if resp.status_code == 200:
            data = _json_body(resp)
            if data is not None:
                return data
            time.sleep(base_delay * (attempt + 1))
            continue

        if resp.status_code in (429, 500, 502, 503, 504):
            time.sleep(base_delay * (attempt + 1))
            continue

        # Other 4xx: do not hammer
        break
    return None


def _nutrients_from_off_product(p: dict[str, Any]) -> dict[str, Any] | None:
    n = p.get("nutriments") or {}
    kcal = n.get("energy-kcal_100g")
    if kcal is None and n.get("energy-kcal") is not None:
        kcal = n.get("energy-kcal")
    if kcal is None and n.get("energy_100g") is not None:
        try:
            kcal = float(n["energy_100g"]) / 4.184
        except (TypeError, ValueError):
            kcal = None
    if kcal is None:
        return None
    try:
        return {
            "food_name": (p.get("product_name") or p.get("product_name_en") or "").strip()
            or "unknown",
            "calories_per_100g": float(kcal),
            "protein": float(n.get("proteins_100g") or 0),
            "carbs": float(n.get("carbohydrates_100g") or 0),
            "fat": float(n.get("fat_100g") or 0),
        }
    except (TypeError, ValueError):
        return None


def lookup_open_food_facts(food_name: str) -> tuple[dict[str, Any], float] | None:
    """
    First suitable hit from Open Food Facts (per-100g where possible).
    Uses v2 search first, then legacy CGI mirrors, with retries on overload.
    """
    q = food_name.strip()
    if not q:
        return None

    saw_valid_json = False
    timeout = httpx.Timeout(25.0, connect=10.0)

    with httpx.Client(timeout=timeout, headers=OFF_HEADERS) as client:
        for url, param_template in OFF_ENDPOINTS:
            params = {**param_template, "search_terms": q}
            data = _fetch_off_search(client, url, params)
            if not data:
                continue
            saw_valid_json = True
            for p in data.get("products") or []:
                if not isinstance(p, dict):
                    continue
                parsed = _nutrients_from_off_product(p)
                if parsed and _looks_like_reasonable_match(q, parsed.get("food_name", "")):
                    return parsed, 0.72

    if not saw_valid_json:
        raise OpenFoodFactsUnavailableError(
            "Open Food Facts is not responding right now (often temporary when their servers are busy). "
            "Wait a minute and try again, or use a dish from the built-in list (e.g. “chicken curry”, “bibimbap”)."
        )
    return None


@dataclass
class ResolvedNutrition:
    display_name: str
    matched_key: str
    grams: float
    calories: float
    protein: float
    carbs: float
    fat: float
    calories_per_100g: float
    protein_per_100g: float
    carbs_per_100g: float
    fat_per_100g: float
    source: str
    confidence: float


def resolve_food(food_name: str, grams: float) -> ResolvedNutrition:
    if grams <= 0:
        raise ValueError("grams must be positive")

    name = food_name.strip()
    nk = _normalize_label(name)
    if nk in _TYPO_ALIASES:
        name = _TYPO_ALIASES[nk]

    local = lookup_local(name)
    if local:
        per, key, conf = local
        cph = per["calories_per_100g"]
        pph = per["protein"]
        carbph = per["carbs"]
        fph = per["fat"]
        factor = grams / 100.0
        return ResolvedNutrition(
            display_name=key.replace("_", " ").title(),
            matched_key=key,
            grams=grams,
            calories=cph * factor,
            protein=pph * factor,
            carbs=carbph * factor,
            fat=fph * factor,
            calories_per_100g=cph,
            protein_per_100g=pph,
            carbs_per_100g=carbph,
            fat_per_100g=fph,
            source="local_csv",
            confidence=conf,
        )

    off = lookup_open_food_facts(name)
    if off:
        parsed, conf = off
        cph = parsed["calories_per_100g"]
        pph = parsed["protein"]
        carbph = parsed["carbs"]
        fph = parsed["fat"]
        factor = grams / 100.0
        label = parsed["food_name"]
        return ResolvedNutrition(
            display_name=label,
            matched_key=label,
            grams=grams,
            calories=cph * factor,
            protein=pph * factor,
            carbs=carbph * factor,
            fat=fph * factor,
            calories_per_100g=cph,
            protein_per_100g=pph,
            carbs_per_100g=carbph,
            fat_per_100g=fph,
            source="open_food_facts",
            confidence=conf,
        )

    raise LookupError(f"No nutrition data found for {food_name!r}")
