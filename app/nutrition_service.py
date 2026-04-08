"""
Resolve nutrition per 100g from local CSV first, then Open Food Facts.
Designed for safer matching, cleaner fallback behavior, and better backend reliability.
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
    "User-Agent": "CalorifyAI/1.0 (nutrition lookup; educational use)",
    "Accept": "application/json",
}

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
    pass


_nutrition_rows: list[dict[str, Any]] | None = None

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


def _normalize_label(value: str) -> str:
    value = str(value or "").lower().strip()
    value = re.sub(r"[^\w\s-]", "", value)
    value = value.replace(" ", "_").replace("-", "_")
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def _token_set(value: str) -> set[str]:
    normalized = _normalize_label(value).replace("_", " ")
    return {token for token in normalized.split() if token}


def _singularize_tokens(tokens: set[str]) -> set[str]:
    out = set()
    for token in tokens:
        if token.endswith("s") and len(token) > 3:
            out.add(token[:-1])
        else:
            out.add(token)
    return out


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _looks_like_reasonable_match(query: str, candidate: str) -> bool:
    q = _token_set(query)
    c = _token_set(candidate)

    if not q or not c:
        return False

    if q.intersection(c):
        return True

    q_relaxed = _singularize_tokens(q)
    c_relaxed = _singularize_tokens(c)
    return bool(q_relaxed.intersection(c_relaxed))


def _load_local_table() -> list[dict[str, Any]]:
    global _nutrition_rows

    if _nutrition_rows is not None:
        return _nutrition_rows

    rows: list[dict[str, Any]] = []

    if not NUTRITION_CSV.is_file():
        _nutrition_rows = rows
        return rows

    with open(NUTRITION_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                food_name = str(row["food"]).strip()
                if not food_name:
                    continue

                rows.append(
                    {
                        "food": food_name,
                        "calories_per_100g": _safe_float(row.get("calories_per_100g")),
                        "protein": _safe_float(row.get("protein")),
                        "carbs": _safe_float(row.get("carbs")),
                        "fat": _safe_float(row.get("fat")),
                    }
                )
            except KeyError:
                continue

    _nutrition_rows = rows
    return rows


def lookup_local(food_name: str) -> tuple[dict[str, Any], str, float] | None:
    query = _normalize_label(food_name)
    if not query:
        return None

    rows = _load_local_table()

    for row in rows:
        key = row["food"]
        normalized_key = _normalize_label(key.replace("_", " "))
        if query == normalized_key or query == key:
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
    best_score = -1

    for row in rows:
        key = row["food"]
        normalized_key = _normalize_label(key.replace("_", " "))
        query_tokens = _token_set(query)
        key_tokens = _token_set(normalized_key)

        overlap = len(query_tokens.intersection(key_tokens))
        if overlap <= 0 and not (query in normalized_key or normalized_key in query):
            continue

        score = overlap * 10 + min(len(normalized_key), len(query))
        if score > best_score:
            best_score = score
            best = (
                {
                    "calories_per_100g": row["calories_per_100g"],
                    "protein": row["protein"],
                    "carbs": row["carbs"],
                    "fat": row["fat"],
                },
                key,
                0.82,
            )

    return best


def _json_body(response: httpx.Response) -> dict[str, Any] | None:
    content_type = (response.headers.get("content-type") or "").lower()
    if "application/json" not in content_type and "json" not in content_type:
        return None

    try:
        payload = response.json()
        return payload if isinstance(payload, dict) else None
    except ValueError:
        return None


def _fetch_off_search(
    client: httpx.Client,
    url: str,
    params: dict[str, str],
    *,
    attempts: int = 4,
) -> dict[str, Any] | None:
    base_delay = 0.85

    for attempt in range(attempts):
        try:
            response = client.get(url, params=params)
        except httpx.RequestError:
            time.sleep(base_delay * (attempt + 1))
            continue

        if response.status_code == 200:
            data = _json_body(response)
            if data is not None:
                return data
            time.sleep(base_delay * (attempt + 1))
            continue

        if response.status_code in (429, 500, 502, 503, 504):
            time.sleep(base_delay * (attempt + 1))
            continue

        break

    return None


def _nutrients_from_off_product(product: dict[str, Any]) -> dict[str, Any] | None:
    nutriments = product.get("nutriments") or {}

    kcal = nutriments.get("energy-kcal_100g")
    if kcal is None and nutriments.get("energy-kcal") is not None:
        kcal = nutriments.get("energy-kcal")
    if kcal is None and nutriments.get("energy_100g") is not None:
        try:
            kcal = float(nutriments["energy_100g"]) / 4.184
        except (TypeError, ValueError):
            kcal = None

    if kcal is None:
        return None

    display_name = (
        product.get("product_name")
        or product.get("product_name_en")
        or ""
    ).strip()

    if not display_name:
        display_name = "unknown"

    return {
        "food_name": display_name,
        "calories_per_100g": _safe_float(kcal),
        "protein": _safe_float(nutriments.get("proteins_100g")),
        "carbs": _safe_float(nutriments.get("carbohydrates_100g")),
        "fat": _safe_float(nutriments.get("fat_100g")),
    }


def lookup_open_food_facts(food_name: str) -> tuple[dict[str, Any], float] | None:
    query = str(food_name or "").strip()
    if not query:
        return None

    saw_valid_json = False
    timeout = httpx.Timeout(25.0, connect=10.0)

    with httpx.Client(timeout=timeout, headers=OFF_HEADERS) as client:
        for url, param_template in OFF_ENDPOINTS:
            params = {**param_template, "search_terms": query}
            data = _fetch_off_search(client, url, params)

            if not data:
                continue

            saw_valid_json = True

            for product in data.get("products") or []:
                if not isinstance(product, dict):
                    continue

                parsed = _nutrients_from_off_product(product)
                if not parsed:
                    continue

                if _looks_like_reasonable_match(query, parsed.get("food_name", "")):
                    return parsed, 0.72

    if not saw_valid_json:
        raise OpenFoodFactsUnavailableError(
            "Open Food Facts is temporarily unavailable. Try again later or use a local food item."
        )

    return None


def resolve_food(food_name: str, grams: float) -> ResolvedNutrition:
    grams = _safe_float(grams)
    if grams <= 0:
        raise ValueError("grams must be positive")

    name = str(food_name or "").strip()
    if not name:
        raise ValueError("food_name cannot be empty")

    normalized_name = _normalize_label(name)
    if normalized_name in _TYPO_ALIASES:
        name = _TYPO_ALIASES[normalized_name]

    local_result = lookup_local(name)
    if local_result:
        per_100g, matched_key, confidence = local_result
        factor = grams / 100.0

        cals_100 = _safe_float(per_100g["calories_per_100g"])
        protein_100 = _safe_float(per_100g["protein"])
        carbs_100 = _safe_float(per_100g["carbs"])
        fat_100 = _safe_float(per_100g["fat"])

        return ResolvedNutrition(
            display_name=matched_key.replace("_", " ").title(),
            matched_key=matched_key,
            grams=grams,
            calories=cals_100 * factor,
            protein=protein_100 * factor,
            carbs=carbs_100 * factor,
            fat=fat_100 * factor,
            calories_per_100g=cals_100,
            protein_per_100g=protein_100,
            carbs_per_100g=carbs_100,
            fat_per_100g=fat_100,
            source="local_csv",
            confidence=confidence,
        )

    off_result = lookup_open_food_facts(name)
    if off_result:
        parsed, confidence = off_result
        factor = grams / 100.0

        cals_100 = _safe_float(parsed["calories_per_100g"])
        protein_100 = _safe_float(parsed["protein"])
        carbs_100 = _safe_float(parsed["carbs"])
        fat_100 = _safe_float(parsed["fat"])

        label = parsed["food_name"]

        return ResolvedNutrition(
            display_name=label,
            matched_key=label,
            grams=grams,
            calories=cals_100 * factor,
            protein=protein_100 * factor,
            carbs=carbs_100 * factor,
            fat=fat_100 * factor,
            calories_per_100g=cals_100,
            protein_per_100g=protein_100,
            carbs_per_100g=carbs_100,
            fat_per_100g=fat_100,
            source="open_food_facts",
            confidence=confidence,
        )

    raise LookupError(f"No nutrition data found for {food_name!r}")