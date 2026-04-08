import os
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from typing import Iterator


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "meal_history.db")


def _local_calendar_date_str(d=None) -> str:
    """Return YYYY-MM-DD in local time."""
    if d is None:
        return datetime.now().strftime("%Y-%m-%d")
    if isinstance(d, date):
        return d.isoformat()
    return str(d)[:10]


def _local_week_start_str(days_back: int = 6) -> str:
    return (date.today() - timedelta(days=days_back)).isoformat()


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def create_table() -> None:
    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS meals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                food_name TEXT NOT NULL,
                grams REAL NOT NULL CHECK (grams > 0),
                calories REAL NOT NULL CHECK (calories >= 0),
                protein REAL NOT NULL CHECK (protein >= 0),
                carbs REAL NOT NULL CHECK (carbs >= 0),
                fat REAL NOT NULL CHECK (fat >= 0),
                confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 100),
                created_at TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS favorite_meals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                food_name TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_meals_created_at
            ON meals(created_at)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_meals_food_name
            ON meals(food_name)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_favorite_meals_food_name
            ON favorite_meals(food_name)
        """)


def insert_meal(
    food_name,
    grams,
    calories,
    protein,
    carbs,
    fat,
    confidence,
    meal_date=None,
) -> int:
    """
    Insert a meal row and return inserted row id.

    meal_date:
        Optional date or YYYY-MM-DD string for the calendar day to assign.
        Time portion is always the current local time.
    """
    create_table()

    food_name = str(food_name).strip()
    if not food_name:
        raise ValueError("food_name cannot be empty")

    grams = float(grams)
    calories = float(calories)
    protein = float(protein)
    carbs = float(carbs)
    fat = float(fat)
    confidence = float(confidence)

    if grams <= 0:
        raise ValueError("grams must be greater than 0")
    if calories < 0 or protein < 0 or carbs < 0 or fat < 0:
        raise ValueError("nutrition values cannot be negative")
    if confidence < 0 or confidence > 100:
        raise ValueError("confidence must be between 0 and 100")

    if meal_date is None:
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    else:
        day = _local_calendar_date_str(meal_date)
        current_time = datetime.now().strftime("%H:%M:%S")
        created_at = f"{day} {current_time}"

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO meals (
                food_name, grams, calories, protein, carbs, fat, confidence, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            food_name,
            grams,
            calories,
            protein,
            carbs,
            fat,
            confidence,
            created_at,
        ))
        return int(cursor.lastrowid)


def get_all_meals():
    create_table()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, food_name, grams, calories, protein, carbs, fat, confidence, created_at
            FROM meals
            ORDER BY created_at DESC, id DESC
        """)
        rows = cursor.fetchall()
        return [tuple(row) for row in rows]


def get_meals_by_date(target_date):
    create_table()
    target = _local_calendar_date_str(target_date)

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, food_name, grams, calories, protein, carbs, fat, confidence, created_at
            FROM meals
            WHERE substr(created_at, 1, 10) = ?
            ORDER BY created_at DESC, id DESC
        """, (target,))
        rows = cursor.fetchall()
        return [tuple(row) for row in rows]


def get_today_totals():
    create_table()
    today = _local_calendar_date_str()

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                COALESCE(SUM(calories), 0) AS total_calories,
                COALESCE(SUM(protein), 0) AS total_protein,
                COALESCE(SUM(carbs), 0) AS total_carbs,
                COALESCE(SUM(fat), 0) AS total_fat
            FROM meals
            WHERE substr(created_at, 1, 10) = ?
        """, (today,))
        row = cursor.fetchone()
        return (
            float(row["total_calories"]),
            float(row["total_protein"]),
            float(row["total_carbs"]),
            float(row["total_fat"]),
        )


def get_daily_calorie_history():
    create_table()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                substr(created_at, 1, 10) AS meal_date,
                COALESCE(SUM(calories), 0) AS total_calories
            FROM meals
            GROUP BY substr(created_at, 1, 10)
            ORDER BY meal_date ASC
        """)
        rows = cursor.fetchall()
        return [(row["meal_date"], float(row["total_calories"])) for row in rows]


def get_daily_macro_history():
    create_table()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                substr(created_at, 1, 10) AS meal_date,
                COALESCE(SUM(protein), 0) AS total_protein,
                COALESCE(SUM(carbs), 0) AS total_carbs,
                COALESCE(SUM(fat), 0) AS total_fat
            FROM meals
            GROUP BY substr(created_at, 1, 10)
            ORDER BY meal_date ASC
        """)
        rows = cursor.fetchall()
        return [
            (
                row["meal_date"],
                float(row["total_protein"]),
                float(row["total_carbs"]),
                float(row["total_fat"]),
            )
            for row in rows
        ]


def get_weekly_summary(days_back: int = 6):
    create_table()
    week_start = _local_week_start_str(days_back)

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                COALESCE(SUM(calories), 0) AS total_calories,
                COALESCE(SUM(protein), 0) AS total_protein,
                COALESCE(SUM(carbs), 0) AS total_carbs,
                COALESCE(SUM(fat), 0) AS total_fat
            FROM meals
            WHERE substr(created_at, 1, 10) >= ?
        """, (week_start,))
        row = cursor.fetchone()
        return (
            float(row["total_calories"]),
            float(row["total_protein"]),
            float(row["total_carbs"]),
            float(row["total_fat"]),
        )


def meal_exists(meal_id) -> bool:
    create_table()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) AS count FROM meals WHERE id = ?", (int(meal_id),))
        row = cursor.fetchone()
        return int(row["count"]) > 0


def delete_meal(meal_id) -> None:
    create_table()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM meals WHERE id = ?", (int(meal_id),))


def clear_all_meals() -> None:
    create_table()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM meals")


def get_recent_foods(limit: int = 8):
    create_table()
    limit = max(1, int(limit))

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT food_name, MAX(created_at) AS last_seen
            FROM meals
            GROUP BY food_name
            ORDER BY last_seen DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        return [str(row["food_name"]) for row in rows]


def get_favorite_foods():
    create_table()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT food_name
            FROM favorite_meals
            ORDER BY created_at DESC, id DESC
        """)
        rows = cursor.fetchall()
        return [str(row["food_name"]) for row in rows]


def add_favorite_food(food_name) -> None:
    create_table()
    food_name = str(food_name).strip()
    if not food_name:
        raise ValueError("food_name cannot be empty")

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO favorite_meals (food_name, created_at)
            VALUES (?, ?)
        """, (
            food_name,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ))


def remove_favorite_food(food_name) -> None:
    create_table()
    food_name = str(food_name).strip()
    if not food_name:
        return

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM favorite_meals WHERE food_name = ?", (food_name,))


def get_meal_count() -> int:
    create_table()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) AS count FROM meals")
        row = cursor.fetchone()
        return int(row["count"])


def get_average_confidence() -> float:
    create_table()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COALESCE(AVG(confidence), 0) AS avg_confidence
            FROM meals
        """)
        row = cursor.fetchone()
        return float(row["avg_confidence"])