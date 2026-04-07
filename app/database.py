import os
import sqlite3
from datetime import date, datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "meal_history.db")


def _local_calendar_date_str(d=None) -> str:
    """YYYY-MM-DD in local time (matches how we store created_at)."""
    if d is None:
        return datetime.now().strftime("%Y-%m-%d")
    if isinstance(d, date):
        return d.isoformat()
    return str(d)[:10]


def _local_week_start_str(days_back: int = 6) -> str:
    return (date.today() - timedelta(days=days_back)).isoformat()


def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def create_table():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS meals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            food_name TEXT NOT NULL,
            grams REAL NOT NULL,
            calories REAL NOT NULL,
            protein REAL NOT NULL,
            carbs REAL NOT NULL,
            fat REAL NOT NULL,
            confidence REAL NOT NULL,
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
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_meals_created_at ON meals(created_at)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_meals_food_name ON meals(food_name)")

    conn.commit()
    conn.close()


def insert_meal(food_name, grams, calories, protein, carbs, fat, confidence, meal_date=None):
    """
    meal_date: optional calendar day for this meal (date or 'YYYY-MM-DD').
    Uses today's local date/time if omitted. Time portion is always current local time
    so ordering within a day stays sensible.
    """
    conn = get_connection()
    cursor = conn.cursor()

    if meal_date is None:
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    else:
        day = _local_calendar_date_str(meal_date)
        t = datetime.now().strftime("%H:%M:%S")
        created_at = f"{day} {t}"

    cursor.execute("""
        INSERT INTO meals (
            food_name, grams, calories, protein, carbs, fat, confidence, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        str(food_name),
        float(grams),
        float(calories),
        float(protein),
        float(carbs),
        float(fat),
        float(confidence),
        created_at,
    ))

    row_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return int(row_id)


def get_all_meals():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, food_name, grams, calories, protein, carbs, fat, confidence, created_at
        FROM meals
        ORDER BY created_at DESC
    """)

    rows = cursor.fetchall()
    conn.close()
    return rows


def get_today_totals():
    conn = get_connection()
    cursor = conn.cursor()

    # Compare calendar day using the stored string prefix so it matches Python local
    # timestamps (SQLite date() can treat naive strings inconsistently).
    today = _local_calendar_date_str()
    cursor.execute("""
        SELECT
            COALESCE(SUM(calories), 0),
            COALESCE(SUM(protein), 0),
            COALESCE(SUM(carbs), 0),
            COALESCE(SUM(fat), 0)
        FROM meals
        WHERE substr(created_at, 1, 10) = ?
    """, (today,))

    row = cursor.fetchone()
    conn.close()
    return row


def get_daily_calorie_history():
    conn = get_connection()
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
    conn.close()
    return rows


def get_daily_macro_history():
    conn = get_connection()
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
    conn.close()
    return rows


def get_weekly_summary():
    conn = get_connection()
    cursor = conn.cursor()

    week_start = _local_week_start_str(6)
    cursor.execute("""
        SELECT
            COALESCE(SUM(calories), 0),
            COALESCE(SUM(protein), 0),
            COALESCE(SUM(carbs), 0),
            COALESCE(SUM(fat), 0)
        FROM meals
        WHERE substr(created_at, 1, 10) >= ?
    """, (week_start,))

    row = cursor.fetchone()
    conn.close()
    return row


def meal_exists(meal_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM meals WHERE id = ?", (int(meal_id),))
    count = cursor.fetchone()[0]

    conn.close()
    return count > 0


def delete_meal(meal_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM meals WHERE id = ?", (int(meal_id),))

    conn.commit()
    conn.close()


def clear_all_meals():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM meals")

    conn.commit()
    conn.close()


def get_recent_foods(limit=8):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT food_name, MAX(created_at) AS last_seen
        FROM meals
        GROUP BY food_name
        ORDER BY last_seen DESC
        LIMIT ?
    """, (int(limit),))
    rows = cursor.fetchall()
    conn.close()
    return [r[0] for r in rows]


def get_favorite_foods():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT food_name
        FROM favorite_meals
        ORDER BY created_at DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [r[0] for r in rows]


def add_favorite_food(food_name):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR IGNORE INTO favorite_meals (food_name, created_at)
        VALUES (?, ?)
    """, (str(food_name), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()


def remove_favorite_food(food_name):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM favorite_meals WHERE food_name = ?", (str(food_name),))
    conn.commit()
    conn.close()