# Calorify AI - Personalized Nutrition Coach

Calorify AI is an AI-powered personalized nutrition system that combines computer vision, data analytics, and user behavior tracking to provide actionable dietary insights.

## Problem

Most food apps stop at simple logging. Users still struggle with:
- what to eat next,
- whether they are aligned with goals,
- and how daily behavior (sleep/activity) affects nutrition outcomes.

## Solution

Calorify AI goes beyond tracking by providing:
- AI-based meal recognition from food photos
- Personalized calorie and macro targets using profile data
- Daily coaching feedback and smart suggestions
- Trend analytics and 30-day projection
- Full health context with water, weight, sleep, and activity tracking

## Key Features

### 1) AI Nutrition Coach
- Daily feedback such as:
  - over/under calorie status
  - carbs over target
  - protein gap recommendations
- Goal-aware suggestions for fat loss, maintenance, and muscle gain

### 2) Personalization Engine
- User profile inputs:
  - age, gender, height, weight
  - activity level
  - goal (Fat Loss / Maintain / Muscle Gain)
- Calculations:
  - BMI
  - BMR (Mifflin-St Jeor)
  - personalized daily calorie target
  - macro targets (protein/carbs/fat)

### 3) Smart Meal Logging
- Add by photo (ResNet50-based classifier)
- Confidence handling (<60% requires confirmation)
- Multi-food photo mode with per-item portion sliders
- Add by text lookup (local nutrition + Open Food Facts fallback)
- Voice-style quick logging (`"I ate 2 eggs and toast"`)
- Quick add from recent meals and favorites

### 4) Advanced Analytics
- Daily and weekly calorie trends
- Goal vs actual comparisons
- Macro distribution over time
- 30-day projected weight change
- AI insights:
  - night calorie concentration
  - weekend vs weekday intake
  - low-sleep / low-activity day patterns
- CSV export for analyst workflows

### 5) Complete Health Ecosystem
- Meal Tracker
- Water Tracker
- Weight Tracker
- Sleep Tracker
- Activity Tracker

## Tech Stack

- Frontend: Streamlit
- AI/CV: PyTorch, TorchVision (ResNet50)
- Data: Pandas, Plotly
- Storage: SQLite
- External Data: Open Food Facts API (via `httpx`)

## Project Structure

- `app/streamlit_app.py` - app landing page with page links
- `app/app.py` - main dashboard
- `app/pages/` - feature pages (meals, analytics, profile, health trackers)
- `app/database.py` - meal/favorites persistence helpers
- `app/profile_utils.py` - personalization and target calculations
- `models/` - trained model + class labels

## Run Locally

1. Create and activate a virtual environment
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start app:

```bash
streamlit run app/streamlit_app.py
```

## Interview Positioning

Use this line in interviews:

> "I developed an AI-powered personalized nutrition system that combines computer vision, data analytics, and user behavior tracking to provide actionable dietary insights."

## Suggested Demo Flow (2-3 min)

1. Open Dashboard and show personalized target + AI coach feedback
2. Add meal from image (single + multi-food flow)
3. Show analytics (goal vs actual, macro trends, 30-day projection)
4. Show sleep/activity trackers and cross-insights
5. Export CSV and mention analyst-ready workflow

## Model Limitations and Ethics

- This system provides decision support, not medical diagnosis or treatment.
- Food image classification can be incorrect for rare foods, mixed dishes, or poor image quality.
- Confidence scores indicate model certainty, not guaranteed truth.
- Portion estimation is approximate and should be corrected by users when possible.
- Nutrition lookup values can vary by brand, recipe, and preparation method.
- User-entered data quality directly affects analytics quality and projections.
- The app should be used as a coaching aid; important health decisions should involve qualified professionals.

## Future Extensions

- Mobile frontend (Flutter/React Native) + FastAPI backend
- Barcode scanner
- Nutrition chatbot
- Restaurant menu recommendation mode
