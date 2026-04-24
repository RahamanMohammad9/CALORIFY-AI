Calorify AI – Personalized Nutrition Coach

Calorify AI is an AI-powered nutrition system that combines computer vision, data analytics, and behavioral tracking to provide actionable dietary insights.

Overview

Calorify AI goes beyond simple calorie tracking by integrating food recognition, personalized targets, and behavioral analysis to support better dietary decisions.

Quick Start
git clone https://github.com/RahamanMohammad9/CALORIFY-AI.git
cd CALORIFY-AI
pip install -r requirements.txt
streamlit run app/app.py
Dataset

The Food-101 dataset is automatically downloaded on the first run. No manual setup is required.

Model Setup

The trained model is not included due to size constraints.

Place the model file in the following directory:

models/food_model.pth
Problem

Most nutrition applications focus only on tracking food intake. Users often lack guidance on:

what to eat next
whether their intake aligns with their goals
how lifestyle factors affect nutrition outcomes
Solution

Calorify AI provides a decision-support system that combines:

food recognition
personalized nutrition targets
behavioral tracking
intelligent feedback
Key Features
AI Nutrition Coach
Daily feedback on calorie intake
Macro imbalance detection
Goal-based recommendations (fat loss, maintenance, muscle gain)
Personalization Engine
BMI and BMR (Mifflin-St Jeor)
Personalized calorie targets
Macro distribution calculations
Smart Meal Logging
Image-based food recognition
Multi-food detection
Confidence-based validation
Text-based and voice-style logging
Open Food Facts integration
Advanced Analytics
Daily and weekly trends
Goal vs actual comparisons
Macro distribution analysis
30-day weight projection
Behavioral insights (sleep and activity patterns)
Health Tracking
Meals
Water intake
Weight
Sleep
Physical activity
Tech Stack
Frontend: Streamlit
Machine Learning: PyTorch, TorchVision
Data Processing: Pandas, Plotly
Database: SQLite
External API: Open Food Facts
Project Structure
app/
 ├── app.py
 ├── dataset_setup.py
 ├── pages/
 ├── database.py
 ├── profile_utils.py

models/
Screenshots


Limitations
Not a substitute for medical advice
Model predictions may be inaccurate for complex or uncommon foods
Portion estimation is approximate
Nutrition values may vary based on preparation and source data
Future Work
Mobile application (Flutter or React Native)
Barcode scanning
Nutrition chatbot
Restaurant recommendation system
