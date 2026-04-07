# Calorify AI Model Card

## Model
- Architecture: ResNet50 classifier (custom head)
- Classes: 100

## Evaluation Summary
- Accuracy: 88.50%
- Macro F1: 0.8834
- Weighted F1: 0.8846
- ROC AUC (micro): 0.9986
- PR AUC / MUI (micro): 0.9476
- ECE (10 bins): 0.0760
- Brier score: 0.0018

## Known Limitations
- Performance may drop on mixed dishes, low-light images, or unusual presentation.
- Confidence is probabilistic, not guaranteed correctness.
- Portion and calorie estimation are approximate and should be user-adjusted.
- This system is decision support only, not medical advice.
