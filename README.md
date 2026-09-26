
---
title: JAMB Tier Classifier
models:
- Gradient Boosting
tags:
- scikit-learn
- classification
- jamb
---

# JAMB Tier Classifier

This model classifies JAMB (Joint Admissions and Matriculation Board) scores into three tiers: 'Low', 'Average', and 'High'. It is trained on features such as study hours, attendance rate, teacher quality, etc.

## 🚀 Live Demo

Experience the model in action through the interactive web application:
👉 **[JAMB Score Tier Predictor Live App](https://jamb-score-classifier-ipynb.onrender.com/)**




## Model Details

- **Model Name**: Gradient Boosting
- **Framework**: scikit-learn
- **Type**: Classification
- **Target**: JAMB Score Tier (Low, Average, High)

## Performance (on test set)

- **Accuracy**: 0.574
- **Macro F1-Score**: 0.543

### Classification Report:

```
              precision    recall  f1-score   support

     Average       0.49      0.55      0.52       383
        High       0.59      0.35      0.44       189
         Low       0.65      0.70      0.67       428

    accuracy                           0.57      1000
   macro avg       0.58      0.53      0.54      1000
weighted avg       0.58      0.57      0.57      1000

```

## Cross-validation Performance (5-fold StratifiedKFold)

- **Macro F1-Score**: 0.515 +/- 0.005

## How to use

```python
import joblib
from huggingface_hub import hf_hub_download

# Download the model from Hugging Face Hub
model_path = hf_hub_download(repo_id="Trainbow/jamb-tier-classifier-model", filename="jamb_tier_classifier.joblib")
model = joblib.load(model_path)

# Example prediction (replace with your actual data)
# from your_preprocessing_steps import preprocessor_scaled
# features = preprocessor_scaled.transform(your_new_data)
# prediction = model.predict(features)
# print(prediction)
```

## Training Code

The full training pipeline can be found in the associated Colab notebook.

