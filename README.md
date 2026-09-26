
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

## 🚀 Interactive Live Demo & API

- **Interactive Swagger UI (Try it Live)**: [https://jamb-score-classifier-ipynb.onrender.com/docs](https://jamb-score-classifier-ipynb.onrender.com/docs)
- **Prediction Endpoint (POST)**: `https://jamb-score-classifier-ipynb.onrender.com/predict`

> **Note**: To test predictions in the browser, open the **[Swagger UI link above](https://jamb-score-classifier-ipynb.onrender.com/docs)**, expand the **`POST /predict`** endpoint, click **"Try it out"**, and hit **"Execute"**. (Opening `/predict` directly in a browser sends a `GET` request which will show `Method Not Allowed`).

### Example API Request (cURL)
```bash
curl -X POST "https://jamb-score-classifier-ipynb.onrender.com/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "Study_Hours_Per_Week": 15.0,
       "Attendance_Rate": 85.0,
       "Teacher_Quality": 4.0,
       "Distance_To_School": 5.0,
       "Age": 17.0,
       "Assignments_Completed": 9.0,
       "School_Type": "Public",
       "School_Location": "Urban",
       "Extra_Tutorials": "Yes",
       "Access_To_Learning_Materials": "Yes",
       "Gender": "Female",
       "Parent_Involvement": "High",
       "IT_Knowledge": "Medium",
       "Socioeconomic_Status": "Middle",
       "Parent_Education_Level": "Tertiary"
     }'
```



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

