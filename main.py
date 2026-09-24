
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
# Assuming the model file is in the same directory as main.py
MODEL_PATH = "jamb_tier_classifier.joblib"
model = joblib.load(MODEL_PATH)

# Define the input data structure for prediction
class PredictionInput(BaseModel):
    Study_Hours_Per_Week: float
    Attendance_Rate: float
    Teacher_Quality: float
    Distance_To_School: float
    Age: float
    Assignments_Completed: float
    School_Type: str
    School_Location: str
    Extra_Tutorials: str
    Access_To_Learning_Materials: str
    Gender: str
    Parent_Involvement: str
    IT_Knowledge: str
    Socioeconomic_Status: str
    Parent_Education_Level: str

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the JAMB Tier Classifier API! Visit /docs for API documentation."}

@app.post("/predict", tags=["Prediction"])
async def predict_tier(data: PredictionInput):
    # Convert input data to pandas DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Make prediction
    prediction = model.predict(input_df)[0]

    return {"predicted_tier": prediction}

# To run this locally with uvicorn:
# uvicorn main:app --reload
