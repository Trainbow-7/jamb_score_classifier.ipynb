from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
import joblib
import pandas as pd
import os

# Initialize FastAPI app
app = FastAPI(
    title="JAMB Tier Classifier API",
    description="Predict JAMB score performance tier (Low, Average, High) based on student educational factors.",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
MODEL_PATH = "jamb_tier_classifier.joblib"
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

# Define the input data structure with Pydantic for validation & documentation
class PredictionInput(BaseModel):
    Age: float = Field(default=17.0, description="Age of the student (years)")
    Gender: Literal["Male", "Female"] = Field(default="Female", description="Gender of the student")
    Study_Hours_Per_Week: float = Field(default=15.0, description="Average study hours per week")
    Attendance_Rate: float = Field(default=85.0, description="Class attendance rate (0 to 100%)")
    Assignments_Completed: float = Field(default=8.0, description="Number of assignments completed")
    School_Type: Literal["Public", "Private"] = Field(default="Public", description="School governance type")
    School_Location: Literal["Urban", "Rural"] = Field(default="Urban", description="School location")
    Teacher_Quality: float = Field(default=4.0, description="Teacher quality rating (1 to 5)")
    Distance_To_School: float = Field(default=5.0, description="Distance from residence to school (in km)")
    Extra_Tutorials: Literal["Yes", "No"] = Field(default="Yes", description="Attending extra tutorials or coaching")
    Access_To_Learning_Materials: Literal["Yes", "No"] = Field(default="Yes", description="Access to textbooks & online resources")
    Parent_Involvement: Literal["Low", "Medium", "High"] = Field(default="High", description="Level of parental involvement")
    IT_Knowledge: Literal["Low", "Medium", "High"] = Field(default="Medium", description="Student's IT/digital literacy level")
    Socioeconomic_Status: Literal["Low", "Middle", "High"] = Field(default="Middle", description="Family socioeconomic background")
    Parent_Education_Level: Literal["None", "Primary", "Secondary", "Tertiary"] = Field(default="Tertiary", description="Highest education level attained by parents")

    model_config = {
        "json_schema_extra": {
            "example": {
                "Age": 17.0,
                "Gender": "Female",
                "Study_Hours_Per_Week": 15.0,
                "Attendance_Rate": 85.0,
                "Assignments_Completed": 8.0,
                "School_Type": "Public",
                "School_Location": "Urban",
                "Teacher_Quality": 4.0,
                "Distance_To_School": 5.0,
                "Extra_Tutorials": "Yes",
                "Access_To_Learning_Materials": "Yes",
                "Parent_Involvement": "High",
                "IT_Knowledge": "Medium",
                "Socioeconomic_Status": "Middle",
                "Parent_Education_Level": "Tertiary"
            }
        }
    }

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JAMB Score Tier Predictor</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #7c3aed;
            --primary-dark: #6d28d9;
            --primary-light: #ede9fe;
            --secondary: #0ea5e9;
            --accent: #f59e0b;
            --bg-gradient: linear-gradient(135deg, #f5f3ff 0%, #f0fdf4 50%, #eff6ff 100%);
            --card-bg: #ffffff;
            --text-main: #1e1b4b;
            --text-muted: #64748b;
            --border-color: #e2e8f0;
            --border-focus: #8b5cf6;
            --shadow-sm: 0 2px 4px rgba(0,0,0,0.04);
            --shadow-md: 0 10px 25px -5px rgba(124, 58, 237, 0.08), 0 8px 10px -6px rgba(124, 58, 237, 0.04);
            --shadow-lg: 0 20px 35px -10px rgba(124, 58, 237, 0.15);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        body {
            background: var(--bg-gradient);
            color: var(--text-main);
            min-height: 100vh;
            padding: 40px 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .container {
            width: 100%;
            max-width: 780px;
        }

        .header {
            text-align: center;
            margin-bottom: 32px;
        }

        .header-icon {
            width: 54px;
            height: 54px;
            background: linear-gradient(135deg, #8b5cf6, #ec4899);
            border-radius: 18px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 26px;
            margin-bottom: 16px;
            box-shadow: 0 10px 20px -5px rgba(139, 92, 246, 0.5);
            animation: float 4s ease-in-out infinite;
        }

        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-6px); }
        }

        .header h1 {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #6d28d9 0%, #9333ea 50%, #c026d3 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
            letter-spacing: -0.02em;
        }

        .header p {
            font-size: 1rem;
            color: var(--text-muted);
        }

        .form-card {
            background: var(--card-bg);
            border-radius: 20px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: var(--shadow-md);
            border: 1.5px solid rgba(226, 232, 240, 0.8);
            transition: transform 0.2s, box-shadow 0.2s;
        }

        .form-card:hover {
            box-shadow: var(--shadow-lg);
        }

        .card-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 20px;
            padding-bottom: 12px;
            border-bottom: 1px solid #f1f5f9;
        }

        .card-icon {
            width: 36px;
            height: 36px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
        }

        .icon-student { background: #ede9fe; color: #7c3aed; }
        .icon-school { background: #e0f2fe; color: #0284c7; }
        .icon-support { background: #fef3c7; color: #d97706; }

        .card-header h2 {
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--text-main);
        }

        .grid-2 {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 16px;
        }

        @media (max-width: 640px) {
            .grid-2 {
                grid-template-columns: 1fr;
            }
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .form-group.full-width {
            grid-column: 1 / -1;
        }

        label {
            font-size: 0.875rem;
            font-weight: 600;
            color: #334155;
        }

        input, select {
            width: 100%;
            padding: 11px 14px;
            border-radius: 10px;
            border: 1.5px solid var(--border-color);
            background: #f8fafc;
            color: var(--text-main);
            font-size: 0.95rem;
            outline: none;
            transition: all 0.2s;
        }

        input:focus, select:focus {
            background: #ffffff;
            border-color: var(--border-focus);
            box-shadow: 0 0 0 4px rgba(139, 92, 246, 0.12);
        }

        .btn-submit {
            width: 100%;
            padding: 16px;
            border: none;
            border-radius: 14px;
            background: linear-gradient(135deg, #7c3aed 0%, #9333ea 100%);
            color: white;
            font-size: 1.1rem;
            font-weight: 700;
            cursor: pointer;
            box-shadow: 0 10px 20px -5px rgba(124, 58, 237, 0.4);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            transition: all 0.2s ease;
            margin-top: 10px;
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 14px 25px -5px rgba(124, 58, 237, 0.5);
        }

        .btn-submit:active {
            transform: translateY(0);
        }

        .result-container {
            display: none;
            margin-top: 30px;
            background: white;
            border-radius: 20px;
            padding: 28px;
            box-shadow: var(--shadow-lg);
            text-align: center;
            animation: slideUp 0.35s ease;
            border: 2px solid #ede9fe;
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .result-badge {
            display: inline-block;
            padding: 8px 24px;
            border-radius: 9999px;
            font-size: 1.35rem;
            font-weight: 800;
            letter-spacing: 0.03em;
            margin: 12px 0 16px;
            text-transform: uppercase;
        }

        .tier-High {
            background: linear-gradient(135deg, #10b981, #059669);
            color: white;
            box-shadow: 0 8px 20px -4px rgba(16, 185, 129, 0.4);
        }

        .tier-Average {
            background: linear-gradient(135deg, #f59e0b, #d97706);
            color: white;
            box-shadow: 0 8px 20px -4px rgba(245, 158, 11, 0.4);
        }

        .tier-Low {
            background: linear-gradient(135deg, #ef4444, #dc2626);
            color: white;
            box-shadow: 0 8px 20px -4px rgba(239, 68, 68, 0.4);
        }

        .result-desc {
            font-size: 1rem;
            color: #475569;
            line-height: 1.6;
            max-width: 520px;
            margin: 0 auto;
        }

        .footer {
            margin-top: 40px;
            text-align: center;
            font-size: 0.85rem;
            color: var(--text-muted);
        }

        .footer a {
            color: var(--primary);
            text-decoration: none;
            font-weight: 600;
        }

        .footer a:hover {
            text-decoration: underline;
        }

        .spinner {
            display: none;
            width: 22px;
            height: 22px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <div class="header-icon">✦</div>
            <h1>JAMB Score Tier Predictor</h1>
            <p>Enter the student's details to predict their likely performance tier.</p>
        </header>

        <form id="predictionForm">
            <!-- Student Info -->
            <div class="form-card">
                <div class="card-header">
                    <div class="card-icon icon-student">🎓</div>
                    <h2>Student Info</h2>
                </div>
                <div class="grid-2">
                    <div class="form-group">
                        <label for="Age">Age</label>
                        <input type="number" id="Age" name="Age" value="17" min="10" max="40" required placeholder="e.g. 17">
                    </div>
                    <div class="form-group">
                        <label for="Gender">Gender</label>
                        <select id="Gender" name="Gender" required>
                            <option value="Female" selected>Female</option>
                            <option value="Male">Male</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="Study_Hours_Per_Week">Study Hours / Week</label>
                        <input type="number" step="0.5" id="Study_Hours_Per_Week" name="Study_Hours_Per_Week" value="15" min="0" max="80" required placeholder="e.g. 15">
                    </div>
                    <div class="form-group">
                        <label for="Attendance_Rate">Attendance Rate (%)</label>
                        <input type="number" step="0.5" id="Attendance_Rate" name="Attendance_Rate" value="85" min="0" max="100" required placeholder="0–100">
                    </div>
                    <div class="form-group full-width">
                        <label for="Assignments_Completed">Assignments Completed</label>
                        <input type="number" id="Assignments_Completed" name="Assignments_Completed" value="8" min="0" max="20" required placeholder="e.g. 8">
                    </div>
                </div>
            </div>

            <!-- School Info -->
            <div class="form-card">
                <div class="card-header">
                    <div class="card-icon icon-school">🏫</div>
                    <h2>School Info</h2>
                </div>
                <div class="grid-2">
                    <div class="form-group">
                        <label for="School_Type">School Type</label>
                        <select id="School_Type" name="School_Type" required>
                            <option value="Public" selected>Public</option>
                            <option value="Private">Private</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="School_Location">School Location</label>
                        <select id="School_Location" name="School_Location" required>
                            <option value="Urban" selected>Urban</option>
                            <option value="Rural">Rural</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="Teacher_Quality">Teacher Quality (1-5)</label>
                        <input type="number" step="0.5" id="Teacher_Quality" name="Teacher_Quality" value="4" min="1" max="5" required placeholder="1 to 5">
                    </div>
                    <div class="form-group">
                        <label for="Distance_To_School">Distance to School (km)</label>
                        <input type="number" step="0.5" id="Distance_To_School" name="Distance_To_School" value="5" min="0" max="50" required placeholder="e.g. 5.0">
                    </div>
                </div>
            </div>

            <!-- Support & Background -->
            <div class="form-card">
                <div class="card-header">
                    <div class="card-icon icon-support">🤝</div>
                    <h2>Support & Background</h2>
                </div>
                <div class="grid-2">
                    <div class="form-group">
                        <label for="Extra_Tutorials">Extra Tutorials</label>
                        <select id="Extra_Tutorials" name="Extra_Tutorials" required>
                            <option value="Yes" selected>Yes</option>
                            <option value="No">No</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="Access_To_Learning_Materials">Access to Learning Materials</label>
                        <select id="Access_To_Learning_Materials" name="Access_To_Learning_Materials" required>
                            <option value="Yes" selected>Yes</option>
                            <option value="No">No</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="Parent_Involvement">Parent Involvement</label>
                        <select id="Parent_Involvement" name="Parent_Involvement" required>
                            <option value="High" selected>High</option>
                            <option value="Medium">Medium</option>
                            <option value="Low">Low</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="IT_Knowledge">IT Knowledge</label>
                        <select id="IT_Knowledge" name="IT_Knowledge" required>
                            <option value="Medium" selected>Medium</option>
                            <option value="High">High</option>
                            <option value="Low">Low</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="Socioeconomic_Status">Socioeconomic Status</label>
                        <select id="Socioeconomic_Status" name="Socioeconomic_Status" required>
                            <option value="Middle" selected>Middle</option>
                            <option value="High">High</option>
                            <option value="Low">Low</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="Parent_Education_Level">Parent Education Level</label>
                        <select id="Parent_Education_Level" name="Parent_Education_Level" required>
                            <option value="Tertiary" selected>Tertiary</option>
                            <option value="Secondary">Secondary</option>
                            <option value="Primary">Primary</option>
                            <option value="None">None</option>
                        </select>
                    </div>
                </div>
            </div>

            <button type="submit" class="btn-submit" id="submitBtn">
                <span id="btnText">Predict Performance Tier</span>
                <div class="spinner" id="btnSpinner"></div>
            </button>
        </form>

        <!-- Prediction Result Container -->
        <div class="result-container" id="resultContainer">
            <h3 style="color: #64748b; font-size: 0.95rem; font-weight: 600;">PREDICTED PERFORMANCE TIER</h3>
            <div id="tierBadge" class="result-badge">---</div>
            <p id="tierDescription" class="result-desc"></p>
        </div>

        <footer class="footer">
            <p>JAMB Score Tier Classifier &bull; <a href="/docs" target="_blank">View API Docs</a></p>
        </footer>
    </div>

    <script>
        const form = document.getElementById('predictionForm');
        const submitBtn = document.getElementById('submitBtn');
        const btnText = document.getElementById('btnText');
        const btnSpinner = document.getElementById('btnSpinner');
        const resultContainer = document.getElementById('resultContainer');
        const tierBadge = document.getElementById('tierBadge');
        const tierDescription = document.getElementById('tierDescription');

        const descriptions = {
            'High': 'Excellent indicators! The student demonstrates strong study habits, consistent attendance, and supportive factors that correlate with top-tier performance.',
            'Average': 'Moderate performance indicators. The student is well-positioned, and targeted improvement in study hours or tutoring could boost their score to the highest tier.',
            'Low': 'Needs targeted support. Factors suggest the student may benefit from additional tutorials, structured study schedules, and access to learning materials.'
        };

        form.addEventListener('submit', async (e) => {
            e.preventDefault();

            btnText.textContent = 'Analyzing...';
            btnSpinner.style.display = 'block';
            submitBtn.disabled = true;

            const payload = {
                Age: parseFloat(document.getElementById('Age').value),
                Gender: document.getElementById('Gender').value,
                Study_Hours_Per_Week: parseFloat(document.getElementById('Study_Hours_Per_Week').value),
                Attendance_Rate: parseFloat(document.getElementById('Attendance_Rate').value),
                Assignments_Completed: parseFloat(document.getElementById('Assignments_Completed').value),
                School_Type: document.getElementById('School_Type').value,
                School_Location: document.getElementById('School_Location').value,
                Teacher_Quality: parseFloat(document.getElementById('Teacher_Quality').value),
                Distance_To_School: parseFloat(document.getElementById('Distance_To_School').value),
                Extra_Tutorials: document.getElementById('Extra_Tutorials').value,
                Access_To_Learning_Materials: document.getElementById('Access_To_Learning_Materials').value,
                Parent_Involvement: document.getElementById('Parent_Involvement').value,
                IT_Knowledge: document.getElementById('IT_Knowledge').value,
                Socioeconomic_Status: document.getElementById('Socioeconomic_Status').value,
                Parent_Education_Level: document.getElementById('Parent_Education_Level').value,
            };

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (!response.ok) {
                    throw new Error('Server returned ' + response.status);
                }

                const data = await response.json();
                const tier = data.predicted_tier;

                tierBadge.textContent = tier + ' Tier';
                tierBadge.className = 'result-badge tier-' + tier;
                tierDescription.textContent = descriptions[tier] || 'Model prediction completed successfully.';

                resultContainer.style.display = 'block';
                resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            } catch (err) {
                alert('Prediction request failed. Please verify the API is running.');
                console.error(err);
            } finally {
                btnText.textContent = 'Predict Performance Tier';
                btnSpinner.style.display = 'none';
                submitBtn.disabled = false;
            }
        });
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse, tags=["Web App"])
async def serve_home():
    return HTMLResponse(content=HTML_TEMPLATE)

@app.post("/predict", tags=["Prediction"])
async def predict_tier(data: PredictionInput):
    # Convert input data to pandas DataFrame
    input_df = pd.DataFrame([data.model_dump() if hasattr(data, 'model_dump') else data.dict()])

    if model is None:
        return {"error": "Model file not found on server"}

    # Make prediction
    prediction = model.predict(input_df)[0]

    return {"predicted_tier": str(prediction)}
