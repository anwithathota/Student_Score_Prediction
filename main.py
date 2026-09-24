from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import os

# -----------------------------------
# Create FastAPI application
# -----------------------------------

app = FastAPI(
    title="Student Score Prediction",
    description="Student Final Score Prediction using Linear Regression",
    version="1.0"
)


# -----------------------------------
# Load trained ML model
# -----------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(
    os.path.join(BASE_DIR, "model", "student_score_model.pkl")
)


# -----------------------------------
# Connect static folder
# -----------------------------------
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(BASE_DIR, "static")),
    name="static"
)

# -----------------------------------
# Input data format
# -----------------------------------

class StudentData(BaseModel):

    hours_studied: float
    previous_score: float
    attendance: float
    sleep_hours: float


# -----------------------------------
# Home page
# -----------------------------------

@app.get("/", response_class=HTMLResponse)
def home():

    html_file = os.path.join(
        BASE_DIR,
        "templates",
        "index.html"
    )

    with open(
        html_file,
        "r",
        encoding="utf-8"
    ) as file:

        return file.read()

# -----------------------------------
# Prediction API
# -----------------------------------

@app.post("/predict")
def predict(data: StudentData):

    input_data = [[
        data.hours_studied,
        data.previous_score,
        data.attendance,
        data.sleep_hours
    ]]

    prediction = model.predict(input_data)

    predicted_score = prediction[0]

    # Keep score between 0 and 100
    predicted_score = max(
        0,
        min(100, predicted_score)
    )

    return {
        "predicted_score": round(
            float(predicted_score),
            2
        )
    }


# -----------------------------------
# API test route
# -----------------------------------

@app.get("/api")
def api_test():

    return {
        "message": "Student Score Prediction API is running!"
    }