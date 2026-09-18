
import gradio as gr
import joblib
import pandas as pd
from pathlib import Path


MODEL_PATH = (
    Path(__file__).resolve().parent
    / "logistic_regression_StudyHrs_model.pkl"
)


model = joblib.load(MODEL_PATH)


def predict_result(study_hours):

    if study_hours is None:
        return "Please enter study hours"

    new_data = pd.DataFrame({
        "StudyHours": [study_hours]
    })

    prediction = model.predict(new_data)[0]

    if prediction == 1:
        return "PASS ✅"
    else:
        return "FAIL ❌"


app = gr.Interface(
    fn=predict_result,
    inputs=gr.Number(
        label="Enter Study Hours",
        minimum=0
    ),
    outputs=gr.Textbox(
        label="Prediction"
    ),
    title="Student Pass/Fail Prediction",
    description="Predict whether a student will pass based on study hours."
)


app.launch(
    server_name="0.0.0.0",
    server_port=7860
)
