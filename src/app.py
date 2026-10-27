"""
FastAPI service that serves real-time predictions from the trained model.

Run locally:
    uvicorn src.app:app --host 0.0.0.0 --port 8000
"""
import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="Student Academic Risk Prediction API")

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
label_encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))
feature_columns = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))

static_dir = os.path.join(BASE_DIR, "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


class StudentFeatures(BaseModel):
    class Config:
        extra = "allow"


@app.get("/")
def root():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Student Academic Risk Prediction API. See /docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: dict):
    try:
        df = pd.DataFrame([data])
        df = pd.get_dummies(df, columns=["Course"], drop_first=True) if "Course" in df.columns else df

        # align incoming columns to the exact training-time feature order
        df = df.reindex(columns=feature_columns, fill_value=0)

        X_scaled = scaler.transform(df)
        prediction = model.predict(X_scaled)
        label = label_encoder.inverse_transform(prediction)[0]

        return {"prediction": label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000)
