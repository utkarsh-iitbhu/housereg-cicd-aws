from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import joblib
from typing import List
import numpy as np
from house_price_pipeline import HousePricePipeline
from pathlib import Path

app = FastAPI()

# Setup templates directory
templates = Jinja2Templates(directory="templates")

class HouseFeatures(BaseModel):
    avg_area_income: float
    avg_area_house_age: float
    avg_area_number_of_rooms: float
    avg_area_number_of_bedrooms: float
    area_population: float

class PredictionResponse(BaseModel):
    predicted_price: float

@app.on_event("startup")
async def load_model():
    global pipeline
    pipeline = HousePricePipeline()
    pipeline.preprocessing_pipeline = joblib.load('preprocessing_pipeline.pkl')

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: HouseFeatures):
    try:
        # Convert input features to DataFrame
        input_data = pd.DataFrame([{
            'Avg. Area Income': features.avg_area_income,
            'Avg. Area House Age': features.avg_area_house_age,
            'Avg. Area Number of Rooms': features.avg_area_number_of_rooms,
            'Avg. Area Number of Bedrooms': features.avg_area_number_of_bedrooms,
            'Area Population': features.area_population
        }])
        
        # Make prediction
        prediction = pipeline.predict(input_data)[0]
        
        return PredictionResponse(predicted_price=float(prediction))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)