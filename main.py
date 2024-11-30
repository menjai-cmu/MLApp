from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import requests
import os
from io import BytesIO
from typing import List, Dict
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="ChurnML Predictor")

# Enable CORS for all origins (use restricted origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable for the model
model = None

# Define Pydantic models for the response
class Prediction(BaseModel):
    CustomerID: str
    CLTV: float
    ChargesPerMonth: float
    ChurnProbabilities: float

class PredictionResponse(BaseModel):
    predictions: List[Prediction]

# Preprocessing function
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data before prediction.
    """
    df = df.dropna(subset=['Total Charges'])
    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
    df['ChargesPerMonth'] = df['Total Charges'] / df['Tenure Months']
    df = df.drop(
        [
            'Count', 'Country', 'State', 'City', 'Zip Code',
            'Lat Long', 'Latitude', 'Longitude', 'Churn Label',
            'Churn Score', 'Churn Reason', 'Total Charges',
            'Monthly Charges', 'Churn Value'
        ],
        axis=1
    )
    df = pd.get_dummies(df)
    return df

@app.on_event("startup")
async def load_model():
    """
    Load the model on application startup.
    """
    global model
    try:
        # Load model from the URL or file
        model_path = os.getenv("MODEL_PATH", "xgb_model.joblib")
        logger.info("Loading model from local file...")
        model = joblib.load(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """
    Serve the index.html file for the frontend.
    """
    try:
        with open("index.html") as file:
            return HTMLResponse(content=file.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html not found")

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}

@app.post("/predict-churn", response_model=PredictionResponse)
async def predict_churn(request: Request, file: UploadFile = File(...)) -> PredictionResponse:
    """
    Predict churn probability from an uploaded CSV file.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Load the uploaded CSV file
        df = pd.read_csv(file.file)

        # Validate required columns
        required_columns = ['CustomerID', 'Total Charges', 'Tenure Months']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Uploaded file is missing required columns: {', '.join(missing_columns)}"
            )

        # Preprocess the data
        customer_ids = df['CustomerID']
        df = preprocess_data(df)

        # Ensure features align with the model
        model_features = model.feature_names_in_
        missing_features = [col for col in model_features if col not in df.columns]
        if missing_features:
            logger.warning(f"Missing required features: {', '.join(missing_features)}")
            raise HTTPException(
                status_code=400,
                detail=f"Input data is missing required feature columns: {', '.join(missing_features)}"
            )
        
        # Align data to model input
        df_features = df.reindex(columns=model_features, fill_value=0)

        # Predict churn probabilities
        churn_probabilities = model.predict_proba(df_features)[:, 1]

        # Prepare the response
        results = []
        for cust_id, cltv, charges, prob in zip(
            customer_ids, df.get('CLTV', [0] * len(customer_ids)),
            df.get('ChargesPerMonth', [0] * len(customer_ids)), churn_probabilities
        ):
            # Ensure floats are JSON-compliant
            results.append({
                "CustomerID": str(cust_id),
                "CLTV": float(cltv) if pd.notna(cltv) else 0.0,
                "ChargesPerMonth": float(charges) if pd.notna(charges) else 0.0,
                "ChurnProbabilities": float(prob) if pd.notna(prob) else 0.0,
            })

        return {"predictions": sorted(results, key=lambda x: x['ChurnProbabilities'], reverse=True)}

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="The uploaded CSV file is empty")
    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Invalid CSV format")
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
