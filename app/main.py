"""
FastAPI application for classifying penguin species using a trained XGBoost model.
"""

import os
import json
import pandas as pd
from enum import Enum
from dotenv import load_dotenv
from google.cloud import storage
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel
import logging

import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Default to 8080 if PORT is not set
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)


# ------------------------------
# Setup logging
# ------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------
# Load environment variables
# ------------------------------
load_dotenv()

# ------------------------------
# Download model from Google Cloud Storage
# ------------------------------
def download_model_from_gcs():
    bucket_name = os.getenv("GCS_BUCKET_NAME")
    blob_name = os.getenv("GCS_BLOB_NAME")

    if not all([bucket_name, blob_name]):
        logger.error("❌ Missing GCS_BUCKET_NAME or GCS_BLOB_NAME environment variables.")
        raise ValueError("GCS environment variables not properly configured.")

    try:
        # Use default credentials automatically provided by Cloud Run
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        local_model_path = os.path.join(os.path.dirname(__file__), "data", "model.json")
        os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
        blob.download_to_filename(local_model_path)

        logger.info("✅ Model downloaded from GCS to %s", local_model_path)
        return local_model_path
    except Exception as e:
        logger.exception("❌ Error downloading model from GCS.")
        raise e


# ------------------------------
# Load Model and Metadata
# ------------------------------
try:
    model_path = download_model_from_gcs()
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    logger.info("✅ Model loaded from %s", model_path)

    base_dir = os.path.dirname(__file__)

    with open(os.path.join(base_dir, "data", "preprocess_meta.json")) as f:
         feature_columns = json.load(f)["feature_columns"]

    with open(os.path.join(base_dir, "data", "label_classes.json")) as f:
         label_classes = json.load(f)


except Exception as e:
    logger.exception("❌ Failed during model or metadata loading.")
    raise e

# ------------------------------
# FastAPI Setup
# ------------------------------
app = FastAPI(title="Penguin Classifier API", version="1.0")

class Island(str, Enum):
    Torgersen = "Torgersen"
    Biscoe = "Biscoe"
    Dream = "Dream"

class Sex(str, Enum):
    male = "male"
    female = "female"

class PenguinFeatures(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    year: int
    sex: Sex
    island: Island

# ------------------------------
# Preprocessing
# ------------------------------
def preprocess_features(features: PenguinFeatures) -> pd.DataFrame:
    df = pd.DataFrame([features.model_dump()])
    df = pd.get_dummies(df)

    # Align to training columns
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df

# ------------------------------
# API Endpoints
# ------------------------------
@app.get("/")
def root() -> dict:
    return {"message": "Penguin Classification API is running."}

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

@app.post("/predict")
def predict(features: PenguinFeatures) -> dict:
    try:
        X_input = preprocess_features(features)
        prediction = model.predict(X_input)[0]
        logger.info("✅ Prediction made successfully: %s", label_classes[int(prediction)])
        return {
            "prediction": int(prediction),
            "species": label_classes[int(prediction)]
        }
    except Exception as e:
        logger.exception("❌ Prediction failed.")
        return {"error": "Prediction failed", "details": str(e)}
    
    
