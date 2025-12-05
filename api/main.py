"""
FastAPI Service Layer for PT100 Sensor Anomaly Detection
==========================================================
Complete REST API with CRUD operations and ML model predictions.

Features:
- CRUD endpoints for sensor records
- Real-time anomaly detection using Isolation Forest
- Automatic value standardization
- Professional error handling
- Pydantic validation

Author: Advanced Computer Programming Course
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import joblib
import numpy as np
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MODEL_PATH, SCALER_PATH

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class SensorRecord(BaseModel):
    """Schema for sensor temperature reading."""
    id: Optional[int] = None
    timestamp: Optional[str] = Field(None, description="ISO 8601 timestamp")
    value: float = Field(..., ge=-50, le=200, description="Temperature in Celsius")
    
    @validator('value')
    def validate_temperature(cls, v):
        """Validate temperature is within reasonable range."""
        if not -50 <= v <= 200:
            raise ValueError('Temperature must be between -50°C and 200°C')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "id": 1,
                "timestamp": "2024-12-05T02:00:00",
                "value": 24.5
            }
        }


class SensorRecordCreate(BaseModel):
    """Schema for creating a new sensor record."""
    value: float = Field(..., ge=-50, le=200, description="Temperature in Celsius")
    timestamp: Optional[str] = Field(None, description="ISO 8601 timestamp (auto-generated if not provided)")
    
    class Config:
        schema_extra = {
            "example": {
                "value": 24.5,
                "timestamp": "2024-12-05T02:00:00"
            }
        }


class SensorRecordUpdate(BaseModel):
    """Schema for updating an existing sensor record."""
    value: Optional[float] = Field(None, ge=-50, le=200, description="Temperature in Celsius")
    timestamp: Optional[str] = Field(None, description="ISO 8601 timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "value": 25.3,
                "timestamp": "2024-12-05T03:00:00"
            }
        }


class PredictionRequest(BaseModel):
    """Schema for anomaly detection prediction request."""
    value: float = Field(..., description="Temperature value to predict")
    
    @validator('value')
    def validate_value(cls, v):
        """Validate temperature is numeric."""
        if not isinstance(v, (int, float)):
            raise ValueError('Value must be a number')
        return float(v)
    
    class Config:
        schema_extra = {
            "example": {
                "value": 150.0
            }
        }


class PredictionResponse(BaseModel):
    """Schema for anomaly detection prediction response."""
    value: float = Field(..., description="Input temperature value")
    standardized_value: float = Field(..., description="Standardized (z-score) value")
    anomaly_score: float = Field(..., description="Anomaly score (lower = more anomalous)")
    anomaly_label: int = Field(..., description="0 = normal, 1 = anomaly")
    prediction: str = Field(..., description="Human-readable prediction result")
    confidence: str = Field(..., description="Confidence level of prediction")
    timestamp: str = Field(..., description="Prediction timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "value": 150.0,
                "standardized_value": 23.45,
                "anomaly_score": -0.75,
                "anomaly_label": 1,
                "prediction": "ANOMALY",
                "confidence": "High",
                "timestamp": "2024-12-05T02:00:00"
            }
        }


class HealthResponse(BaseModel):
    """Schema for health check endpoint."""
    status: str
    model_loaded: bool
    scaler_loaded: bool
    total_records: int
    timestamp: str


# ============================================================================
# FASTAPI APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title="PT100 Sensor Anomaly Detection API",
    description="REST API for real-time sensor fault detection using Isolation Forest",
    version="1.0.0",
    contact={
        "name": "Sensor Fault Detection Team",
        "email": "support@sensorfault.com"
    },
    license_info={
        "name": "MIT",
    }
)

# ============================================================================
# IN-MEMORY DATABASE (Simple Storage)
# ============================================================================

# In-memory storage for sensor records
sensor_records_db: List[Dict[str, Any]] = [
    {"id": 1, "timestamp": "2024-12-05T01:00:00", "value": 24.5},
    {"id": 2, "timestamp": "2024-12-05T01:05:00", "value": 25.1},
    {"id": 3, "timestamp": "2024-12-05T01:10:00", "value": 23.8},
    {"id": 4, "timestamp": "2024-12-05T01:15:00", "value": 24.9},
    {"id": 5, "timestamp": "2024-12-05T01:20:00", "value": 149.6},  # Anomaly
]

next_id = 6  # Counter for auto-incrementing IDs

# ============================================================================
# MODEL & SCALER LOADING
# ============================================================================

def load_model():
    """Load the trained Isolation Forest model."""
    try:
        model = joblib.load(MODEL_PATH)
        print(f"✓ Model loaded from: {MODEL_PATH}")
        return model
    except FileNotFoundError:
        print(f"⚠ Model not found at: {MODEL_PATH}")
        print(f"⚠ Run 'python src/train_isolation_forest.py' first to train the model")
        return None
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None


def load_scaler():
    """Load the StandardScaler for value normalization."""
    try:
        scaler = joblib.load(SCALER_PATH)
        print(f"✓ Scaler loaded from: {SCALER_PATH}")
        return scaler
    except FileNotFoundError:
        print(f"⚠ Scaler not found at: {SCALER_PATH}")
        print(f"⚠ Using placeholder scaler with mean=24.20, std=5.41")
        return None
    except Exception as e:
        print(f"✗ Error loading scaler: {e}")
        return None


def standardize_value(value: float, scaler=None) -> float:
    """
    Standardize a temperature value using the scaler.
    If scaler is not available, use placeholder statistics from training data.
    
    Formula: z = (x - mean) / std
    """
    if scaler is not None:
        # Use fitted scaler
        value_array = np.array([[value]])
        standardized = scaler.transform(value_array)[0][0]
    else:
        # Fallback: use known statistics from training data
        # These values were computed during model training
        mean = 24.20  # Mean temperature from training data
        std = 5.41    # Std deviation from training data
        standardized = (value - mean) / std
    
    return float(standardized)


# Load model and scaler on startup
isolation_forest_model = load_model()
standard_scaler = load_scaler()

# ============================================================================
# CRUD ENDPOINTS
# ============================================================================

@app.get("/", response_model=Dict[str, str], tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "PT100 Sensor Anomaly Detection API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health_check": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint to verify API and model status."""
    return {
        "status": "healthy" if isolation_forest_model is not None else "degraded",
        "model_loaded": isolation_forest_model is not None,
        "scaler_loaded": standard_scaler is not None,
        "total_records": len(sensor_records_db),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/records", response_model=List[SensorRecord], tags=["CRUD Operations"])
async def get_all_records():
    """
    Retrieve all sensor records.
    
    Returns:
        List of all sensor records in the database.
    """
    return sensor_records_db


@app.get("/records/{record_id}", response_model=SensorRecord, tags=["CRUD Operations"])
async def get_record_by_id(record_id: int):
    """
    Retrieve a single sensor record by ID.
    
    Parameters:
        record_id: Unique identifier of the sensor record
        
    Returns:
        The requested sensor record.
        
    Raises:
        404: Record not found
    """
    for record in sensor_records_db:
        if record["id"] == record_id:
            return record
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Record with ID {record_id} not found"
    )


@app.post("/records", response_model=SensorRecord, status_code=status.HTTP_201_CREATED, tags=["CRUD Operations"])
async def create_record(record: SensorRecordCreate):
    """
    Create a new sensor record.
    
    Parameters:
        record: Sensor record data (value and optional timestamp)
        
    Returns:
        The created sensor record with assigned ID.
    """
    global next_id
    
    # Auto-generate timestamp if not provided
    timestamp = record.timestamp if record.timestamp else datetime.utcnow().isoformat()
    
    new_record = {
        "id": next_id,
        "timestamp": timestamp,
        "value": record.value
    }
    
    sensor_records_db.append(new_record)
    next_id += 1
    
    return new_record


@app.put("/records/{record_id}", response_model=SensorRecord, tags=["CRUD Operations"])
async def update_record(record_id: int, record_update: SensorRecordUpdate):
    """
    Update an existing sensor record.
    
    Parameters:
        record_id: Unique identifier of the record to update
        record_update: Updated record data (partial updates allowed)
        
    Returns:
        The updated sensor record.
        
    Raises:
        404: Record not found
    """
    for i, record in enumerate(sensor_records_db):
        if record["id"] == record_id:
            # Update only provided fields
            if record_update.value is not None:
                sensor_records_db[i]["value"] = record_update.value
            if record_update.timestamp is not None:
                sensor_records_db[i]["timestamp"] = record_update.timestamp
            
            return sensor_records_db[i]
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Record with ID {record_id} not found"
    )


@app.delete("/records/{record_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["CRUD Operations"])
async def delete_record(record_id: int):
    """
    Delete a sensor record by ID.
    
    Parameters:
        record_id: Unique identifier of the record to delete
        
    Raises:
        404: Record not found
    """
    for i, record in enumerate(sensor_records_db):
        if record["id"] == record_id:
            sensor_records_db.pop(i)
            return
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Record with ID {record_id} not found"
    )


# ============================================================================
# PREDICTION ENDPOINT
# ============================================================================

@app.post("/predict", response_model=PredictionResponse, tags=["ML Predictions"])
async def predict_anomaly(request: PredictionRequest):
    """
    Predict whether a temperature value is anomalous using the Isolation Forest model.
    
    The model uses the same standardization approach as during training:
    - Standardizes the input value using StandardScaler
    - Predicts anomaly using Isolation Forest
    - Returns anomaly score and binary label
    
    Parameters:
        request: Temperature value to analyze
        
    Returns:
        Prediction results including:
        - anomaly_score: Lower values = more anomalous
        - anomaly_label: 0 = normal, 1 = anomaly
        - prediction: Human-readable result
        - confidence: Based on score magnitude
        
    Raises:
        503: Model not available
    """
    # Check if model is loaded
    if isolation_forest_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please train the model first."
        )
    
    # Standardize the input value
    standardized_value = standardize_value(request.value, standard_scaler)
    
    # Prepare input for model (needs to be 2D array)
    input_features = np.array([[standardized_value]])
    
    # Get prediction (-1 = anomaly, 1 = normal)
    prediction_raw = isolation_forest_model.predict(input_features)[0]
    
    # Get anomaly score (lower = more anomalous)
    anomaly_score = isolation_forest_model.score_samples(input_features)[0]
    
    # Convert to binary label (0 = normal, 1 = anomaly)
    anomaly_label = 1 if prediction_raw == -1 else 0
    
    # Determine confidence based on score magnitude
    if anomaly_label == 1:
        # For anomalies, lower scores = higher confidence
        if anomaly_score < -0.65:
            confidence = "High"
        elif anomaly_score < -0.55:
            confidence = "Medium"
        else:
            confidence = "Low"
    else:
        # For normal readings, higher (less negative) scores = higher confidence
        if anomaly_score > -0.45:
            confidence = "High"
        elif anomaly_score > -0.50:
            confidence = "Medium"
        else:
            confidence = "Low"
    
    # Human-readable prediction
    prediction_text = "ANOMALY" if anomaly_label == 1 else "NORMAL"
    
    return PredictionResponse(
        value=request.value,
        standardized_value=round(standardized_value, 4),
        anomaly_score=round(float(anomaly_score), 4),
        anomaly_label=anomaly_label,
        prediction=prediction_text,
        confidence=confidence,
        timestamp=datetime.utcnow().isoformat()
    )


# ============================================================================
# BATCH PREDICTION ENDPOINT
# ============================================================================

@app.post("/predict/batch", response_model=List[PredictionResponse], tags=["ML Predictions"])
async def predict_anomaly_batch(requests: List[PredictionRequest]):
    """
    Predict anomalies for multiple temperature values at once.
    
    Parameters:
        requests: List of temperature values to analyze
        
    Returns:
        List of prediction results for each input value.
        
    Raises:
        503: Model not available
    """
    if isolation_forest_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please train the model first."
        )
    
    # Process each request
    results = []
    for req in requests:
        result = await predict_anomaly(req)
        results.append(result)
    
    return results


# ============================================================================
# STATISTICS ENDPOINT
# ============================================================================

@app.get("/statistics", tags=["Analytics"])
async def get_statistics():
    """
    Get statistical summary of all sensor records.
    
    Returns:
        Statistical metrics including count, mean, std, min, max, quartiles.
    """
    if not sensor_records_db:
        return {"message": "No records available"}
    
    values = [record["value"] for record in sensor_records_db]
    values_array = np.array(values)
    
    return {
        "count": len(values),
        "mean": round(float(np.mean(values_array)), 2),
        "std": round(float(np.std(values_array)), 2),
        "min": round(float(np.min(values_array)), 2),
        "25th_percentile": round(float(np.percentile(values_array, 25)), 2),
        "median": round(float(np.median(values_array)), 2),
        "75th_percentile": round(float(np.percentile(values_array, 75)), 2),
        "max": round(float(np.max(values_array)), 2),
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("="*70)
    print("PT100 SENSOR ANOMALY DETECTION API")
    print("="*70)
    print(f"\n🚀 Starting FastAPI server...")
    print(f"📍 API Documentation: http://localhost:8000/docs")
    print(f"📍 Alternative Docs: http://localhost:8000/redoc")
    print(f"💾 Model loaded: {isolation_forest_model is not None}")
    print(f"📊 Scaler loaded: {standard_scaler is not None}")
    print(f"📝 Total records: {len(sensor_records_db)}")
    print("="*70 + "\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
