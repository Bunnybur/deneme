"""
FastAPI Application - Sensor Fault Detection System
====================================================
RESTful API for real-time sensor fault detection using ML.

Endpoints:
- POST /predict: Predict if a sensor value is anomalous
- GET /readings: Get all stored readings
- POST /readings: Create a new reading
- PUT /readings/{id}: Update a reading
- DELETE /readings/{id}: Delete a reading
- GET /stats: Get statistics

Author: Advanced Computer Programming Course
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import joblib
import numpy as np
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MODEL_PATH, SCALER_PATH, API_HOST, API_PORT

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class PredictionRequest(BaseModel):
    """Request model for /predict endpoint."""
    value: float = Field(..., description="Temperature value in °C")
    
    class Config:
        json_schema_extra = {"example": {"value": 25.5}}

class PredictionResponse(BaseModel):
    """Response model for /predict endpoint."""
    value: float
    status: str
    confidence_score: float = Field(..., description="Anomaly score from model")
    
    class Config:
        json_schema_extra = {
            "example": {"value": 25.5, "status": "Normal", "confidence_score": -0.15}
        }

class ReadingCreate(BaseModel):
    """Request model for creating a reading."""
    timestamp: str
    value: float
    
    class Config:
        json_schema_extra = {
            "example": {"timestamp": "2024-12-04T21:00:00+03:00", "value": 22.5}
        }

class ReadingUpdate(BaseModel):
    """Request model for updating a reading."""
    timestamp: Optional[str] = None
    value: Optional[float] = None
    
    class Config:
        json_schema_extra = {"example": {"value": 30.0}}

class Reading(BaseModel):
    """Response model for a reading."""
    id: int
    timestamp: str
    value: float
    status: str
    confidence_score: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "timestamp": "2024-12-04T21:00:00+03:00",
                "value": 22.5,
                "status": "Normal",
                "confidence_score": -0.15
            }
        }

class DeleteResponse(BaseModel):
    """Response for delete operations."""
    message: str
    deleted_id: int

# ============================================================================
# FASTAPI APP SETUP
# ============================================================================

app = FastAPI(
    title="Sensor Fault Detection API",
    description="IoT sensor fault detection using Isolation Forest ML",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# GLOBAL STATE
# ============================================================================

# In-memory database
readings_db: Dict[int, dict] = {}
next_id: int = 1

# ML model and scaler
model = None
scaler = None

@app.on_event("startup")
async def load_models():
    """Load ML model and scaler on startup."""
    global model, scaler
    
    print("\n" + "="*60)
    print("SENSOR FAULT DETECTION API - STARTING")
    print("="*60)
    
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            print("\n✓ ML model loaded successfully")
            print(f"  Model: {MODEL_PATH}")
            print(f"  Scaler: {SCALER_PATH}")
        else:
            print("\n⚠️  Warning: Model files not found")
            print("  Run the data pipeline first:")
            print("    1. python src/data_clean.py")
            print("    2. python src/data_standardization.py")
            print("    3. python src/train_model.py")
    except Exception as e:
        print(f"\n⚠️  Error loading model: {e}")
    
    print("\n" + "="*60)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def predict_value(value: float) -> tuple:
    """
    Predict if a value is anomalous.
    
    Returns:
        tuple: (status_str, confidence_score)
    """
    if model is None or scaler is None:
        # Fallback: threshold-based
        if value > 100 or value < -50:
            return "Fault", -1.0
        return "Normal", 0.0
    
    try:
        value_scaled = scaler.transform([[value]])
        prediction = model.predict(value_scaled)[0]
        score = model.score_samples(value_scaled)[0]
        
        status = "Fault" if prediction == -1 else "Normal"
        return status, float(score)
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "Normal", 0.0

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Sensor Fault Detection API",
        "version": "2.0.0",
        "status": "Model loaded" if model is not None else "Model not loaded",
        "endpoints": {
            "docs": "/docs",
            "predict": "/predict",
            "readings": "/readings",
            "stats": "/stats"
        }
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_fault(request: PredictionRequest):
    """
    Predict if a sensor reading is anomalous.
    
    Uses Isolation Forest trained on historical data.
    Returns prediction and confidence score.
    """
    status, score = predict_value(request.value)
    
    return PredictionResponse(
        value=request.value,
        status=status,
        confidence_score=score
    )

@app.get("/readings", response_model=List[Reading], tags=["CRUD"])
async def get_all_readings():
    """Retrieve all stored sensor readings."""
    return list(readings_db.values())

@app.post("/readings", response_model=Reading, status_code=status.HTTP_201_CREATED, tags=["CRUD"])
async def create_reading(reading: ReadingCreate):
    """
    Create a new sensor reading with automatic classification.
    
    The reading is automatically classified using the ML model.
    """
    global next_id
    
    # Classify using ML
    classification, score = predict_value(reading.value)
    
    # Create reading
    new_reading = {
        "id": next_id,
        "timestamp": reading.timestamp,
        "value": reading.value,
        "status": classification,
        "confidence_score": score
    }
    
    readings_db[next_id] = new_reading
    next_id += 1
    
    return new_reading

@app.get("/readings/{reading_id}", response_model=Reading, tags=["CRUD"])
async def get_reading(reading_id: int):
    """Retrieve a specific reading by ID."""
    if reading_id not in readings_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Reading {reading_id} not found"
        )
    return readings_db[reading_id]

@app.put("/readings/{reading_id}", response_model=Reading, tags=["CRUD"])
async def update_reading(reading_id: int, update: ReadingUpdate):
    """Update an existing reading. Re-classifies if value changes."""
    if reading_id not in readings_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Reading {reading_id} not found"
        )
    
    reading = readings_db[reading_id]
    
    if update.timestamp is not None:
        reading["timestamp"] = update.timestamp
    
    if update.value is not None:
        reading["value"] = update.value
        # Re-classify
        status, score = predict_value(update.value)
        reading["status"] = status
        reading["confidence_score"] = score
    
    return reading

@app.delete("/readings/{reading_id}", response_model=DeleteResponse, tags=["CRUD"])
async def delete_reading(reading_id: int):
    """Delete a reading."""
    if reading_id not in readings_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Reading {reading_id} not found"
        )
    
    del readings_db[reading_id]
    
    return DeleteResponse(
        message="Reading deleted successfully",
        deleted_id=reading_id
    )

@app.get("/stats", tags=["Statistics"])
async def get_statistics():
    """Get database statistics."""
    total = len(readings_db)
    
    if total == 0:
        return {
            "total_readings": 0,
            "normal_count": 0,
            "fault_count": 0,
            "fault_percentage": 0.0
        }
    
    normal_count = sum(1 for r in readings_db.values() if r["status"] == "Normal")
    fault_count = total - normal_count
    
    return {
        "total_readings": total,
        "normal_count": normal_count,
        "fault_count": fault_count,
        "fault_percentage": round((fault_count / total) * 100, 2)
    }

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\nStarting API server...")
    print(f"Documentation: http://localhost:{API_PORT}/docs")
    print("Press CTRL+C to stop\n")
    
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True
    )
