# 🚀 PT100 Sensor Anomaly Detection API

FastAPI service layer for real-time sensor fault detection using Isolation Forest machine learning model.

## 📋 Features

- **CRUD Operations**: Complete Create, Read, Update, Delete endpoints for sensor records
- **ML Predictions**: Real-time anomaly detection using trained Isolation Forest model
- **Auto Standardization**: Automatic value normalization using StandardScaler
- **Batch Predictions**: Process multiple values in a single request
- **Health Monitoring**: API and model status endpoints
- **Statistical Analytics**: Built-in statistics calculation
- **OpenAPI Documentation**: Interactive API docs with Swagger UI

## 🛠️ Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Ensure model is trained:**
```bash
python src/train_isolation_forest.py
```

3. **Run the API server:**
```bash
cd api
python main.py
```

Or use uvicorn directly:
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## 📡 API Endpoints

### Root & Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/health` | Health check and model status |
| GET | `/docs` | Interactive API documentation (Swagger UI) |
| GET | `/redoc` | Alternative API documentation (ReDoc) |

### CRUD Operations

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/records` | Get all sensor records |
| GET | `/records/{id}` | Get single record by ID |
| POST | `/records` | Create new sensor record |
| PUT | `/records/{id}` | Update existing record |
| DELETE | `/records/{id}` | Delete record by ID |

### ML Predictions

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Predict anomaly for single value |
| POST | `/predict/batch` | Predict anomalies for multiple values |

### Analytics

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/statistics` | Get statistical summary of all records |

## 📝 Usage Examples

### 1. Health Check

```bash
curl -X GET "http://localhost:8000/health"
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true,
  "total_records": 5,
  "timestamp": "2024-12-05T02:00:00"
}
```

### 2. Get All Records

```bash
curl -X GET "http://localhost:8000/records"
```

Response:
```json
[
  {
    "id": 1,
    "timestamp": "2024-12-05T01:00:00",
    "value": 24.5
  },
  {
    "id": 2,
    "timestamp": "2024-12-05T01:05:00",
    "value": 25.1
  }
]
```

### 3. Create New Record

```bash
curl -X POST "http://localhost:8000/records" \
  -H "Content-Type: application/json" \
  -d '{"value": 26.3, "timestamp": "2024-12-05T02:00:00"}'
```

Response:
```json
{
  "id": 6,
  "timestamp": "2024-12-05T02:00:00",
  "value": 26.3
}
```

### 4. Update Record

```bash
curl -X PUT "http://localhost:8000/records/1" \
  -H "Content-Type: application/json" \
  -d '{"value": 24.8}'
```

Response:
```json
{
  "id": 1,
  "timestamp": "2024-12-05T01:00:00",
  "value": 24.8
}
```

### 5. Delete Record

```bash
curl -X DELETE "http://localhost:8000/records/1"
```

Response: `204 No Content`

### 6. Predict Anomaly (Normal Value)

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"value": 24.5}'
```

Response:
```json
{
  "value": 24.5,
  "standardized_value": 0.0554,
  "anomaly_score": -0.4721,
  "anomaly_label": 0,
  "prediction": "NORMAL",
  "confidence": "High",
  "timestamp": "2024-12-05T02:00:00"
}
```

### 7. Predict Anomaly (Anomalous Value)

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"value": 150.0}'
```

Response:
```json
{
  "value": 150.0,
  "standardized_value": 23.2553,
  "anomaly_score": -0.7063,
  "anomaly_label": 1,
  "prediction": "ANOMALY",
  "confidence": "High",
  "timestamp": "2024-12-05T02:00:00"
}
```

### 8. Batch Prediction

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '[{"value": 24.5}, {"value": 150.0}, {"value": 25.1}]'
```

Response:
```json
[
  {
    "value": 24.5,
    "anomaly_label": 0,
    "prediction": "NORMAL",
    ...
  },
  {
    "value": 150.0,
    "anomaly_label": 1,
    "prediction": "ANOMALY",
    ...
  },
  {
    "value": 25.1,
    "anomaly_label": 0,
    "prediction": "NORMAL",
    ...
  }
]
```

### 9. Get Statistics

```bash
curl -X GET "http://localhost:8000/statistics"
```

Response:
```json
{
  "count": 5,
  "mean": 50.26,
  "std": 50.91,
  "min": 23.8,
  "25th_percentile": 24.5,
  "median": 24.9,
  "75th_percentile": 25.1,
  "max": 149.6,
  "timestamp": "2024-12-05T02:00:00"
}
```

## 🧪 Using Interactive API Documentation

1. Start the server
2. Open browser to: `http://localhost:8000/docs`
3. Try out endpoints directly from the Swagger UI
4. View request/response schemas
5. Test with example values

## 📊 Prediction Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `value` | float | Original input temperature |
| `standardized_value` | float | Z-score normalized value |
| `anomaly_score` | float | Model confidence score (lower = more anomalous) |
| `anomaly_label` | int | Binary classification: 0 = normal, 1 = anomaly |
| `prediction` | string | Human-readable result: "NORMAL" or "ANOMALY" |
| `confidence` | string | Confidence level: "High", "Medium", or "Low" |
| `timestamp` | string | Prediction timestamp (ISO 8601) |

## 🔒 Confidence Levels

### For Anomalies (label = 1):
- **High**: Score < -0.65 (very confident it's anomalous)
- **Medium**: Score -0.65 to -0.55
- **Low**: Score > -0.55

### For Normal Readings (label = 0):
- **High**: Score > -0.45 (very confident it's normal)
- **Medium**: Score -0.50 to -0.45
- **Low**: Score < -0.50

## 🛡️ Error Handling

The API provides detailed error responses:

### 404 Not Found
```json
{
  "detail": "Record with ID 99 not found"
}
```

### 422 Validation Error
```json
{
  "detail": [
    {
      "loc": ["body", "value"],
      "msg": "Temperature must be between -50°C and 200°C",
      "type": "value_error"
    }
  ]
}
```

### 503 Service Unavailable
```json
{
  "detail": "Model is not loaded. Please train the model first."
}
```

## 📁 Project Structure

```
api/
├── main.py           # Complete FastAPI application
├── __init__.py       # Package initializer
└── README.md         # This file

data/
├── models/
│   ├── isolation_forest_model.pkl  # Trained ML model
│   └── standard_scaler.pkl         # Feature scaler
└── processed/
    └── sensor_data_standardized.csv

src/
├── config.py                    # Configuration paths
├── train_isolation_forest.py   # Model training script
└── ...
```

## 🚦 Production Deployment

For production deployment, consider:

1. **Use production ASGI server:**
```bash
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

2. **Add environment variables:**
```python
import os
MODEL_PATH = os.getenv("MODEL_PATH", "data/models/isolation_forest_model.pkl")
```

3. **Enable CORS** (if needed):
```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"])
```

4. **Add authentication** (JWT, API keys, etc.)

5. **Database integration** (PostgreSQL, MongoDB, etc.)

6. **Logging and monitoring** (Sentry, ELK stack, etc.)

## 📞 Support

For issues or questions, please contact the development team.

## 📄 License

MIT License
