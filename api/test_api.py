"""
API Test Script
===============
Test all endpoints of the FastAPI service.

Run this script to verify that the API is working correctly.
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def print_response(title, response):
    """Print formatted API response."""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    print(f"Status Code: {response.status_code}")
    try:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except:
        print(f"Response: {response.text}")

def test_api():
    """Run comprehensive API tests."""
    print("\n🧪 TESTING PT100 SENSOR ANOMALY DETECTION API")
    print("="*70)
    
    # Test 1: Root endpoint
    response = requests.get(f"{BASE_URL}/")
    print_response("TEST 1: Root Endpoint", response)
    
    # Test 2: Health check
    response = requests.get(f"{BASE_URL}/health")
    print_response("TEST 2: Health Check", response)
    
    # Test 3: Get all records
    response = requests.get(f"{BASE_URL}/records")
    print_response("TEST 3: Get All Records", response)
    
    # Test 4: Get single record
    response = requests.get(f"{BASE_URL}/records/1")
    print_response("TEST 4: Get Record by ID (ID=1)", response)
    
    # Test 5: Create new record
    new_record = {
        "value": 26.5,
        "timestamp": "2024-12-05T02:30:00"
    }
    response = requests.post(f"{BASE_URL}/records", json=new_record)
    print_response("TEST 5: Create New Record", response)
    created_id = response.json().get("id") if response.status_code == 201 else None
    
    # Test 6: Update record
    if created_id:
        update_data = {"value": 27.0}
        response = requests.put(f"{BASE_URL}/records/{created_id}", json=update_data)
        print_response(f"TEST 6: Update Record (ID={created_id})", response)
    
    # Test 7: Predict normal value
    normal_prediction = {"value": 24.5}
    response = requests.post(f"{BASE_URL}/predict", json=normal_prediction)
    print_response("TEST 7: Predict NORMAL Value (24.5°C)", response)
    
    # Test 8: Predict anomalous value
    anomaly_prediction = {"value": 150.0}
    response = requests.post(f"{BASE_URL}/predict", json=anomaly_prediction)
    print_response("TEST 8: Predict ANOMALY Value (150.0°C)", response)
    
    # Test 9: Batch prediction
    batch_predictions = [
        {"value": 24.5},
        {"value": 150.0},
        {"value": 25.1},
        {"value": 10.0}
    ]
    response = requests.post(f"{BASE_URL}/predict/batch", json=batch_predictions)
    print_response("TEST 9: Batch Prediction", response)
    
    # Test 10: Get statistics
    response = requests.get(f"{BASE_URL}/statistics")
    print_response("TEST 10: Get Statistics", response)
    
    # Test 11: Delete record
    if created_id:
        response = requests.delete(f"{BASE_URL}/records/{created_id}")
        print_response(f"TEST 11: Delete Record (ID={created_id})", response)
    
    # Test 12: 404 Error - Record not found
    response = requests.get(f"{BASE_URL}/records/9999")
    print_response("TEST 12: 404 Error - Record Not Found", response)
    
    print("\n" + "="*70)
    print("✅ API TESTING COMPLETE!")
    print("="*70)
    print("\n💡 View interactive documentation at:")
    print(f"   📍 {BASE_URL}/docs")
    print(f"   📍 {BASE_URL}/redoc")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        test_api()
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API server")
        print("   Make sure the server is running:")
        print("   python api/main.py")
        print()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print()
