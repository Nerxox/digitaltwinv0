import requests
import json

def test_prediction():
    url = "http://localhost:8000/api/v1/predict"
    headers = {"Content-Type": "application/json"}
    data = {
        "machine_id": "machine_001",
        "horizon_min": 30
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        print("Prediction successful!")
        print("Response:", json.dumps(response.json(), indent=2))
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        if response.text:
            print("Response:", response.text)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_prediction()
