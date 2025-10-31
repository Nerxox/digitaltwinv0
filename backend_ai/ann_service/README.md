# ANN Anomaly Service (FastAPI)

Run locally (Windows PowerShell):

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn backend_ai.ann_service.app:app --reload --port 9000
```

Example request:

```
curl -X POST http://127.0.0.1:9000/api/v1/ann/anomaly_score ^
  -H "Content-Type: application/json" ^
  -d '{
    "machine_id": "M001",
    "series": [{"t":0,"y":1.0},{"t":1,"y":1.1},{"t":2,"y":6.0}]
  }'
```

Replace the placeholder z-score logic with your exported ANN model inference (ONNX/TorchScript) and keep the response shape stable for the web UI.
