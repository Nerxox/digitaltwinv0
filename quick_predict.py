from backend_ai.services.lstm_predictor import LSTMPredictor
p = LSTMPredictor()
p.load_model()
print(p.predict("machine_001", 3))