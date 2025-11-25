# kmeans-api/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)

# --- 1. Load Models ---
MODEL_PATH = os.path.join("data", "kmeans_model.pkl")
SCALER_PATH = os.path.join("data", "kmeans_scaler.pkl")

kmeans = None
scaler = None

try:
    kmeans = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"✅ K-Means model and scaler loaded.")
except Exception as e:
    print(f"❌ FATAL: Error loading models: {e}")

# --- 2. Define Cluster Labels (Based on Daily Volume) ---
# These define the "Personality" of the road based on 24-hour capacity.
CLUSTER_LABELS = {
    1: { # ~13k Daily
        "label": "Quiet Road",
        "description": "Low daily traffic volume. Usually free-flowing.",
        "color": "#4CAF50" # Green
    },
    3: { # ~25k Daily
        "label": "Standard City Road",
        "description": "Moderate daily volume. Typical urban traffic patterns.",
        "color": "#FFC107" # Amber
    },
    0: { # ~38k Daily
        "label": "Busy Arterial Road",
        "description": "High daily volume. Prone to regular congestion.",
        "color": "#FF9800" # Orange
    },
    2: { # ~52k Daily
        "label": "Major Junction / Hub",
        "description": "Extreme daily volume. Critical traffic node.",
        "color": "#F44336" # Red
    }
}

# --- 3. Prediction Endpoint ---
@app.route('/predict/', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        lat = data['coordinates']['lat']
        lng = data['coordinates']['lng']
        
        # --- SIMULATION LOGIC ---
        # Since we don't have live sensors, we simulate the "Daily Volume"
        # for this specific coordinate to see what "Type" of road it is.
        
        # 1. Create a stable seed from coordinates (so the result is consistent)
        seed = int((lat + lng) * 10000)
        np.random.seed(seed)
        
        # 2. Generate a random DAILY volume (10k to 60k)
        # This matches the range your K-Means model was trained on.
        simulated_daily_volume = np.random.randint(10000, 60000)
        
        # --- PREDICTION ---
        # 1. Reshape & Scale
        input_data = np.array([[simulated_daily_volume]])
        scaled_data = scaler.transform(input_data)
        
        # 2. Predict Cluster ID
        cluster_id = int(kmeans.predict(scaled_data)[0])
        
        # 3. Get Label info
        cluster_info = CLUSTER_LABELS.get(cluster_id, {
            "label": "Unknown",
            "description": "Pattern not recognized",
            "color": "#9E9E9E"
        })

        # --- RESPONSE ---
        # We calculate an "Estimated Hourly Avg" just for user reference
        est_hourly = int(simulated_daily_volume / 12)

        response = {
            "clusterID": cluster_id,
            "trafficState": cluster_info["label"], # e.g. "Busy Arterial Road"
            "description": cluster_info["description"],
            "color": cluster_info["color"],
            "volumeAnalyzed": f"{simulated_daily_volume} (Daily Total)",
            "estimatedHourly": f"~{est_hourly} vehicles/hour",
            
            # Chart Data: Distribution of road types in the city
            "patternDistribution": {
                "labels": ["Quiet Road", "Standard Road", "Busy Arterial", "Major Hub"],
                "data": [35, 30, 20, 15] # Dummy distribution stats
            }
        }
        
        return jsonify(response)

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "K-Means Clustering API is running"})

if __name__ == '__main__':
    # Run on port 8004
    app.run(host='0.0.0.0', port=8004)