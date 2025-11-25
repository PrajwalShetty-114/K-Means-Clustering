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

# --- 2. Define Cluster Labels (Based on your Colab Output) ---
# We map Cluster ID -> (Label, Description)
# Cluster 1: ~14k (Lowest)
# Cluster 3: ~26k
# Cluster 0: ~38k
# Cluster 2: ~52k (Highest)

CLUSTER_LABELS = {
    1: {
        "label": "Moderate Flow",
        "description": "Traffic is moving reasonably well. Typical for off-peak hours.",
        "color": "#4CAF50" # Green
    },
    3: {
        "label": "High Traffic",
        "description": "Volume is high. Expect some delays and slower speeds.",
        "color": "#FFC107" # Amber/Yellow
    },
    0: {
        "label": "Very High Traffic",
        "description": "Roads are becoming saturated. Significant delays likely.",
        "color": "#FF9800" # Orange
    },
    2: {
        "label": "Severe Congestion",
        "description": "Critical traffic volume. Expect stop-and-go conditions.",
        "color": "#F44336" # Red
    }
}

# --- 3. Prediction Endpoint ---
@app.route('/predict/', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # In a real scenario, we would fetch the *current* live traffic volume
        # for the requested location to classify it.
        # Since we don't have live sensors, we will use a 'Simulated' volume
        # based on the location to show how the clustering works.
        
        # A. Extract Location (We use this to seed our random simulation)
        lat = data['coordinates']['lat']
        lng = data['coordinates']['lng']
        
        # B. Simulate a Traffic Volume (for demonstration)
        # We use the coordinate to generate a consistent pseudo-random number
        # This ensures the same location always gives the same result
        seed = int((lat + lng) * 10000)
        np.random.seed(seed)
        
        # Generate a random volume between 10,000 and 55,000 (your data range)
        simulated_volume = np.random.randint(10000, 55000)
        
        # C. Prepare Data for Model
        # 1. Reshape to 2D array
        input_data = np.array([[simulated_volume]])
        # 2. Scale using the saved scaler
        scaled_data = scaler.transform(input_data)
        
        # D. Predict Cluster
        cluster_id = int(kmeans.predict(scaled_data)[0])
        
        # E. Get Label info
        cluster_info = CLUSTER_LABELS.get(cluster_id, {
            "label": "Unknown",
            "description": "Pattern not recognized",
            "color": "#9E9E9E"
        })

        # F. Response
        response = {
            "clusterID": cluster_id,
            "trafficState": cluster_info["label"],
            "description": cluster_info["description"],
            "color": cluster_info["color"],
            "volumeAnalyzed": simulated_volume,
            # Chart Data: We return the distribution of clusters as "Typical Patterns"
            "patternDistribution": {
                "labels": ["Moderate Flow", "High Traffic", "Very High", "Severe"],
                # Dummy distribution data for the pie chart
                "data": [35, 30, 20, 15] 
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