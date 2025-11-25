<p align="center">
  <h1>🕵️‍♂️ Traffic Pattern Detective — Pattern Detective (kmeans-api)</h1>
  <p>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python"/>
  </a>
  <a href="https://flask.palletsprojects.com/">
    <img src="https://img.shields.io/badge/Flask-3.0.0-lightgrey?logo=flask&logoColor=black" alt="Flask"/>
  </a>
  <a href="https://scikit-learn.org/">
    <img src="https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange?logo=scikit-learn&logoColor=white" alt="Scikit-Learn"/>
  </a>
  <a href="https://www.docker.com/">
    <img src="https://img.shields.io/badge/Docker-Container-blue?logo=docker&logoColor=white" alt="Docker"/>
  </a>
  <a href="https://render.com/">
    <img src="https://img.shields.io/badge/Render-Ready-5A67D8?logo=render&logoColor=white" alt="Render"/>
  </a>
  </p>
  <em>Microservice for profiling road "personalities" using a pre-trained K‑Means model</em>
</p>

---

**🧐 About the "Pattern Detective" Model**

- **Purpose:** Unlike prediction models in the Traffic Flow Prediction system that forecast future counts, the Pattern Detective microservice analyzes the static "personality" of a road — i.e., what typical traffic-volume cluster a location belongs to based on a representative daily volume.
- **Technique:** Unsupervised learning (K‑Means) trained on daily traffic-volume samples. Given a simulated daily volume for a coordinate, the model returns a cluster describing that road's typical traffic behavior.
- **Clusters (as defined in `app.py`):**
  - Quiet Road — Low daily traffic volume. Usually free‑flowing. (green)
  - Standard City Road — Moderate daily volume. Typical urban patterns. (amber)
  - Busy Arterial Road — High daily volume. Prone to regular congestion. (orange)
  - Major Junction / Hub — Extreme daily volume. Critical traffic node. (red)

These cluster labels are sourced from the `CLUSTER_LABELS` mapping in `app.py` and are returned to API clients as human‑readable descriptions plus a color code useful for visualizations.

---

**⚙️ How It Works (The Logic Pipeline)**

1. Input: Client sends geographic coordinates (latitude and longitude) to `POST /predict/`.
2. Simulation Engine (Transparent):
   - We do not require a live sensor database to make this microservice useful.
   - To provide *consistent* and *reproducible* behavior for any coordinate, the service deterministically simulates a "Daily Traffic Volume" using a seed derived from the coordinates.
   - Implementation detail: the seed is created as `int((lat + lng) * 10000)`, then `np.random.seed(seed)` is called and `np.random.randint(10000, 60000)` is used to generate a pseudo-random daily volume in the same range the K‑Means model was trained on.
3. Scaling & Clustering:
   - The simulated daily volume is placed into a 2D array, scaled with the saved scaler (`scaler.transform`) and passed into the loaded `kmeans` model (`kmeans.predict`) to obtain a `clusterID`.
4. Profiling:
   - The cluster ID is mapped to a human readable label and description using the `CLUSTER_LABELS` dict in `app.py`.
   - The response contains `trafficState`, `description`, `color`, `volumeAnalyzed` and additional visualization payload data.

This design ensures deterministic, stable outputs for identical coordinates while remaining transparent that these are simulated volumes (useful for visualization, UI decisions, and downstream routing heuristics in the Gateway).

---

**🔌 API Documentation**

- Base URL: `http://<host>:8004` (local) — the service listens on port `8004` by default.

- Endpoint: `POST /predict/`

- Request (JSON):

```json
{
  "coordinates": { "lat": 37.7749, "lng": -122.4194 }
}
```

- Successful Response (JSON):

```json
{
  "clusterID": 0,
  "trafficState": "Busy Arterial Road",
  "description": "High daily volume. Prone to regular congestion.",
  "color": "#FF9800",
  "volumeAnalyzed": "38234 (Daily Total)",
  "estimatedHourly": "~3186 vehicles/hour",
  "patternDistribution": {
    "labels": ["Quiet Road", "Standard Road", "Busy Arterial", "Major Hub"],
    "data": [35, 30, 20, 15]
  }
}
```

- Notes on the response fields:
  - `clusterID`: internal cluster index returned by the K‑Means model.
  - `trafficState`: human label for quick display.
  - `description`: short contextual description for UX/tooltips.
  - `color`: hex code intended for UI charts (e.g., maps, pies).
  - `volumeAnalyzed`: the deterministic, simulated daily volume used for classification.
  - `estimatedHourly`: a rough hourly estimate for quick reference.
  - `patternDistribution`: a small set of distribution data intended for pie charts or overview dashboards. These values are currently static/dummy values in `app.py` but formatted for immediate use.

---

**🛠️ Setup & Installation (Local)**

1. Clone the repository

```bash
git clone <repo-url>
cd "K-Means Clustering"
```

2. Create and activate a virtual environment (recommended `.venv`)

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows (bash.exe)
```

3. Git LFS (important)

The trained model artifacts (`*.pkl`) are binary files that should be stored with Git LFS to avoid repository bloat and ensure proper transfers.

```bash
git lfs install
git lfs track "*.pkl"
git add .gitattributes
git commit -m "track pkl with LFS"
```

4. Install Python dependencies

```bash
pip install -r requirements.txt
```

5. Run locally

```bash
python app.py
# Or for production-like server
gunicorn -w 4 -b 0.0.0.0:8004 app:app
```

After starting, visit `http://localhost:8004/` to confirm the microservice is running.

---

**🐳 Docker & Deployment**

- The repository contains a `Dockerfile` for containerized deployment. Example local build and run:

```bash
docker build -t pattern-detective:latest .
docker run -p 8004:8004 pattern-detective:latest
```

- Render deployment notes
  - Push the repository to GitHub with `*.pkl` tracked by Git LFS.
  - In Render, create a new Web Service and connect your repo.
  - Use Docker or the Python environment — ensure the service is exposed on port `8004`.
  - Configure any environment variables (if you externalize model paths) and enable automatic deploys on push.

---

**🧩 Integration: Microservice Role & Gateway**

- Role: This microservice is intentionally narrow in scope — it performs pattern analysis for a supplied coordinate and returns a small, human‑friendly profile useful for UI, routing decisions, or data enrichment.
- Integration pattern: A Node.js Gateway (or API Gateway) should call this microservice as an internal service. The Gateway can batch requests, cache results per coordinate, and combine Pattern Detective outputs with predictive models for richer responses (e.g., annotate predictions with `trafficState`).

---

**Security & Reliability Notes**

- Transparency: Generated volumes are simulated and deterministic for reproducibility; they are not live sensor measurements.
- Caching: Because the simulation is deterministic, caching responses per coordinate (or per hashed coordinate) is safe and recommended to reduce repeated compute and lower backend calls.

---

**Files of interest**

- `app.py` — Flask app and prediction logic (seeded simulation + K‑Means inference)
- `data/kmeans_model.pkl` — serialized scikit‑learn K‑Means model (tracked with Git LFS)
- `data/kmeans_scaler.pkl` — serialized scaler used to normalize input
- `requirements.txt` — Python dependencies

---

If you want, I can:
- Run a quick local test request and show sample output,
- Add an OpenAPI/Swagger wrapper for interactive docs,
- Convert `patternDistribution` to be dynamically computed from training statistics.

Drop a note on which next step you prefer.

---

_Generated on behalf of the Pattern Detective microservice — clear, deterministic, and ready for integration._
