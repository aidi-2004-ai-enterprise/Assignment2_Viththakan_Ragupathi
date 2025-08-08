# 🐧 Lab 3 – Penguins Species Classification with XGBoost & FastAPI

**Author:** Viththakan Ragupathi  
**Course:** AIDI‑2004 – AI Enterprise  
**Assignment:** Lab 3  
**Goal:** Build and deploy a machine learning model to classify penguin species.

---

## 🧠 Project Description

This project involves training an **XGBoost** classifier using the [Palmer Penguins dataset](https://github.com/allisonhorst/palmerpenguins) and deploying it as a web API using **FastAPI**. The API accepts penguin measurements as input and returns the predicted species.

It includes:

- Clean preprocessing (one-hot for categorical fields, label encoding for target)
- XGBoost classifier with anti-overfitting parameters
- RESTful API endpoint (`/predict`) with input validation
- Interactive Swagger UI at `/docs`
- Model and encoder saved for reuse during prediction

---

## 🗂️ Project Structure
```bash
Lab3_Viththakan_Ragupathi/
├── train.py # Train model and save artifacts
├── app/
│ ├── main.py # FastAPI app
│ └── data/
│ ├── model.json # Saved XGBoost model
│ └── preprocess_meta.json # Stores encoders & label mappings
├── requirements.txt
└── README.md # This file
```



## 🚀 Getting Started

### 1. Install Dependencies

First, create a virtual environment and activate it:

```bash
python -m venv .venv
.venv\Scripts\activate   # (for Windows)
Then install the required libraries:
```
Then install the required libraries:

```bash
pip install -r requirements.txt
```
2. Train the Model
   
Run the training script:
```bash
python train.py
```
This will generate:

model.json: the trained model

preprocess_meta.json: encoders and label mappings used during training

3. Start the API Server
```bash
uvicorn app.main:app --reload
```
Open your browser and visit: http://127.0.0.1:8000/docs to access the Swagger UI.

🎯 API Endpoint
POST /predict
Predicts the species of a penguin based on input features.

✅ Sample Request Body
```bash
{
  "bill_length_mm": 44.0,
  "bill_depth_mm": 17.5,
  "flipper_length_mm": 190,
  "body_mass_g": 4200,
  "year": 2008,
  "sex": "female",
  "island": "Torgersen"
}
```
✅ Sample Response
```bash
{
  "species": "Adelie"
}
```
❌ Error Handling
Invalid Category
If an invalid value is sent for an enum field (e.g., island: Atlantis):

```bash
{
  "detail": [
    {
      "loc": ["body", "island"],
      "msg": "value is not a valid enumeration member",
      "type": "type_error.enum"
    }
  ]
}
```
Missing Field
If a required field is missing:
```bash
{
  "detail": [
    {
      "loc": ["body", "sex"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```
📦 Dependencies
Make sure your requirements.txt includes at least:

1. fastapi
2. uvicorn
3. scikit-learn
4. pandas
5. xgboost
6. pydantic

📚 Dataset Info
The Palmer Penguins dataset includes measurements for three species:

1. Adelie
2. Chinstrap
3. Gentoo

Features used in the model:

1. bill_length_mm
2. bill_depth_mm
3. flipper_length_mm
4. body_mass_g
5. year
6. sex
7. island

Here's the modified Q&A section formatted for a GitHub README.md, using bolding and clear separation for readability.

### Technical Q&A
## What edge cases might break your model in production that aren't in your training data?
- Unseen categorical values: New species, sexes, or island values.

- Out-of-distribution numerical values: Extremely large or negative body mass.

- Missing features or unexpected schema changes in incoming requests.

- Non-English or malformed inputs if data is user-submitted.

## What happens if your model file becomes corrupted?
- The API will fail to load the model at startup.

- Inference endpoints (/predict) will throw runtime exceptions.

- Solution: Add model validation, fallbacks, and monitor model health on startup.

## What's a realistic load for a penguin classification service?
- Low to moderate: 5–50 requests/sec is realistic for demo or educational tools.

- In production, design for spikes up to 100–200 RPS with auto-scaling.

## How would you optimize if response times are too slow?
- Use batch inference if traffic is predictable.

- Load the model once at startup, avoiding reloading on each request.

- Profile bottlenecks using APM tools (e.g., New Relic, Datadog).

- Host the model in memory (e.g., via TorchServe or TF Serving for heavy models).

- Scale horizontally with multiple containers.

## What metrics matter most for ML inference APIs?
- Latency (p50, p95, p99): End-to-end response time.

- Throughput (RPS): How many requests can be served per second.

- Error rate: Percentage of failed requests.

- Model confidence distribution: Anomaly detection.

- Resource usage: CPU, memory.

## Why is Docker layer caching important for build speed? (Did you leverage it?)
- Reuses unchanged layers (e.g., Python packages, OS updates), drastically speeding up rebuilds.

- Yes, it was used by ordering Dockerfile steps properly (e.g., installing requirements before copying the full code).

## What security risks exist with running containers as root?
- Root access inside the container can lead to potential root access to the host.

- Increases the blast radius of container breakout attacks.

-Best practice: Use a non-root user with the least privileges (USER appuser in Dockerfile).

## How does cloud auto-scaling affect your load test results?
- Initial load spikes may fail if instances aren’t "warm" yet.

- Cold starts increase latency temporarily.

- Metrics like response time may only normalize after the system scales up.

- You must include a ramp-up phase in load testing.

## What would happen with 10x more traffic?
- Increased latency or timeouts if horizontal scaling isn’t configured.

- Possible OOM errors if too many requests overload memory.

- The backend (e.g., DB or storage) might become a bottleneck.

- Requires a stress-tested auto-scaling configuration, queuing, and circuit breaking.

## How would you monitor performance in production?
- Use Prometheus + Grafana or Datadog for:

- Request/response times

- CPU/memory usage

- Error rate (5xx/4xx)

- Log predictions and status codes with correlation IDs.

- Alert on latency thresholds or failure spikes.

## How would you implement blue-green deployment?
- Deploy a new version (green) alongside the current one (blue).

- Route a small percentage of traffic to green and monitor its behavior.

- Gradually shift 100% of traffic to green if there are no issues.

- Roll back instantly to blue on failure.

## What would you do if deployment fails in production?
- Automatically roll back to the last working image/version.

- Log the error and alert the ops team.

- Use feature flags or deployment gates to isolate issues before a full rollout.

## What happens if your container uses too much memory?
- It may be killed by the OS (OOMKilled).

- This causes degraded performance or a failure to respond to requests.

- Mitigation: Set memory limits in docker-compose or Kubernetes, and profile memory usage during tests.
