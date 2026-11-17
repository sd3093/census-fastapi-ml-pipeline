"""Query the deployed (live) Census Income API on Render with a POST request."""
import os

import requests

# Live root domain of the Render web service (override with LIVE_API_URL if needed)
LIVE_API_URL = os.environ.get(
    "LIVE_API_URL", "https://census-fastapi-ml-pipeline.onrender.com"
)

data = {
    "age": 52,
    "workclass": "Self-emp-inc",
    "fnlgt": 287927,
    "education": "Doctorate",
    "education-num": 16,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 15024,
    "capital-loss": 0,
    "hours-per-week": 60,
    "native-country": "United-States",
}

# Send a POST to the live model inference endpoint
r = requests.post(f"{LIVE_API_URL}/data/", json=data, timeout=120)

print(f"POST {LIVE_API_URL}/data/")
# Print the HTTP status code
print(f"Status Code: {r.status_code}")
# Print the model inference result
print(f"Result: {r.json()['result']}")
