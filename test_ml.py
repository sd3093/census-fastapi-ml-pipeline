import os

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier

from main import app
from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_PATH, "data", "census.csv")

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


@pytest.fixture(scope="module")
def data():
    """A small, reproducible sample of the census data."""
    df = pd.read_csv(DATA_PATH)
    return df.sample(n=1000, random_state=0)


@pytest.fixture(scope="module")
def processed(data):
    """Training arrays plus the fitted encoder and label binarizer."""
    X, y, encoder, lb = process_data(
        data, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    return X, y, encoder, lb


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_process_data_shapes_and_labels(data, processed):
    """
    process_data returns one row per input row, a binary label vector,
    and the same number of columns when re-used in inference mode.
    """
    X, y, encoder, lb = processed
    assert X.shape[0] == data.shape[0]
    assert y.shape == (data.shape[0],)
    assert set(np.unique(y)) <= {0, 1}

    X_inf, _, _, _ = process_data(
        data, categorical_features=CAT_FEATURES, label="salary",
        training=False, encoder=encoder, lb=lb,
    )
    assert X_inf.shape == X.shape


def test_train_model_returns_random_forest(processed):
    """train_model returns a fitted RandomForestClassifier."""
    X, y, _, _ = processed
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "estimators_")  # only present after fit()


def test_inference_returns_binary_predictions(processed):
    """inference returns one 0/1 prediction per row."""
    X, y, _, _ = processed
    model = train_model(X, y)
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == y.shape
    assert set(np.unique(preds)) <= {0, 1}


def test_compute_model_metrics_exact_values():
    """compute_model_metrics returns the exact precision, recall and F1."""
    y = np.array([1, 1, 0, 0, 1, 0])
    preds = np.array([1, 0, 0, 1, 1, 0])
    # TP=2, FP=1, FN=1 -> precision 2/3, recall 2/3, F1 2/3
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == pytest.approx(2 / 3)
    assert recall == pytest.approx(2 / 3)
    assert fbeta == pytest.approx(2 / 3)


def test_save_and_load_model_roundtrip(processed, tmp_path):
    """A saved model loads back and makes identical predictions."""
    X, y, _, _ = processed
    model = train_model(X, y)
    path = tmp_path / "model.pkl"
    save_model(model, str(path))
    loaded = load_model(str(path))
    assert np.array_equal(inference(model, X), inference(loaded, X))


def test_performance_on_categorical_slice_returns_valid_metrics(data, processed):
    """Slice metrics are three floats between 0 and 1."""
    X, y, encoder, lb = processed
    model = train_model(X, y)
    metrics = performance_on_categorical_slice(
        data, "sex", "Female", CAT_FEATURES, "salary", encoder, lb, model
    )
    assert len(metrics) == 3
    assert all(0.0 <= m <= 1.0 for m in metrics)


def test_api_get_root(client):
    """GET / returns 200 and a welcome message."""
    r = client.get("/")
    assert r.status_code == 200
    assert "Welcome" in r.json()["message"]


def test_api_post_predicts_low_income(client):
    """POST /data/ classifies a young, low-hours profile as <=50K."""
    sample = {
        "age": 19,
        "workclass": "Private",
        "fnlgt": 215646,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Never-married",
        "occupation": "Handlers-cleaners",
        "relationship": "Own-child",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 20,
        "native-country": "United-States",
    }
    r = client.post("/data/", json=sample)
    assert r.status_code == 200
    assert r.json() == {"result": "<=50K"}


def test_api_post_predicts_high_income(client):
    """POST /data/ classifies a high-earning profile as >50K."""
    sample = {
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
    r = client.post("/data/", json=sample)
    assert r.status_code == 200
    assert r.json() == {"result": ">50K"}
