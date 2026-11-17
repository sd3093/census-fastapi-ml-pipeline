# Deploying a Machine Learning Model with FastAPI

A complete ML pipeline that trains a classifier on the UCI Census Income data, checks its fairness across data slices, and serves predictions through a **FastAPI** REST API. **GitHub Actions** CI runs `flake8` and `pytest` on every push.

The model predicts whether a person earns `<=50K` or `>50K` a year.

## Repository layout

| Path | Purpose |
|---|---|
| `data/census.csv` | Census Income dataset (32,561 rows) |
| `ml/data.py` | `process_data` (one-hot encoding + label binarizing) and `apply_label` |
| `ml/model.py` | `train_model`, `inference`, `compute_model_metrics`, `save_model` / `load_model`, `performance_on_categorical_slice` |
| `train_model.py` | Loads and splits the data, trains the model, saves `model/model.pkl`, `model/encoder.pkl` and `model/lb.pkl`, prints test metrics, and writes `slice_output.txt` |
| `slice_output.txt` | Precision / recall / F1 for every value of every categorical feature on the test set |
| `main.py` | FastAPI app: `GET /` (welcome message), `POST /data/` (model inference) |
| `local_api.py` | Calls the running API with a GET and a POST and prints the status codes and results |
| `test_ml.py` | 9 pytest unit tests (data processing, training, inference, exact metrics, save/load, slice metrics, and the API) |
| `model_card.md` | Model card: details, intended use, data, metrics, ethics, caveats |
| `.github/workflows/manual.yml` | CI: flake8 + pytest on push / pull request to `main`/`master` |
| `screenshots/` | `continuous_integration.png`, `unit_test.png`, `local_api.png` |

## Environment setup

Python 3.10 is used both locally and in CI.

```bash
# Option 1: pip
python3.10 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Option 2: conda
conda env create -f environment.yml && conda activate fastapi
```

## Train the model

```bash
python train_model.py
```

This trains a `RandomForestClassifier` on an 80/20 stratified train-test split. It saves the model and encoders to `model/`, prints the test metrics, and regenerates `slice_output.txt`:

```
Precision: 0.8018 | Recall: 0.5599 | F1: 0.6594
```

## Run the tests and linting

```bash
pytest test_ml.py -v
flake8 .
```

## Run the API

```bash
uvicorn main:app --reload
```

- `GET http://127.0.0.1:8000/` returns `{"message": "Welcome to the Census Income Prediction API!"}`.
- `POST http://127.0.0.1:8000/data/` takes one census record (hyphenated field names) and returns `{"result": "<=50K"}` or `{"result": ">50K"}`.
- Interactive docs are at http://127.0.0.1:8000/docs.

With the server running, in a second terminal:

```bash
python local_api.py
```

```
Status Code: 200
Result: Welcome to the Census Income Prediction API!
Status Code: 200
Result: <=50K
```

## Continuous integration

`.github/workflows/manual.yml` installs `requirements.txt` on Python 3.10. It then runs `flake8 .` (configured in `setup.cfg`; any error fails the build) and `pytest test_ml.py -v` on every push and pull request to `main` or `master`.

## Model card

See [model_card.md](model_card.md).
