# Deploying a Machine Learning Model with FastAPI

A complete ML pipeline that trains a classifier on the UCI Census Income data, checks its fairness across data slices, and serves predictions through a **FastAPI** REST API. **GitHub Actions** CI runs `flake8` and `pytest` on every push, and **Render** continuously deploys the API from `main` after CI passes.

- **Repository platform:** GitHub - https://github.com/sd3093/census-fastapi-ml-pipeline
- **Continuous integration:** GitHub Actions (`.github/workflows/manual.yml`)
- **Continuous deployment:** Render web service, auto-deploy from `main` *after CI checks pass*
- **Live API:** https://census-fastapi-ml-pipeline.onrender.com (interactive docs at `/docs`)

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
| `local_api.py` | Calls the locally running API with a GET and a POST and prints the status codes and results |
| `live_post.py` | Sends a POST to the **live** Render API with `requests` and prints the status code and inference result |
| `test_ml.py` | 9 pytest unit tests (data processing, training, inference, exact metrics, save/load, slice metrics, and the API) |
| `model_card.md` | Model card: details, intended use, data, metrics, ethics, caveats |
| `.github/workflows/manual.yml` | CI: flake8 + pytest on push / pull request to `main`/`master` |
| `screenshots/` | `continuous_integration.png`, `continuous_deployment.png`, `example.png`, `live_get.png`, `live_post.png`, `unit_test.png`, `local_api.png` |

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

## Continuous deployment (Render)

The API runs as a Render web service connected to this GitHub repository:

| Setting | Value |
|---|---|
| Branch | `main` |
| Build command | `pip install -r requirements.txt` |
| Start command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| Python version | `PYTHON_VERSION=3.10.15` (same as development and CI) |
| Auto-Deploy | **After CI Checks Pass** - Render deploys a commit on `main` only after the GitHub Actions `flake8` + `pytest` checks succeed |

No deployment credentials are stored in the repository; Render pulls the code through its GitHub integration. See `screenshots/continuous_deployment.png`.

## Query the live API

```bash
python live_post.py
```

```
POST https://census-fastapi-ml-pipeline.onrender.com/data/
Status Code: 200
Result: >50K
```

`screenshots/live_get.png` shows the browser receiving the GET greeting from the live root URL, `screenshots/live_post.png` shows the output above, and `screenshots/example.png` shows the request example in the FastAPI docs.

## Model card

See [model_card.md](model_card.md).
