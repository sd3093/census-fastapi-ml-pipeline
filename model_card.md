# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

- **Developer:** Sohbet Dovranov, for the Udacity project "Deploying a Machine Learning Model with FastAPI".
- **Date / version:** November 2026, version 1.0.
- **Model type:** A scikit-learn (1.5.1) `RandomForestClassifier` with 60 trees, `max_depth=12`, `min_samples_leaf=5` and `random_state=42`. It is a binary classifier. The forest is deliberately kept compact (about 2.3 MB pickled) so the trained model can be committed to the repository and served directly.
- **Inputs:** 14 census attributes. Six are continuous: age, fnlgt, education-num, capital-gain, capital-loss and hours-per-week. Eight are categorical: workclass, education, marital-status, occupation, relationship, race, sex and native-country. The categorical features are one-hot encoded with `OneHotEncoder(handle_unknown="ignore")`, and the label is binarized with `LabelBinarizer`.
- **Output:** a salary class, `<=50K` or `>50K`.
- **Artifacts:** `model/model.pkl` (the classifier), `model/encoder.pkl` (the one-hot encoder) and `model/lb.pkl` (the label binarizer). They are produced by `train_model.py` and served through the FastAPI app in `main.py`.

## Intended Use

The model predicts whether a person's annual income is above $50K from basic demographic and employment attributes. It is meant for education and demonstration: showing how to train, test, version and serve an ML model behind a REST API with CI. It is **not** intended for real decisions about individuals, such as hiring, lending, insurance, housing or eligibility for services.

## Training Data

The data is the UCI "Census Income" (Adult) dataset, from the 1994 US Census, provided as `data/census.csv`. It has 32,561 rows, 14 features and the `salary` label. About 75.9% of rows are `<=50K` and 24.1% are `>50K`. The data was split 80/20 with `train_test_split(test_size=0.20, random_state=42)`, stratified on `salary`. The training set therefore has 26,048 rows. The encoders were fit on the training split only.

## Evaluation Data

The evaluation set is the held-out 20% test split: 6,513 rows with the same class balance as the full dataset. It was processed with the encoder and label binarizer fitted on the training data. Unknown categories are ignored by the encoder. Performance on slices of every categorical feature was also computed on this split; the output is in `slice_output.txt`.

## Metrics

_Please include the metrics used and your model's performance on those metrics._

The model is evaluated with **precision**, **recall** and **F1** (F-beta with beta = 1), where the positive class is `>50K`.

| Metric | Test set (6,513 rows) |
|---|---|
| Precision | 0.8018 |
| Recall | 0.5599 |
| F1 | 0.6594 |

Selected slice results from `slice_output.txt`:

| Slice | Count | Precision | Recall | F1 |
|---|---|---|---|---|
| sex = Male | 4,355 | 0.7874 | 0.5767 | 0.6658 |
| sex = Female | 2,158 | 0.9127 | 0.4694 | 0.6199 |
| race = White | 5,533 | 0.8067 | 0.5652 | 0.6647 |
| race = Black | 662 | 0.8367 | 0.4409 | 0.5775 |
| race = Asian-Pac-Islander | 200 | 0.6471 | 0.6346 | 0.6408 |
| relationship = Own-child | 1,032 | 1.0000 | 0.1333 | 0.2353 |

## Ethical Considerations

The dataset contains sensitive attributes: race, sex, marital status and native country. It reflects income patterns from 1994, including historical inequalities. The slice analysis shows uneven performance across groups. Recall is lower for women (0.47) than for men (0.58), and lower for Black individuals (0.44) than for White individuals (0.57). This means the model misses high earners in those groups more often. Very small groups, such as `race = Other` (45 rows) or `workclass = Never-worked` (1 row), give unstable metrics that should not be over-interpreted. Using this model to make or support decisions about people could reinforce existing biases.

## Caveats and Recommendations

- The data is over 30 years old and US-specific, so the income threshold and feature relationships do not represent today's population. Retrain on current, representative data before any real use.
- Recall (0.56) is noticeably lower than precision (0.80). If missing high earners is costly, consider class weighting, threshold tuning or hyperparameter search (for example K-fold cross-validation with grid search).
- Monitor slice metrics, not only overall metrics, and investigate groups with low recall or few samples.
- Removing the sensitive features alone does not remove bias, because proxies such as relationship or occupation remain. Apply proper fairness assessment and mitigation if the model is extended.
