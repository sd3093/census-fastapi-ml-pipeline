# Model artifacts

Created by `python train_model.py`:

- `model.pkl`: trained `RandomForestClassifier`
- `encoder.pkl`: fitted `OneHotEncoder` for the categorical features
- `lb.pkl`: fitted `LabelBinarizer` for the `salary` label

`main.py` loads `model.pkl` and `encoder.pkl` at startup to serve predictions.
