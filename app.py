"""
DiabetesIQ - Dual Model Prediction Server
==========================================

REQUIRED FILES in /models:
  clinical_model.pkl          — your clinical ensemble
  clinical_preprocessor.pkl   — fitted RobustScaler  (joblib.dump(transformer, ...))
  clinical_medians.pkl        — outcome-stratified medians dict
  clinical_insulin_cap.pkl    — Insulin IQR upper cap float  (joblib.dump(upper, ...))

  lifestyle_model.pkl         — your lifestyle ensemble
  lifestyle_pipeline.pkl      — (OPTIONAL) fitted sklearn Pipeline
                                 If sklearn version mismatch prevents loading,
                                 the app rebuilds the pipeline automatically.

============================================================
SAVE FROM YOUR COLAB NOTEBOOKS (add to end of each):
============================================================

--- CLINICAL ---
import joblib
joblib.dump(transformer,          "clinical_preprocessor.pkl")
joblib.dump(upper,                "clinical_insulin_cap.pkl")
medians = {}
for col in ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']:
    medians[col] = {0: df.loc[df['Outcome']==0,col].median(),
                    1: df.loc[df['Outcome']==1,col].median()}
joblib.dump(medians, "clinical_medians.pkl")
joblib.dump(ensemble_soft, "clinical_model.pkl")   # or whichever ensemble

--- LIFESTYLE ---
import joblib
joblib.dump(pipeline,      "lifestyle_pipeline.pkl")   # optional — app can rebuild
joblib.dump(ensemble_soft, "lifestyle_model.pkl")      # or whichever ensemble
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os, logging

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


# ─────────────────────────────────────────────────────────────
# Fallback constants (PIMA dataset approximations)
# ─────────────────────────────────────────────────────────────
FALLBACK_MEDIANS = {
    'Glucose':       {0: 107.0, 1: 140.0},
    'BloodPressure': {0: 70.0,  1: 74.0},
    'SkinThickness': {0: 27.0,  1: 32.0},
    'Insulin':       {0: 102.5, 1: 169.5},
    'BMI':           {0: 30.1,  1: 34.3},
}
FALLBACK_INSULIN_CAP = 196.0

FALLBACK_ROBUST = {
    'Pregnancies':              (3.0,   4.0),
    'Glucose':                  (117.0, 37.0),
    'BloodPressure':            (72.0,  18.0),
    'SkinThickness':            (29.0,  14.0),
    'Insulin':                  (125.0, 93.75),
    'BMI':                      (32.0,  9.9),
    'DiabetesPedigreeFunction': (0.37,  0.38),
    'Age':                      (29.0,  14.0),
}


def safe_load(filename):
    """Load a pkl file, return None (with warning) if missing or broken."""
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        log.warning(f"Not found: {filename}")
        return None
    try:
        obj = joblib.load(path)
        log.info(f"✓ Loaded: {filename}")
        return obj
    except Exception as e:
        log.error(f"✗ Failed to load {filename}: {e}")
        return None


# ── Load models ──────────────────────────────────────────────
clinical_model       = safe_load("clinical_model.pkl")
clinical_scaler      = safe_load("clinical_preprocessor.pkl")
clinical_medians     = safe_load("clinical_medians.pkl")
clinical_insulin_cap = safe_load("clinical_insulin_cap.pkl")

lifestyle_model      = safe_load("lifestyle_model.pkl")
lifestyle_pipeline   = safe_load("lifestyle_pipeline.pkl")   # may fail on version mismatch

'''
# ─────────────────────────────────────────────────────────────
# LIFESTYLE PIPELINE REBUILD
# If the saved pipeline can't be loaded (sklearn version mismatch),
# we rebuild it here using the SAME logic as your Colab.
# The OHE categories are fitted on synthetic data that covers all
# possible values — equivalent to what your training pipeline saw.
# ─────────────────────────────────────────────────────────────
LIFESTYLE_COLS = [
    'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness',
    'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching',
    'Irritability', 'delayed healing', 'partial paresis', 'muscle stiffness',
    'Alopecia', 'Obesity', 'Age'
]
BINARY_COLS = [c for c in LIFESTYLE_COLS if c != 'Age']

def _build_lifestyle_pipeline():
    """
    Rebuilds the exact pipeline from your Colab:
      ColumnTransformer: OHE(drop='first') on all cols except Age
      + MinMaxScaler on the full output
    Fits it on synthetic data covering all category values.
    """
    log.info("Rebuilding lifestyle pipeline from scratch (sklearn version workaround)...")
    # Synthetic training data — all binary combos covered, ages span full range
    rows = []
    for gender in ['Male', 'Female']:
        for yn in ['Yes', 'No']:
            rows.append({
                'Gender': gender,
                **{c: yn for c in BINARY_COLS if c != 'Gender'},
                'Age': 30.0
            })
    # Add age range coverage for MinMaxScaler
    for age in [1.0, 100.0]:
        rows.append({
            'Gender': 'Male',
            **{c: 'Yes' for c in BINARY_COLS if c != 'Gender'},
            'Age': age
        })

    df_fit = pd.DataFrame(rows, columns=LIFESTYLE_COLS)

    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(drop='first'), BINARY_COLS)],
        remainder='passthrough'
    )
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('scaler', MinMaxScaler())
    ])
    pipe.fit(df_fit)
    log.info("✓ Lifestyle pipeline rebuilt successfully.")
    return pipe

# Use loaded pipeline if available, else rebuild
if lifestyle_pipeline is None:
    lifestyle_pipeline = _build_lifestyle_pipeline()

'''
# ─────────────────────────────────────────────────────────────
# CLINICAL PREPROCESSING
# Mirrors your Colab exactly:
#  1. 0 → NaN for Glucose, BP, SkinThickness, Insulin, BMI
#  2. Fill NaN with outcome-stratified medians (avg of both at inference)
#  3. Cap Insulin at IQR upper bound
#  4. Engineer NewBMI, NewInsulinScore, NewGlucose
#  5. OHE with drop_first=True (same as pd.get_dummies)
#  6. RobustScale 8 numeric cols
#  7. Concat scaled + OHE → model input
# ─────────────────────────────────────────────────────────────
NUMERIC_COLS = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

def preprocess_clinical(raw: dict) -> np.ndarray:
    row = {
        'Pregnancies':              float(raw['pregnancies']),
        'Glucose':                  float(raw['glucose']),
        'BloodPressure':            float(raw['blood_pressure']),
        'SkinThickness':            float(raw['skin_thickness']),
        'Insulin':                  float(raw['insulin']),
        'BMI':                      float(raw['bmi']),
        'DiabetesPedigreeFunction': float(raw['dpf']),
        'Age':                      float(raw['age']),
    }

    # Step 1: 0 → NaN
    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        if row[col] == 0:
            row[col] = np.nan

    # Step 2: Fill NaN
    medians = clinical_medians if clinical_medians else FALLBACK_MEDIANS
    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        if np.isnan(row[col]):
            if col in medians:
                row[col] = (medians[col][0] + medians[col][1]) / 2.0

    # Step 3: Cap Insulin
    cap = float(clinical_insulin_cap) if clinical_insulin_cap is not None else FALLBACK_INSULIN_CAP
    if row['Insulin'] > cap:
        row['Insulin'] = cap

    # Step 4: Feature engineering
    bmi = row['BMI']
    if   bmi < 18.5:   new_bmi = 'Underweight'
    elif bmi <= 24.9:  new_bmi = 'Normal'
    elif bmi <= 29.9:  new_bmi = 'Overweight'
    elif bmi <= 34.9:  new_bmi = 'Obesity 1'
    elif bmi <= 39.9:  new_bmi = 'Obesity 2'
    else:              new_bmi = 'Obesity 3'

    new_insulin = 'Normal' if 16 <= row['Insulin'] <= 166 else 'Abnormal'

    g = row['Glucose']
    if   g <= 70:   new_glucose = 'Low'
    elif g <= 99:   new_glucose = 'Normal'
    elif g <= 126:  new_glucose = 'Overweight'
    else:           new_glucose = 'Secret'

    # Step 5: OHE — drop_first=True drops first alphabetical level per column
    # NewBMI ref='Normal', NewInsulinScore ref='Abnormal', NewGlucose ref='Low'
    ohe = {
        'NewBMI_Obesity 1':       int(new_bmi == 'Obesity 1'),
        'NewBMI_Obesity 2':       int(new_bmi == 'Obesity 2'),
        'NewBMI_Obesity 3':       int(new_bmi == 'Obesity 3'),
        'NewBMI_Overweight':      int(new_bmi == 'Overweight'),
        'NewBMI_Underweight':     int(new_bmi == 'Underweight'),
        'NewInsulinScore_Normal': int(new_insulin == 'Normal'),
        'NewGlucose_Low':         int(new_glucose == 'Low'),
        'NewGlucose_Normal':      int(new_glucose == 'Normal'),
        'NewGlucose_Overweight':  int(new_glucose == 'Overweight'),
        'NewGlucose_Secret':      int(new_glucose == 'Secret'),
    }

    # Step 6: RobustScale
    if clinical_scaler:
        scaled = clinical_scaler.transform([[row[c] for c in NUMERIC_COLS]])[0]
    else:
        log.warning("clinical_preprocessor.pkl not found — using fallback scaler constants")
        scaled = np.array([
            (row[col] - FALLBACK_ROBUST[col][0]) / FALLBACK_ROBUST[col][1]
            for col in NUMERIC_COLS
        ])

    # Step 7: Concat
    return np.concatenate([scaled, list(ohe.values())]).reshape(1, -1)


# ─────────────────────────────────────────────────────────────
# LIFESTYLE PREPROCESSING
# Converts 0/1 inputs back to 'Yes'/'No'/'Male'/'Female' strings,
# rebuilds the exact DataFrame format, runs through the pipeline.
# ─────────────────────────────────────────────────────────────
BIN  = {0: 'No',     1: 'Yes'}
GEN  = {0: 'Female', 1: 'Male'}

def preprocess_lifestyle(raw: dict) -> np.ndarray:
    row = {
        'Gender':             GEN[int(raw['gender'])],
        'Polyuria':           BIN[int(raw['polyuria'])],
        'Polydipsia':         BIN[int(raw['polydipsia'])],
        'sudden weight loss': BIN[int(raw['weight_loss'])],
        'weakness':           BIN[int(raw['weakness'])],
        'Polyphagia':         BIN[int(raw['polyphagia'])],
        'Genital thrush':     BIN[int(raw['genital_thrush'])],
        'visual blurring':    BIN[int(raw['visual_blurring'])],
        'Itching':            BIN[int(raw['itching'])],
        'Irritability':       BIN[int(raw['irritability'])],
        'delayed healing':    BIN[int(raw['delayed_healing'])],
        'partial paresis':    BIN[int(raw['partial_paresis'])],
        'muscle stiffness':   BIN[int(raw['muscle_stiffness'])],
        'Alopecia':           BIN[int(raw['alopecia'])],
        'Obesity':            BIN[int(raw['obesity'])],
        'Age':                float(raw['age']),
    }
    df_row = pd.DataFrame([row], columns=LIFESTYLE_COLS)
    return lifestyle_pipeline.transform(df_row)


# ─────────────────────────────────────────────────────────────
# Prediction helper
# ─────────────────────────────────────────────────────────────
def run_predict(model, X):
    pred = int(model.predict(X)[0])
    if hasattr(model, 'predict_proba'):
        prob = float(model.predict_proba(X)[0][1])
    elif hasattr(model, 'decision_function'):
        prob = float(1 / (1 + np.exp(-model.decision_function(X)[0])))
    else:
        prob = float(pred)
    return prob, pred


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/status")
def status():
    return jsonify({
        "clinical_model_loaded":     clinical_model is not None,
        "lifestyle_model_loaded":    lifestyle_model is not None,
        "clinical_scaler_loaded":    clinical_scaler is not None,
        "lifestyle_pipeline_loaded": lifestyle_pipeline is not None,
        "clinical_medians_loaded":   clinical_medians is not None,
    })


@app.route("/predict/clinical", methods=["POST"])
def predict_clinical_route():
    if clinical_model is None:
        return jsonify({"error": "clinical_model.pkl not found in /models"}), 503
    try:
        X = preprocess_clinical(request.get_json())
        prob, pred = run_predict(clinical_model, X)
        return jsonify({"probability": prob, "prediction": pred})
    except Exception as e:
        log.exception("Clinical prediction error")
        return jsonify({"error": str(e)}), 400


@app.route("/predict/lifestyle", methods=["POST"])
def predict_lifestyle_route():
    if lifestyle_model is None:
        return jsonify({"error": "lifestyle_model.pkl not found in /models"}), 503
    try:
        X = preprocess_lifestyle(request.get_json())
        prob, pred = run_predict(lifestyle_model, X)
        return jsonify({"probability": prob, "prediction": pred})
    except Exception as e:
        log.exception("Lifestyle prediction error")
        return jsonify({"error": str(e)}), 400


@app.route("/predict/combined", methods=["POST"])
def predict_combined_route():
    if clinical_model is None or lifestyle_model is None:
        return jsonify({"error": "Both models must be loaded for combined prediction"}), 503
    try:
        data = request.get_json()
        Xc = preprocess_clinical(data)
        Xl = preprocess_lifestyle(data)
        cp, cpred = run_predict(clinical_model,  Xc)
        lp, lpred = run_predict(lifestyle_model, Xl)
        ep = (cp + lp) / 2.0
        return jsonify({
            "clinical":  {"probability": cp,  "prediction": cpred},
            "lifestyle": {"probability": lp,  "prediction": lpred},
            "combined":  {"probability": ep,  "prediction": int(ep >= 0.5)},
        })
    except Exception as e:
        log.exception("Combined prediction error")
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
