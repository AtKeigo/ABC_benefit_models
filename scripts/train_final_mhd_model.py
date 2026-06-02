from pathlib import Path
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score
import joblib


DOSE_FILE = "data/FB-ABC-dose.xlsx"
FEATURE_FILE = "data/FB-geo.xlsx"

OUTPUT_MODEL_PATH = "models/gb_mhd_5feat.pkl"
LABEL_COLUMN = "Heart-mean"
THRESHOLD = 0.20

SELECTED_FEATURES = ["CVR", "RLV", "LLV", "L", "TLV"]

MODEL_PARAMS = {
    "random_state": 42,
    "n_estimators": 100,
    "learning_rate": 0.05,
}


def main():
    dose_df = pd.read_excel(DOSE_FILE, engine="openpyxl")
    feature_df = pd.read_excel(FEATURE_FILE, engine="openpyxl")


    missing_features = [f for f in SELECTED_FEATURES if f not in feature_df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns in feature file: {missing_features}")

    if LABEL_COLUMN not in dose_df.columns:
        raise ValueError(f"Missing label column in dose file: {LABEL_COLUMN}")


    y = (dose_df[LABEL_COLUMN] > THRESHOLD).astype(int)

    X = feature_df[SELECTED_FEATURES].copy()

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    model = GradientBoostingClassifier(**MODEL_PARAMS)
    model.fit(X_resampled, y_resampled)


    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y, y_prob)
    acc = accuracy_score(y, y_pred)
    sen = recall_score(y, y_pred)

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0


    model_path = Path(OUTPUT_MODEL_PATH)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


    print("Final MHD model training completed.")
    print(f"Selected features: {SELECTED_FEATURES}")
    print(f"Saved model to: {model_path}")
    print("This model is compatible with predict_mhd.py")

    print("\nReference performance on the original full dataset:")
    print(f"AUC: {auc:.3f}")
    print(f"ACC: {acc:.3f}")
    print(f"SEN: {sen:.3f}")
    print(f"SPE: {spe:.3f}")


if __name__ == "__main__":
    main()
