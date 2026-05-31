import argparse
from pathlib import Path

import joblib
import pandas as pd


REQUIRED_FEATURES = ["MaxHD", "MaxLD", "AB", "TLV", "RLV", "CD"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict whether mean left lung dose (MLD) reduction >10% with ABC compared with FB."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file containing patient features(dose and anatomical).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/gb_mld_6feat.pkl",
        help="Path to the trained model file (.pkl).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output CSV file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    model_path = Path(args.model)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # 1. Read input data
    df = pd.read_csv(input_path)

    # 2. Check required features
    missing_features = [f for f in REQUIRED_FEATURES if f not in df.columns]
    if missing_features:
        raise ValueError(
            f"Missing required feature columns: {missing_features}. "
            f"Input file must contain: {REQUIRED_FEATURES}"
        )

    # 3. Select model input
    X = df[REQUIRED_FEATURES]

    # 4. Load trained model
    model = joblib.load(model_path)

    # 5. Predict probability and class
    probability = model.predict_proba(X)[:, 1]
    prediction = (probability >= 0.5).astype(int)

    # 6. Save results
    result_df = df.copy()
    result_df["probability"] = probability
    result_df["prediction"] = prediction

    result_df.to_csv(output_path, index=False)

    print(f"Prediction completed.")
    print(f"Input file: {input_path}")
    print(f"Model file: {model_path}")
    print(f"Output file: {output_path}")
    print("Output columns added: probability, prediction")
    print("prediction = 1 indicates predicted MLD reduction >10% with ABC compared with FB.")


if __name__ == "__main__":
    main()
