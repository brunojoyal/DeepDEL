import argparse
import sys

from deepdelgfn.autodock_proxy.model import load_autodock_proxy

def main():
    ap = argparse.ArgumentParser(description="Predict docking score for one SMILES using a saved ECFP model.")
    ap.add_argument(
        "--model",
        default="models/autodock_model.joblib",
        help="Path to saved model (.joblib RF or .pt NN). Default: models/autodock_model.joblib",
    )
    ap.add_argument("--smiles", required=True, help="SMILES string to predict")
    args = ap.parse_args()

    try:
        proxy = load_autodock_proxy(args.model)
    except Exception as e:
        sys.exit(f"[ERROR] Could not load proxy from {args.model}: {e}")

    try:
        pred_arr = proxy.predict_smiles([args.smiles])
        if pred_arr.size == 0:
            sys.exit(f"[ERROR] Invalid SMILES or empty prediction for: {args.smiles}")
        pred = float(pred_arr[0])
    except Exception as e:
        sys.exit(f"[ERROR] Prediction failed: {e}")

    print(f"[INFO] Using ECFP radius={proxy.radius}, n_bits={proxy.n_bits}")
    print(f"[INFO] SMILES: {args.smiles}")
    print(f"[PRED] Docking score (kcal/mol): {pred:.6f}")

if __name__ == "__main__":
    main()