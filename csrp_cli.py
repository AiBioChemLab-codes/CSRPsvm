#!/usr/bin/env python3
"""
CSRP Predictor Command-Line Tool
Usage: python csrp_cli.py --input sequences.fasta --output predictions.csv
"""

import os
import argparse
import torch
import pandas as pd
import joblib
from utils.fasta import fasta_to_pandas

try:
    from utils.getPureESM import ESM_Embed
    HAS_PROTTRANS = True
except ImportError:
    HAS_PROTTRANS = False


def predict_csrp(fasta_path, model_dir="./ml_model/CSRPsvm/"):
    """
    Predict cellular senescence-related proteins from a FASTA file.
    
    Parameters
    ----------
    fasta_path : str
        Path to the input FASTA file.
    model_dir : str
        Directory containing the pre-trained SVM model files.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ID, Sequence, pred_label, pred_prob.
    """
    if not HAS_PROTTRANS:
        raise ImportError("ESM_Embed module not found. Please ensure utils/getPureESM.py is available.")

    # Determine device: cuda:0 if GPU available, otherwise CPU
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Read FASTA into DataFrame (must have columns: ID, Sequence)
    seq_pd = fasta_to_pandas(fasta_path)
    if seq_pd.empty:
        raise ValueError("No sequences found in the FASTA file.")

    # Generate ESM embeddings with explicit device
    feats = ESM_Embed(
        seq_pd,
        model_type="m7",
        batch_size=8,
        max_seq_len=2000,
        device=device          # pass torch.device object directly
    )

    # Load the list of 405 feature names used by the SVM
    feat_name_path = os.path.join(model_dir, "SFfeatureName.csv")
    if not os.path.exists(feat_name_path):
        raise FileNotFoundError(f"Feature name file not found: {feat_name_path}")
    feat_name_df = pd.read_csv(feat_name_path, header=0)
    feature_cols = feat_name_df.columns.tolist()   # e.g. ['esm_F296', 'esm_F969', ...]

    # Select only the required features
    X = feats[feature_cols]

    # Load the trained scaler and SVM model
    scaler_path = os.path.join(model_dir, "SVMSF_5to500_esmt33_n2560_features_405_scaler.joblib")
    model_path = os.path.join(model_dir, "SVMSF_5to500_esmt33_n2560_features_405_best_model.joblib")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    # Standardise and predict
    X_scaled = scaler.transform(X)
    labels = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)

    # Build output DataFrame
    outpd = seq_pd[['ID', 'Sequence']].copy()
    outpd["pred_label"] = labels
    outpd["pred_prob"] = probs[ 1]   # probability of being senescence-related

    return outpd


def main():
    parser = argparse.ArgumentParser(
        description="Predict cellular senescence-related proteins from a FASTA file using ESM embeddings and an SVM classifier."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the input FASTA file containing protein sequences."
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to save the output CSV file with predictions."
    )
    parser.add_argument(
        "--model_dir", "-m",
        default="./ml_model/CSRPsvm/",
        help="Directory containing the pre-trained SVM model files (default: ./ml_model/CSRPsvm/)."
    )
    args = parser.parse_args()

    # Check input file exists
    if not os.path.isfile(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        exit(1)

    # Perform prediction
    print(f"Reading sequences from {args.input} ...")
    try:
        result_df = predict_csrp(args.input, model_dir=args.model_dir)
    except Exception as e:
        print(f"Prediction failed: {e}")
        exit(1)

    # Save results
    result_df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")
    print(f"Total sequences processed: {len(result_df)}")
    print(f"Senescence-related: {(result_df['pred_label']==1).sum()}")
    print(f"Non-senescence: {(result_df['pred_label']==0).sum()}")


if __name__ == "__main__":
    main()