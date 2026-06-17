# CSRPsvm

CSRPsvm is a command-line predictor for cellular senescence-related proteins. It extracts ESM2 protein embeddings from FASTA sequences and applies a pre-trained SVM classifier to report prediction labels and probabilities.

## Repository Contents

- `csrp_cli.py`: command-line entry point for prediction
- `data/`: example training and test FASTA files
- `ml_model/CSRPsvm/`: pre-trained scaler, SVM model, and selected feature names
- `utils/`: FASTA parsing, ESM embedding, and feature-selection utilities

## Requirements

Python 3.9 or newer is recommended.

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

The embedding step uses the ESM2 model `esm2_t33_650M_UR50D`. Place the local Hugging Face model files in:

```text
utils/ESM2/esm2_t33_650M_UR50D/
```

The code first tries to load the model from this local path. A CUDA-capable GPU is recommended for faster embedding, but CPU execution is also supported.

## Input Format

Prepare a protein FASTA file:

```text
>protein_1
MSEQUENCE...
>protein_2
MSEQUENCE...
```

## Usage

Run prediction from the repository root:

```bash
python csrp_cli.py --input data/Repi30PN.Test.fa --output predictions.csv
```

You can also provide a custom model directory:

```bash
python csrp_cli.py \
  --input path/to/sequences.fasta \
  --output predictions.csv \
  --model_dir ml_model/CSRPsvm/
```

## Output

The output CSV contains:

- `ID`: FASTA sequence identifier
- `Sequence`: protein sequence
- `pred_label`: predicted class, where `1` indicates senescence-related and `0` indicates non-senescence
- `pred_prob`: predicted probability for the senescence-related class

## Notes

- The bundled SVM model expects the selected ESM feature columns listed in `ml_model/CSRPsvm/SFfeatureName.csv`.
- Long sequences and large FASTA files can require substantial memory during embedding.
- If model loading fails, verify that the ESM2 model directory exists and contains the required Hugging Face files.
