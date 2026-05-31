# ABC Benefit Prediction Models for Left-Sided Breast Cancer Radiotherapy

This repository provides the final Gradient Boosting (GB) models developed to predict the dosimetric benefit of Active Breathing Coordinator deep inspiration breath-hold (ABC-DIBH) in left-sided breast cancer radiotherapy.

Two prediction tasks are supported:
1. Prediction of mean heart dose (MHD) reduction >20% with ABC compared with free breathing (FB)
2. Prediction of mean left lung dose (MLD) reduction >10% with ABC compared with FB

The repository includes trained model files, inference scripts, example input files, and feature descriptions.

## Related manuscript

[]

## Repository structure

- `models/`
  - `gb_mhd_5feat.pkl`: final 5-feature GB model for predicting MHD reduction >20%
  - `gb_mld_6feat.pkl`: final 6-feature GB model for predicting MLD reduction >10%

- `scripts/`
  - `predict_mhd.py`: inference script for the MHD model
  - `predict_mld.py`: inference script for the MLD model

- `sample_data/`
  - `sample_input_mhd.csv`: example input file for the MHD model
  - `sample_input_mld.csv`: example input file for the MLD model

- `feature_description.md`: definitions and units of input features
- `requirements.txt`: required Python packages

## Input features

### Model 1: Prediction of MHD reduction >20%
Input features:
- CVR
- RLV
- LLV
- L
- TLV

### Model 2: Prediction of MLD reduction >10%
Input features:
- MaxHD
- MaxLD
- AB
- TLV
- RLV
- CD

## Output

Each model outputs:
- `probability`: predicted probability of achieving the predefined dosimetric benefit
- `prediction`: binary classification result
  - `1` = predicted benefit achieved
  - `0` = predicted benefit not achieved
