# Start-to-finish instructions

## 1. Create the environment

Use Python 3.10 or newer on CPU.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows activation: `.venv\Scripts\activate`.

## 2. Keep this structure

```text
eot_submission/
├── predict.py
├── features.py
├── train_model.py
├── validate.py
├── eot_model.pkl
├── score.py
├── predictions_english.csv
├── predictions_hindi.csv
├── SUMMARY.html
├── RUNLOG.md
├── NOTES.md
├── ERRORS_TO_REVIEW.md
├── README.md
├── SETUP_AND_SUBMISSION.md
└── requirements.txt
```

Keep `eot_data/` outside the submitted folder unless the portal explicitly requests it.

## 3. Generate predictions

```bash
python predict.py --data_dir ../eot_data/english --out predictions_english.csv
python predict.py --data_dir ../eot_data/hindi --out predictions_hindi.csv
```

The saved inference model contains only dictionaries and NumPy arrays, so prediction is not tied to a scikit-learn pickle version. `predict.py` does not train and never reads `label` or `pause_end`.

## 4. Score both files

```bash
python score.py --data_dir ../eot_data/english --pred predictions_english.csv
python score.py --data_dir ../eot_data/hindi --pred predictions_hindi.csv
```

Expected supplied-folder results are 100 ms with 3.0% interrupted turns for English and 100 ms with 1.0% interrupted turns for Hindi. These are fitted-folder results; grouped out-of-fold estimates in `RUNLOG.md` are the hidden-test guide.

## 5. Optional rebuild and validation

```bash
python train_model.py --data_root ../eot_data --out eot_model.pkl
python validate.py --data_root ../eot_data --out_dir validation
```

Training may use `pause_end` only for sample weighting. Prediction features remain strictly causal.

## 6. Final check

Open `SUMMARY.html`, confirm `NOTES.md` remains at most 10 sentences, rerun both prediction commands, and verify that each CSV has columns `turn_id,pause_index,p_eot`. Do not include `.venv`, `__pycache__`, `.DS_Store`, or the large `eot_data` folder in the ZIP.
