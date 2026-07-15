# End-of-Turn Detection Submission

## Run predictions

```bash
python predict.py --data_dir /path/to/eot_data/english --out predictions_english.csv
python predict.py --data_dir /path/to/eot_data/hindi --out predictions_hindi.csv
```

`predict.py` reads only `turn_id`, `audio_file`, `pause_index`, and `pause_start`. Every feature call copies `audio[:pause_start]`; inference never reads `label`, `pause_end`, the current pause duration, or future audio.

The included `eot_model.pkl` is a portable NumPy-data model and does not unpickle scikit-learn estimators. This avoids scikit-learn-version failures during grading.

## Score the supplied folders

```bash
python score.py --data_dir /path/to/eot_data/english --pred predictions_english.csv
python score.py --data_dir /path/to/eot_data/hindi --pred predictions_hindi.csv
```

## Optional model rebuild

```bash
python train_model.py --data_root /path/to/eot_data --out eot_model.pkl
```

The rebuild path uses scikit-learn 1.8.0 and exports the same portable model format.
