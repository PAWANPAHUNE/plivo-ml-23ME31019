# Run Log

All delays are the official scorer's mean response delay at no more than 5% interrupted turns.

| Run | Data | Evaluation | AUC | Delay | Cutoff | Change and reason |
|---|---|---|---:|---:|---:|---|
| 1 | English | silence baseline | 0.514 | 1600 ms | 0.0% | Set every pause to `p_eot=1`; this establishes the supplied silence-only reference. |
| 2 | Hindi | silence baseline | 0.501 | 850 ms | 5.0% | Repeated the same baseline because Hindi has a different hold-duration distribution. |
| 3 | English | weak starter, fitted on supplied folder | 0.596 | 1190 ms | 5.0% | Added the starter's final energy, final pitch and context length to verify the pipeline. |
| 4 | Hindi | weak starter, fitted on supplied folder | 0.634 | 850 ms | 5.0% | The weak prosody model did not improve the Hindi operating point, so richer causal contours were needed. |
| 5 | English | 5-fold grouped out-of-fold | 0.626 | 1250 ms | 5.0% | Added energy decay, voicing, pitch trajectory, spectral shape, rhythm and causal turn-position features; folds split whole turns. |
| 6 | Hindi | 5-fold grouped out-of-fold | 0.742 | 802 ms | 5.0% | Used the same feature family with a Hindi-specific ensemble; long hold pauses received higher training weight because they cause false cutoffs. |
| 7 | English | pooled-language grouped out-of-fold | 0.678 | 1252 ms | 5.0% | Tested a language-neutral model; it raised AUC but did not improve the scorer, so the English-specific model was retained. |
| 8 | Hindi | pooled-language grouped out-of-fold | 0.730 | 840 ms | 5.0% | Compared pooled training with language-specific training; the Hindi-specific model was better at the operating point. |
| 9 | English | final model fitted on all supplied English pauses | 1.000 | 100 ms | 3.0% | Refit the selected ensemble on all supplied English turns. This is an in-sample deliverable score, not a hidden-test estimate. |
| 10 | Hindi | final model fitted on all supplied Hindi pauses | 1.000 | 100 ms | 1.0% | Refit on all supplied Hindi turns; the honest generalization estimate remains Run 6. |
| 11 | English | final Mac command-line reproduction | 1.000 | 100 ms | 3.0% | Regenerated all 248 predictions and verified the supplied scorer from a clean virtual environment. |
| 12 | Hindi | final Mac command-line reproduction | 1.000 | 100 ms | 1.0% | Regenerated all 248 predictions and verified the supplied scorer from the same environment. |
| 13 | English | portable-model regression check | 1.000 | 100 ms | 3.0% | Replaced scikit-learn object pickling with NumPy-only model data so inference works across scikit-learn versions without changing scores. |
| 14 | Hindi | portable-model regression check | 1.000 | 100 ms | 1.0% | Confirmed the portable inference path reproduces all 248 Hindi predictions and the same official score. |
