# Model Evaluation Summary

Data: `student_learning.csv` (280 synthetic students). Stratified 80/20 split, `random_state=42` -> 224 train / 56 test. All models were scored on the **same** held-out test set.

**Target:** `passed`.  **Features:** hours_studied, attendance, practice_tests, sleep_hours, study_method.  **Excluded:** `final_score` (target leakage).

## Metrics

| Model | Test Accuracy | Test F1 | CV Accuracy (train) | CV F1 (train) |
|---|---|---|---|---|
| Baseline (Dummy, most_frequent) | 0.554 | 0.000 | 0.563 | 0.000 |
| Logistic Regression | 0.750 | 0.720 | 0.822 | 0.797 |
| Random Forest | 0.750 | 0.696 | 0.795 | 0.774 |
| Gradient Boosting | 0.768 | 0.723 | 0.800 | 0.778 |

**Selected model:** Logistic Regression (highest cross-validated F1 on training data; chosen without looking at the test set).

## Confusion matrices (test set, rows = actual)

### Baseline (Dummy, most_frequent)
| | Pred Fail | Pred Pass |
|---|---|---|
| **Actual Fail** | 31 | 0 |
| **Actual Pass** | 25 | 0 |

### Logistic Regression
| | Pred Fail | Pred Pass |
|---|---|---|
| **Actual Fail** | 24 | 7 |
| **Actual Pass** | 7 | 18 |

### Random Forest
| | Pred Fail | Pred Pass |
|---|---|---|
| **Actual Fail** | 26 | 5 |
| **Actual Pass** | 9 | 16 |

### Gradient Boosting
| | Pred Fail | Pred Pass |
|---|---|---|
| **Actual Fail** | 26 | 5 |
| **Actual Pass** | 8 | 17 |

## Error analysis

The selected model misclassified 14 of 56 test students. Errors are mostly borderline students whose predicted pass probability is near 0.5 and whose final scores lie near the pass mark. Study-habit features explain only part of exam performance, so some errors are unavoidable. With a 56-student test set, differences of a few points between models are within noise.

## Caveats

- Synthetic data; results do not transfer to real students.
- Small test set -> wide uncertainty on every metric.
- Predictions are for demonstration only and must not be used to judge or penalize real students.