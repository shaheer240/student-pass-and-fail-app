# Model Card — Student Pass/Fail Classifier

| | |
|---|---|
| **Author** | Muhammad Shaheer (TRN-000836) |
| **Model** | scikit-learn `Pipeline`: `ColumnTransformer` (median-impute + scale numeric; most-frequent-impute + one-hot categorical) → `LogisticRegression(max_iter=1000)` |
| **File** | `models/student_pass_fail_pipeline.joblib` (trained with scikit-learn 1.9.1, `random_state=42`) |
| **Task** | Binary classification: `passed` (1 = pass, 0 = fail) |
| **Status** | Educational demo — not for production |

## Intended use
Demonstrating an end-to-end ML workflow (preprocessing, baseline comparison, evaluation, saving, serving through Streamlit) in a bootcamp setting.

**Out of scope / must not be used for:** grading, ranking, admissions, scholarships, discipline, early-warning labels attached to real students, or any decision that affects a real person.

## Data
Supplied synthetic file `student_learning.csv`: 280 students, no missing values, 44 % passed. Stratified split: 224 train / 56 test.

**Input features (known before the final result):** `hours_studied`, `attendance` (%), `practice_tests`, `sleep_hours`, `study_method` (`self` / `group` / `video`).
**Excluded:** `final_score` — it determines the label (target leakage).

## Performance (held-out test set, n = 56)

| Metric | Baseline (Dummy) | This model |
|---|---|---|
| Accuracy | 0.554 | **0.750** |
| F1 (pass class) | 0.000 | **0.720** |
| 5-fold CV accuracy / F1 (train only) | 0.563 / 0.000 | 0.822 / 0.797 |

Confusion matrix (rows = actual, cols = predicted): Fail → [24, 7]; Pass → [7, 18].
Random Forest (0.750 / 0.696) and Gradient Boosting (0.768 / 0.723) performed similarly; differences are within noise for 56 test rows. Logistic Regression was selected by cross-validated F1 on the training set.

## Limitations
- Synthetic, small dataset; no evidence of performance on real students.
- Only study-habit inputs; exam performance depends on many unmeasured factors. Errors concentrate on borderline students (13 of 14 errors within 8 marks of the pass mark).
- Metrics have wide uncertainty (one test mistake ≈ 1.8 accuracy points). Probabilities were not calibrated.
- Valid input ranges follow the training data (e.g. hours 0–10, attendance ~50–100 %); behaviour outside them is unreliable. Unknown `study_method` values are encoded as all-zeros rather than raising an error.
- Saved with scikit-learn 1.9.1; loading under another version may fail or warn — re-run the notebook to regenerate.

## Ethical considerations and risks
- **Fairness:** no demographic or sensitive attributes in the data, so subgroup performance could not be checked. Features like attendance, sleep and study method may proxy for socioeconomic or health circumstances.
- **Error costs:** false “fail” labels may stigmatize or demotivate a student; false “pass” labels may delay support. Use outputs only as a conversation starter for offering help, with human review.
- **Privacy:** real student data is personal data and needs consent, minimization and secure handling.
- **Transparency:** the Streamlit app displays the demo-only warning, the pass probability and an uncertainty note near 50 %.

## How to use
```python
import joblib, pandas as pd
pipe = joblib.load("models/student_pass_fail_pipeline.joblib")
new = pd.DataFrame([{"hours_studied": 6.0, "attendance": 80, "practice_tests": 4,
                     "sleep_hours": 7.0, "study_method": "group"}])
pipe.predict(new), pipe.predict_proba(new)[:, 1]
```
