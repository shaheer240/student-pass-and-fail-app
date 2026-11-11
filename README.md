# Student Pass/Fail Prediction System

**Student:** Muhammad Shaheer | **Training ID:** TRN-000836 | **Module 03 – Machine Learning (Beginner ML Project)**

An end-to-end tabular ML workflow that predicts whether a student passes (`passed = 1`) or fails (`passed = 0`) from study-habit features, with a saved scikit-learn pipeline and a Streamlit prediction app.

> ⚠️ **Synthetic classroom data.** Predictions are for demonstration only and must **not** be used to judge, grade or penalize real students.

## Project contents

| File | Purpose |
|---|---|
| `student_pass_fail_project.ipynb` | Complete runnable notebook (Jupyter / Colab) with outputs already included |
| `models/student_pass_fail_pipeline.joblib` | Saved fitted pipeline (preprocessing + Logistic Regression) |
| `outputs/model_evaluation_summary.md` | Metrics, confusion matrices, error analysis |
| `outputs/*.png / *.csv / metrics.json` | Confusion-matrix plot, feature importance, model comparison, predictions for new students |
| `app.py` | Streamlit prediction app |
| `MODEL_CARD.md` | Model card (intended use, limits, ethics) |
| `data/student_learning.csv` | Supplied dataset (280 rows) |
| `data/new_student_records.csv` | 8 new records used to demo prediction with the reloaded pipeline (see note below) |
| `requirements.txt` | Dependencies |

## How to run

```bash
pip install -r requirements.txt
jupyter notebook student_pass_fail_project.ipynb   # Run All -> re-creates model + outputs
streamlit run app.py                               # opens the prediction form in the browser
```
**Google Colab:** upload the project folder (at least `data/`), open the notebook, *Runtime → Run all*. The notebook retrains and re-saves the pipeline, so it works with whatever scikit-learn version Colab has. If `app.py` complains about loading the `.joblib` file, you are on a different scikit-learn version than the one used to save it (1.9.1) — re-run the notebook to regenerate the file.

## Approach

1. **Inspect** – 280 rows, no missing values, target 56 % fail / 44 % pass.
2. **Features & leakage** – target is `passed`. `final_score` is **excluded**: it is the exam result the label comes from (a rule `final_score ≥ 62` reproduces the label 94 % of the time), so using it would be target leakage. Features: `hours_studied`, `attendance`, `practice_tests`, `sleep_hours` (numeric) and `study_method` (categorical).
3. **Split** – stratified 80/20, `random_state=42` → 224 train / 56 test. The test set is used only for the final comparison.
4. **Pipelines** – `ColumnTransformer` inside a `Pipeline`: numeric → median imputer + `StandardScaler`; categorical → most-frequent imputer + `OneHotEncoder(handle_unknown="ignore")`. Everything is fitted on training data only.
5. **Models** – `DummyClassifier` (baseline), `LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`. The final model was chosen by 5-fold CV F1 **on the training set** (not by peeking at the test set).
6. **Save & reload** – whole pipeline saved with `joblib`, reloaded, verified to give identical predictions, then used on `new_student_records.csv` without retraining (including one record with a missing value).
7. **App** – Streamlit form (single student) plus optional CSV batch prediction; shows predicted class, pass probability and a demo-only warning.

## Results (same held-out test set, 56 students)

| Model | Test Accuracy | Test F1 | CV Accuracy | CV F1 |
|---|---|---|---|---|
| Baseline (Dummy) | 0.554 | 0.000 | 0.563 | 0.000 |
| **Logistic Regression (selected)** | 0.750 | 0.720 | 0.822 | 0.797 |
| Random Forest | 0.750 | 0.696 | 0.795 | 0.774 |
| Gradient Boosting | 0.768 | 0.723 | 0.800 | 0.778 |

Selected model confusion matrix (rows = actual): Fail → 24 correct, 7 wrongly predicted Pass; Pass → 18 correct, 7 wrongly predicted Fail.

All three real models clearly beat the baseline (which predicts “fail” for everyone and therefore has F1 = 0). They are **close to each other**: with only 56 test students, one extra mistake changes accuracy by ~1.8 points, so the ranking between them is not statistically reliable. See `outputs/model_evaluation_summary.md` for full details.

**Error analysis:** 14/56 test students were misclassified. 13 of those 14 have final scores within 8 marks of the pass mark, versus 26 % of correctly classified students, and 12 of 14 had a predicted pass probability between 25 % and 75 %. In other words, the model mostly errs on borderline students whose exam outcome cannot be inferred from study habits alone. `hours_studied` is the most important feature.

## Limitations

- **Synthetic, small data** (280 rows, 56 test rows): metrics are noisy and will not transfer to real students.
- **Only 5 features.** Prior knowledge, exam difficulty, health, home environment, etc. are not captured, which caps achievable accuracy (~75 % on the test set).
- Test accuracy (0.75) is lower than CV accuracy (0.82) — consistent with test-set noise, but also a reminder not to over-trust any single number.
- No probability calibration check; probabilities are rough.
- Pass/fail is a threshold on a score, so a classifier treats a 61 and a 62 as completely different outcomes.
- `new_student_records.csv` was **not** included in the supplied files, so I created 8 illustrative records following the dataset's feature format (one with a blank `sleep_hours`). Replace it with the official file if you have one — the notebook reads it from `data/`.
- Streamlit was developed against the saved pipeline; the app's prediction logic was exercised offline with a stubbed `streamlit` module, and the same pipeline call is exercised in the notebook.

## Responsible AI considerations

- **Intended use:** classroom demonstration of an ML workflow. **Not** for grading, ranking, selection, discipline or any decision about real people.
- **Leakage avoided:** `final_score` is excluded and a guard assertion prevents it from entering `X`.
- **Fairness:** the data has no demographic columns, so subgroup fairness could not be audited. Features such as `study_method`, `attendance` and `sleep_hours` can reflect a student's circumstances (work, health, access to resources) rather than effort — a real deployment could unfairly penalize disadvantaged students.
- **Harm of errors:** false “fail” predictions could label capable students (7 of 25 passing test students here); false “pass” predictions could hide students who need help. Any use should keep a human in the loop and focus on *offering support*, never on punishment.
- **Transparency:** the app states that it is a demo, shows probabilities rather than only a hard label, and flags uncertain cases.
- **Privacy:** real student records would be personal data; collect and use them only with consent and appropriate protection.

See `MODEL_CARD.md` for the short model card.
