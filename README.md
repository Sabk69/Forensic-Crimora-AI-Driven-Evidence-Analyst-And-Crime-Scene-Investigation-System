# Homicide Prediction System (Modular C++)

This project predicts five targets from your homicide dataset:

- Perpetrator Sex
- Perpetrator Age (regression)
- Perpetrator Race
- Perpetrator Ethnicity
- Weapon

It contains a Decision Tree, Random Forest, custom histogram-based XGBoost-style model, and a three-model ensemble.

## 1. Files and folders

```text
HomicideModular/
  include/       Header files
  src/           C++ source files
  data/          Optional data folder
  models/        Saved model files and learned patterns
  Makefile
  README.md
```

## 2. Put your dataset here

Copy `prepared_homicide.csv` into this folder, beside the Makefile. The program removes all five target columns from the input features. For each target, rows whose target value is `-1` are skipped during training.

## 3. Compile from Linux terminal

```bash
sudo apt install g++ make unzip
make clean
make
```

## 4. Run the interactive system

```bash
./train prepared_homicide.csv
```

The menu is:

```text
1. Train all five targets and evaluate models
2. Predict a case (manual sample or select dataset row)
3. Exit
```

### Option 1 — Train all targets

The program trains and evaluates RF, XGBoost-style model, Decision Tree, and their ensemble for all five targets. Before training, it asks:

```text
Save/update learned model patterns in TXT and JSON for future predictions? (y/n):
```

Choose `y` to save. Choose `n` to view the scores without saving the models/patterns.

For each target, the `models/` folder receives:

- `*_rf.model`, `*_xgb.model`, `*_dt.model`: complete saved model data in readable text format, including learned tree nodes, split bins, leaf values and probabilities.
- `*_patterns.txt`: human-readable tree patterns and feature index names.
- `*_patterns.json`: the same learned tree structures in JSON format.
- `*_schema.txt`: feature names/order and original numeric target class codes.

These files let option 2 load the learned models without training again.

### Option 2 — Predict a case (does NOT train)

Needs `encoding_mappings.json` beside the Makefile (or pass it as the 2nd argument: `./train prepared_homicide.csv my_mappings.json`). Everything is shown and typed in plain text, never as codes.

```text
1. Enter a new sample manually
2. Pick a random sample from the dataset
3. Back
```

- **1:** Type each value as text, e.g. `Municipal Police`, `Texas`, `Male`, `1995` (not case-sensitive). Type `r` for a random value of that column (the program prints what it picked), or `?` to list the allowed choices.
- **2:** The program picks a random dataset row by itself, shows its details first, then predicts. Target columns are not used for the prediction.
- **3:** Return to the main menu.

The sample details and predictions (names + confidence, age in years) are printed and saved to `prediction_result.txt` and `prediction_result.json`.

**Important:** Option 2 requires that option 1 has already trained and saved all five targets.

### Scores in option 1

Each model prints **training scores** and **test scores** (Accuracy and Weighted F1; MAE/RMSE/R2 for age). The per-class table is shown for the test set, with class names instead of codes.

## 5. Preprocessing rules for manual/new cases

1. Use the same category-to-number mappings used to create `prepared_homicide.csv`. Do not fit new encoders.
2. Use the same feature engineering and feature order.
3. The current CSV reader converts empty or invalid feature values to `0`. If training used median imputation, apply the same medians to manual/new values first. Zero may not be a safe missing value.
4. The feature columns must match the saved schema. Extra columns in the dataset are ignored.

## 6. Model settings

### Random Forest

Uses quantile histogram bins, weighted Gini for classification, MSE reduction for age, bootstrap sampling, and averaged tree probabilities. It uses the target-specific settings from the old RF implementation:

| Target | Trees | Depth | Min leaf | Bins | Features per split |
|---|---:|---:|---:|---:|---:|
| Sex | 40 | 20 | 2 | 32 | 8 |
| Age | 30 | 14 | 30 | 32 | 6 |
| Race | 40 | 20 | 2 | 32 | 8 |
| Ethnicity | 40 | 10 | 2 | 32 | 3 |
| Weapon | 40 | 20 | 2 | 32 | 8 |

### XGBoost-style model

Uses 32 bins, depth 4, 30 rounds, learning rate 0.1, lambda 1.0, gamma 0.0 and minimum child weight 1.0. Classification uses balanced class weights and softmax probabilities.

### Decision Tree and ensemble

The Decision Tree is one independent histogram-based tree. The ensemble averages the probabilities from RF, XGBoost and DT. For age, it averages their numeric predictions.

## 7. Fair comparison note

The modular program uses the same main algorithm settings as the supplied old implementations, but exact scores can differ because the random-number sequence and split sequence may differ. The old RF source also noted that its settings were selected using the test split; use a separate validation set for any future tuning and keep the final test set untouched.
