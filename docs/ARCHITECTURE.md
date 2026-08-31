# Architecture

```text
Raw CSV
  |
  v
src.data.loader
  |
  +--> X / y split
  |
  v
ColumnTransformer
  |-- numeric: median imputation -> StandardScaler
  `-- categorical: most-frequent imputation -> OneHotEncoder
  |
  +-------------------+-------------------+
  |                   |                   |
Logistic Regression  Random Forest      MLPClassifier
  |                   |                   |
  +-------------------+-------------------+
                      |
              Stratified 5-fold CV
                      |
              model comparison
                      |
              final holdout test
                      |
                saved pipeline
                      |
                  FastAPI
               /health /predict
```

## Design principles

1. **No leakage:** preprocessing is fitted inside each training fold through the Scikit-Learn Pipeline.
2. **Comparable models:** all three candidates consume the same transformed feature space.
3. **Reproducibility:** random seeds and split parameters are centralized.
4. **Separation of concerns:** loading, features, models, inference and API are separate modules.
5. **Human-in-the-loop:** churn predictions support retention prioritization and are not intended to automate irreversible customer decisions.
