# Housing Prices ML Toolkit

## Project Purpose

This project explores the Housing dataset to predict `SalePrice` values and highlight the features that drive home prices. Two companion scripts work together:

- `housing_analyzer.py` prepares the data, encodes categorical variables, and trains a baseline scikit-learn `LinearRegression` model on the Kaggle House Prices dataset.
- `feature_visualization.py` reuses the analyzer utilities to surface the most influential features and generate publication-ready figures saved to `figures/`.

## Dataset

The raw data comes from Kaggle's **House Prices: Advanced Regression Techniques** competition. Copies of the required CSVs are included under `home-data-for-ml-course/`:

- `train.csv` — labeled training data with `SalePrice`
- `test.csv` — unlabeled evaluation data
- `sample_submission.csv` and `data_description.txt` for reference

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install pandas numpy scikit-learn matplotlib seaborn  # seaborn optional but recommended
```

## Running The Analyzer

1. Activate your virtual environment.
2. From the repository root, run:
   ```bash
   python housing_analyzer.py
   ```
3. The script walks through data loading, missing-value handling, categorical encoding, and linear regression fitting. It reports core metrics (RMSE, MAE, R²) and persists helper artifacts such as model coefficients, predictions, and diagnostic plots (for example, `predictions.csv`, `house_price_model.pkl`, `actual_vs_predicted.png`).

## Visualizing Feature Importance

1. Ensure the analyzer pipeline has run at least once so the preprocessing helpers reflect the latest data.
2. Execute:
   ```bash
   python feature_visualization.py
   ```
3. The module recomputes the linear model, extracts the top three coefficients, and writes multiple charts highlighting feature influence to the `figures/` directory (bar charts, coefficient spectrum, box plots, and mean price comparisons). Install `seaborn` for enhanced categorical plots; otherwise matplotlib fallbacks are used automatically.

## Outputs

- `predictions.csv` — Predicted `SalePrice` values for the Kaggle test set.
- `house_price_model.pkl` — Serialized scikit-learn linear regression model.
- `figures/*.png` — Visual summaries of the most important features and their relationship to sales price.

## Repository Layout

```
home-data-for-ml-course/  # Kaggle dataset files
feature_visualization.py  # Visualization utilities for feature importance
housing_analyzer.py       # Data prep + linear regression modeling helpers
predictions.csv           # Sample output from the analyzer pipeline
figures/                  # Generated visualizations
```

## Next Steps

- Experiment with feature alternative models to improve predictive accuracy.
