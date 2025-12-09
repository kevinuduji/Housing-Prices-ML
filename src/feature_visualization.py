"""
Feature Visualization for House Prices (Linear Regression)
Outputs are saved in the 'figures/' folder.
"""
from __future__ import annotations
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Try to import seaborn for better plots
try:
   import seaborn as sns
   HAVE_SEABORN = True
except ImportError:
   HAVE_SEABORN = False
   print("seaborn not installed; run 'pip install seaborn' for better plots.")


from housing_analyzer import (
   load_and_preprocess_data,
   handle_missing_values,
   encode_categorical_features,
   prepare_features_and_labels,
   train_linear_regression,
   find_top_features,
)


TRAIN_PATH = "home-data-for-ml-course/train.csv"
TEST_PATH = "home-data-for-ml-course/test.csv"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


def load_raw_data_for_plots(path: str) -> pd.DataFrame:
   # Load raw training data for plots
   return pd.read_csv(path)


def compute_linear_regression_importance():
   # Run pipeline to get linear regression coefficients and importance
   train_df, test_df = load_and_preprocess_data(TRAIN_PATH, TEST_PATH)
   train_df, test_df = handle_missing_values(train_df, test_df)
   train_df_enc, test_df_enc, _ = encode_categorical_features(train_df.copy(), test_df.copy())
   X_train, y_train, feature_names = prepare_features_and_labels(train_df_enc)
   model = train_linear_regression(X_train, y_train)
   importance_df = pd.DataFrame({
       'Feature': feature_names,
       'Coefficient': model.coef_,
       'AbsCoefficient': np.abs(model.coef_),
   }).sort_values('AbsCoefficient', ascending=False).reset_index(drop=True)
   top3_df = find_top_features(model, feature_names, top_n=3)
   top3_list = top3_df['Feature'].tolist()
   return importance_df, top3_df, top3_list, train_df


def plot_feature_comparison(importance_df: pd.DataFrame, top_features: list[str], top_n: int = 20):
   # Bar chart of top features by absolute coefficient
   subset = importance_df.head(top_n).copy()
   missing = [f for f in top_features if f not in subset['Feature'].tolist()]
   if missing:
       subset = pd.concat([subset, importance_df[importance_df['Feature'].isin(missing)]], ignore_index=True)
   subset = subset.sort_values('AbsCoefficient', ascending=False)
   colors = ['red' if f in top_features else 'steelblue' for f in subset['Feature']]
   plt.figure(figsize=(10, 6))
   plt.barh(subset['Feature'][::-1], subset['AbsCoefficient'][::-1], color=colors[::-1])
   plt.title(f'Top {top_n} Feature Importance')
   plt.xlabel('Absolute Coefficient')
   plt.ylabel('Feature')
   plt.tight_layout()
   out = os.path.join(FIG_DIR, 'feature_comparison_bar.png')
   plt.savefig(out, dpi=150)
   plt.close()
   print(f"Saved Feature Comparison Bar Chart -> {out}")


def plot_top3_detailed(top3_df: pd.DataFrame):
   # Bar chart for top 3 feature coefficients (signed)
   df_top3 = top3_df.copy()
   colors = ['green' if v > 0 else 'red' for v in df_top3['Coefficient']]
   plt.figure(figsize=(6, 4))
   plt.bar(df_top3['Feature'], df_top3['Coefficient'], color=colors)
   plt.axhline(0, color='black', linewidth=1)
   plt.title('Top 3 Features Coefficient Impact')
   plt.ylabel('Coefficient')
   for x, y in zip(df_top3['Feature'], df_top3['Coefficient']):
       plt.text(x, y, f"{y:,.0f}", ha='center', va='bottom' if y > 0 else 'top')
   plt.tight_layout()
   out = os.path.join(FIG_DIR, 'top3_detailed_impact.png')
   plt.savefig(out, dpi=150)
   plt.close()
   print(f"Saved Top 3 Detailed Impact -> {out}")


def plot_boxplots(df: pd.DataFrame, top_features: list[str]):
   # Box plots for SalePrice by top features
   for feature in top_features:
       if feature not in df.columns:
           continue
       plt.figure(figsize=(6, 4))
       if HAVE_SEABORN:
           sns.boxplot(data=df, x=feature, y='SalePrice')
       else:
           cats = df[feature].astype(str)
           cat_codes = pd.factorize(cats)[0]
           plt.scatter(cat_codes + np.random.uniform(-0.15, 0.15, size=len(cat_codes)), df['SalePrice'], alpha=0.5)
           plt.xticks(range(len(np.unique(cats))), np.unique(cats))
       plt.title(f'SalePrice Distribution by {feature}')
       plt.xlabel(feature)
       plt.ylabel('SalePrice')
       plt.tight_layout()
       out = os.path.join(FIG_DIR, f'boxplot_{feature}.png')
       plt.savefig(out, dpi=150)
       plt.close()
       print(f"Saved Box Plot for {feature} -> {out}")


def plot_mean_price_by_category(df: pd.DataFrame, top_features: list[str]):
   # Bar chart of mean SalePrice by category for top features
   for feature in top_features:
       if feature not in df.columns:
           continue
       mean_vals = df.groupby(feature)['SalePrice'].mean().sort_values(ascending=False)
       plt.figure(figsize=(6, 4))
       plt.bar(mean_vals.index.astype(str), mean_vals.values, color='steelblue')
       plt.title(f'Mean SalePrice by {feature}')
       plt.xticks(rotation=30, ha='right')
       plt.ylabel('Mean SalePrice')
       for x, y in zip(mean_vals.index.astype(str), mean_vals.values):
           plt.text(x, y, f"{y:,.0f}", ha='center', va='bottom', rotation=90)
       plt.tight_layout()
       out = os.path.join(FIG_DIR, f'mean_price_{feature}.png')
       plt.savefig(out, dpi=150)
       plt.close()
       print(f"Saved Mean Price Bar for {feature} -> {out}")


def plot_coefficient_spectrum(importance_df: pd.DataFrame, top_features: list[str]):
   # Scatter plot of all coefficients, highlight top features
   plt.figure(figsize=(12, 6))
   x = np.arange(len(importance_df))
   plt.scatter(x, importance_df['AbsCoefficient'], s=30, color='gray', alpha=0.6, label='Other Features')
   for feature in top_features:
       row = importance_df[importance_df['Feature'] == feature]
       if not row.empty:
           idx = row.index[0]
           plt.scatter(idx, row['AbsCoefficient'].iloc[0], s=120, color='red', edgecolors='black', label=feature)
           plt.text(idx, row['AbsCoefficient'].iloc[0], feature, fontsize=9, ha='center', va='bottom')
   plt.title('Coefficient Spectrum')
   plt.xlabel('Feature Index (sorted by importance)')
   plt.ylabel('Absolute Coefficient')
   handles, labels = plt.gca().get_legend_handles_labels()
   unique = dict(zip(labels, handles))
   plt.legend(unique.values(), unique.keys(), fontsize=8)
   plt.tight_layout()
   out = os.path.join(FIG_DIR, 'coefficient_spectrum.png')
   plt.savefig(out, dpi=150)
   plt.close()
   print(f"Saved Coefficient Spectrum -> {out}")


def main():
   print("=== Feature Visualization Pipeline (Linear Regression Aligned) ===")
   importance_df, top3_df, top3_list, raw_train_df = compute_linear_regression_importance()
   print("Top 3 Features:", top3_list)


   # Fill missing values for top features (categorical)
   for f in top3_list:
       if f in raw_train_df.columns and raw_train_df[f].dtype == object:
           raw_train_df[f] = raw_train_df[f].fillna('None')


   print("Generating Feature Comparison Bar Chart...")
   plot_feature_comparison(importance_df, top_features=top3_list, top_n=20)


   print("Generating Top 3 Impact plot...")
   plot_top3_detailed(top3_df)


   print("Generating Box Plots...")
   plot_boxplots(raw_train_df, top_features=top3_list)


   print("Generating Mean Price by Category plots...")
   plot_mean_price_by_category(raw_train_df, top_features=top3_list)


   print("Generating Coefficient Spectrum plot...")
   plot_coefficient_spectrum(importance_df, top_features=top3_list)


   print("All figures saved in 'figures/'.")
   print("Done.")


if __name__ == '__main__':
   main()
