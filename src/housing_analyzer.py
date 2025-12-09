"""
House Price Prediction using Linear Regression
================================================
This script predicts the SalePrice of houses using Linear Regression
Identifies the top 3 most influential features.
"""


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import pickle
from matplotlib import pyplot as plt


def load_and_preprocess_data(train_path, test_path):
   print("=" * 60)
   print("STEP 1: LOADING DATA")
   print("=" * 60)
   train_df = pd.read_csv(train_path)
   test_df = pd.read_csv(test_path)
   print(f"Training data shape: {train_df.shape}")
   print(f"Test data shape: {test_df.shape}")
   print(f"\nFirst few rows of training data:")
   print(train_df.head())
   duplicates = train_df.duplicated().sum()
   print(f"\nNumber of duplicate rows in training data: {duplicates}")
   if duplicates > 0:
       train_df = train_df.drop_duplicates()
       print(f"Removed {duplicates} duplicate rows")
   return train_df, test_df


def handle_missing_values(train_df, test_df):
   print("\n" + "=" * 60)
   print("STEP 2: HANDLING MISSING VALUES")
   print("=" * 60)
   missing_train = train_df.isnull().sum()
   print(missing_train[missing_train > 0].sort_values(ascending=False).head(10))
   numeric_columns = train_df.select_dtypes(include=[np.number]).columns
   for col in numeric_columns:
       test_has_col = col in test_df.columns
       train_missing = train_df[col].isnull().sum() > 0
       test_missing = test_has_col and test_df[col].isnull().sum() > 0
       if train_missing or test_missing:
           median_val = train_df[col].median()
           train_df[col].fillna(median_val, inplace=True)
           if test_has_col:
               test_df[col].fillna(median_val, inplace=True)
   categorical_columns = train_df.select_dtypes(include=['object']).columns
   for col in categorical_columns:
       test_has_col = col in test_df.columns
       train_missing = train_df[col].isnull().sum() > 0
       test_missing = test_has_col and test_df[col].isnull().sum() > 0
       if train_missing or test_missing:
           train_df[col].fillna('None', inplace=True)
           if test_has_col:
               test_df[col].fillna('None', inplace=True)
   print(f"\nAfter handling missing values:")
   print(f"Training data missing values: {train_df.isnull().sum().sum()}")
   print(f"Test data missing values: {test_df.isnull().sum().sum()}")
   return train_df, test_df


def encode_categorical_features(train_df, test_df):
   print("\n" + "=" * 60)
   print("STEP 3: ENCODING CATEGORICAL FEATURES")
   print("=" * 60)
   categorical_columns = train_df.select_dtypes(include=['object']).columns
   categorical_columns = [col for col in categorical_columns if col not in ['Id', 'SalePrice']]
   print(f"\nNumber of categorical columns to encode: {len(categorical_columns)}")
   label_encoders = {}
   for col in categorical_columns:
       le = LabelEncoder()
       combined_values = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
       le.fit(combined_values)
       train_df[col] = le.transform(train_df[col].astype(str))
       test_df[col] = le.transform(test_df[col].astype(str))
       label_encoders[col] = le
   print(f"Encoded {len(label_encoders)} categorical columns")
   return train_df, test_df, label_encoders


def prepare_features_and_labels(train_df):
   print("\n" + "=" * 60)
   print("STEP 4: PREPARING FEATURES AND LABELS")
   print("=" * 60)
   feature_columns = [col for col in train_df.columns if col not in ['Id', 'SalePrice']]
   X_train = train_df[feature_columns]
   y_train = train_df['SalePrice']
   print(f"\nNumber of features: {len(feature_columns)}")
   print(f"Number of training samples: {len(X_train)}")
   print(f"\nTarget variable (SalePrice) statistics:")
   print(y_train.describe())
   return X_train, y_train, feature_columns
