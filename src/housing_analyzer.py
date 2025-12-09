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
   """
   Load and preprocess the training and test data.
  
   Parameters:
   -----------
   train_path : str
       Path to the training CSV file
   test_path : str
       Path to the test CSV file
  
   Returns:
   --------
   tuple : (train_df, test_df)
   """
   print("=" * 60)
   print("STEP 1: LOADING DATA")
   print("=" * 60)
  
   # Load the data
   train_df = pd.read_csv(train_path)
   test_df = pd.read_csv(test_path)
  
   print(f"Training data shape: {train_df.shape}")
   print(f"Test data shape: {test_df.shape}")
   print(f"\nFirst few rows of training data:")
   print(train_df.head())
  
   # Check for duplicates
   duplicates = train_df.duplicated().sum()
   print(f"\nNumber of duplicate rows in training data: {duplicates}")
   if duplicates > 0:
       train_df = train_df.drop_duplicates()
       print(f"Removed {duplicates} duplicate rows")
  
   return train_df, test_df
