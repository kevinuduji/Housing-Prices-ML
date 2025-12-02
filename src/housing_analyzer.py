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