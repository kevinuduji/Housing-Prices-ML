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
   import seaborn as sns  # type: ignore
   HAVE_SEABORN = True
except ImportError:
   HAVE_SEABORN = False
   print("seaborn not installed; run 'pip install seaborn' for better plots.")





