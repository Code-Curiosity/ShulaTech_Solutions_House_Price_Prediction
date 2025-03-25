# House Price Prediction using Machine Learning
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the training dataset
data = pd.read_csv('train.csv')

# Keeping only essential features
essential_features = ['LotArea', 'Neighborhood', 'GrLivArea', 'BedroomAbvGr', 'SalePrice']
data = data[essential_features]

# One-hot encoding for 'Neighborhood'
data = pd.get_dummies(data, columns=['Neighborhood'], drop_first=True)

# Loading test data
test_data = pd.read_csv("test.csv")
