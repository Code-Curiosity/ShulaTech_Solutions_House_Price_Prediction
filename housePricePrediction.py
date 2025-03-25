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

# Checking for missing values
miss_values = data.isnull().sum()
print(miss_values)
# If there are any missing values then 
data.dropna(inplace=True)

# Removing sales price from the features, as we have to predict it.
X = data.drop('SalePrice', axis=1)
y = data['SalePrice'] # Target variable which we have to predict

# Loading test data
test_data = pd.read_csv("test.csv")

# Keeping only essential features (excluding "SalePrice" as it is not present in test data)
test_essential_features = ['LotArea', 'Neighborhood', 'GrLivArea', 'BedroomAbvGr']
test_data = test_data[test_essential_features]

# Preprocessing test data
#Handling missing values and one-hot encoding for 'Neighborhood' feature.
test_data.fillna(0, inplace=True)  # Example handling missing values
test_data = pd.get_dummies(test_data, columns=['Neighborhood'], drop_first=True)  # Encoding

# Ensuring both datasets have same features
missing_cols = set(data.columns) - set(test_data.columns) # Here data.columns is the training data columns
for col in missing_cols:
    test_data[col] = 0  # Add missing columns with default value

test_data = test_data[data.columns]  # Reorder columns to match training set

# Training the model
model = LinearRegression()
model.fit(X, y)

# Making predictions on test data
predictions = model.predict(test_data)

# Saving the predicted values to csv file
save = pd.DataFrame({"Predicted Sale Price": predictions})
save.to_csv("house_price_predictions.csv", index=False)


