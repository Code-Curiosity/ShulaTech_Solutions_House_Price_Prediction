import pandas as pd
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the dataset
try:
    df = pd.read_csv("custom_housing_data.csv")
except FileNotFoundError:
    print("Error: Dataset not found. Check the file path.")
    exit()

#Displaying basic information about the DataSet
print(df.info())
print(df.head())

## Data Preprocessing ##
# Check for missing values
print(df.isnull().sum())

# Checking for outliers
# Features to check for outliers
features = ["LotArea", "BedroomAbvGr", "SalePrice"]

plt.figure(figsize=(12, 6))
for i, feature in enumerate(features, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(y=df[feature])
    plt.title(f"Boxplot of {feature}")

plt.tight_layout()
#plt.show()
# We have no missing values and we have some outliers which are processed using log transformation
# Log transformation
df['SalePrice_log'] = np.log1p(df['SalePrice'])  

# Plot the before and after transformation
plt.figure(figsize=(12, 5))

# Original SalePrice boxplot
plt.subplot(1, 2, 1)
sns.boxplot(y=df['SalePrice'])
plt.title("Before Log Transformation")
plt.ylabel("SalePrice")
plt.xticks([]) 
# Transformed SalePrice boxplot
plt.subplot(1, 2, 2)
sns.boxplot(y=df['SalePrice_log'])
plt.title("After Log Transformation")
plt.ylabel("Log-transformed SalePrice")

plt.tight_layout()
#plt.show()
## Model Training ##
# Define feature columns and target
features = ['LotArea', 'Neighborhood_encoded', 'BedroomAbvGr']
target = 'SalePrice_log'

# Splitting data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=27)

#Initialize and Train a Linear Regression model

model = LinearRegression()

model.fit(X_train, y_train)

#Evaluating the model on Test set
# Make predictions
y_pred_log = model.predict(X_test)  # Predictions in log scale

# Inverse the log transformation
y_pred_actual = np.exp(y_pred_log)  # Convert back to actual price

# Now compare with the real SalePrice
mse = mean_squared_error(np.exp(y_test), y_pred_actual)  # y_test was log-transformed
mae = mean_absolute_error(np.exp(y_test), y_pred_actual)
r2 = r2_score(np.exp(y_test), y_pred_actual)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R² Score:", r2)

# Saving the trained model
joblib.dump(model, "house_price_model.pkl")

print("Model trained successfully! and saved as house_price_model.pkl")