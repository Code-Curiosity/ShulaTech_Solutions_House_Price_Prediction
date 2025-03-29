import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the dataset
df = pd.read_csv("train.csv")

## Data Preprocessing ##

# No need to handdle missing values as the features which we have chosen do not have any missing values.
# Thsir plot showws that there are some outliers in the LotArea column.
plt.boxplot(df["LotArea"])
plt.title("Box Plot of LotArea")
#plt.show()
# We can create boundarys for these outliers and then replace them using the IQR method
Q1 = df["LotArea"].quantile(0.25)  # 25th percentile
Q3 = df["LotArea"].quantile(0.75)  # 75th percentile
IQR = Q3 - Q1  # Interquartile range
# Define lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
#Identify outliers
#outliers = df[(df["LotArea"] < lower_bound) | (df["LotArea"] > upper_bound)]
#print("Outliers in LotArea column:")
#print(outliers)
#print(outliers["LotArea"])
# Instead of removing outliers we are applying log transformation to the LotArea column
df["LotArea_log"] = np.log1p(df["LotArea"])
# Plotting the log-transformed LotArea column
# Create a side-by-side comparison of box plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Original Box Plot
axes[0].boxplot(df["LotArea"])
axes[0].set_title("Box Plot of Original LotArea")
axes[0].set_ylabel("LotArea")

# Log-Transformed Box Plot
axes[1].boxplot(df["LotArea_log"])
axes[1].set_title("Box Plot of Log-Transformed LotArea")
axes[1].set_ylabel("Log(LotArea)")

#plt.tight_layout()
#plt.show()
# Now checking the scales of other features
df.describe()
# Identifying the need for feature scaling
(df.describe().loc[["min", "max"]]).T  # Transpose for better readability

# Applying log rransformation to the SalePrice column
df["SalePrice_log"] = np.log1p(df["SalePrice"])
# Plotting the log-transformed SalePrice column
# Create a side-by-side comparison of box plots 
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Original Box Plot
axes[0].boxplot(df["SalePrice"])
axes[0].set_title("Box Plot of Original SalePrice")
axes[0].set_ylabel("SalePrice")

# Log-Transformed Box Plot
axes[1].boxplot(df["SalePrice_log"])
axes[1].set_title("Box Plot of Log-Transformed SalePrice")
axes[1].set_ylabel("Log(SalePrice)")

#plt.tight_layout()
#plt.show()

# Performing Label Encoding on Nieghborhood column
# Initialize LabelEncoder
le = LabelEncoder()

# Fit and transform the 'Neighborhood' column
df["Neighborhood_encoded"] = le.fit_transform(df["Neighborhood"])

# Save the mapping for reference
neighborhood_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

# Save the neighborhood mapping for reference
with open("neighborhood_mapping.pkl", "wb") as f:
    pickle.dump(neighborhood_mapping, f)

# Save the encoder for later use (testing and user input)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# Drop the original column after encoding
df.drop(columns=["Neighborhood"], inplace=True)

# Applying correlation heatmap to identify correlations between features
plt.figure(figsize=(6,4))
sns.heatmap(df[['LotArea_log', 'Neighborhood_encoded', 'BedroomAbvGr', 'SalePrice_log']].corr(), annot=True, cmap='coolwarm')
plt.show()

# Select final features for training
X = df[["LotArea_log", "Neighborhood_encoded", "BedroomAbvGr", "LotArea"]]  # Keep LotArea for inverse transformation
y = df["SalePrice_log"]  # Training target is the log-transformed price

# Save preprocessed feature and target datasets
X.to_csv("X_train_pre_processed.csv", index=False)
y.to_csv("y_train_pre_processed.csv", index=False)

print("Preprocessing complete! Files saved: X_train_pre_processed.csv, y_train_pre_processed.csv, label_encoder.pkl, neighborhood_mapping.pkl")

# Final check on the prerpocessed data
# Checking for -inf or NaN values
if np.isinf(X).any().any() or np.isnan(X).any().any():
    print("There are -inf or NaN values in the preprocessed features.")
if np.isinf(y).any() or np.isnan(y).any():
    print("There are -inf or NaN values in the preprocessed target variable.")
else:
    print("No -inf or NaN values in the preprocessed data.")

## Model Training ##

# Now Loadintg the preprocessed data for training the model
# Load the preprocessed data
X = pd.read_csv("X_train_pre_processed.csv")
y = pd.read_csv("y_train_pre_processed.csv")


# Selecting the relevant features and target variable
X = df[["LotArea_log", "Neighborhood_encoded", "BedroomAbvGr"]]
Lot_Area_orginal = df["LotArea"] # Keep original LotArea for inverse transformation
y = df["SalePrice_log"]

#Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=24)
#random_state=42 ensures that the same split will be generated each time you run your code.
# This is useful for reproducibility, especially when you're trying to debug or compare results across different runs.
# The random_state parameter is a seed value for the random number generator used in the train_test_split function.
# If you set random_state to None, the function will generate a different split each time you run it.
# This can lead to different training and testing sets, which may affect the performance of your model.
# Choosing a specific random_state value allows you to control the randomness and ensure that your results are consistent across different runs.

# Creating a Linear Regression model
model = LinearRegression()
# Fitting the model to the training data
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Before Evaluation we will inverse the log transformation applied earlier
# As this will generate results based on the actual sale prices rather than their logarithmic representations.
# Hence giving accurate insights about how well our model predicts house prices.
#Inversing the log transformation for y_test and y_pred
y_pred_actual = np.expm1(y_pred)  # Convert predicted log SalePrice back to actual values
y_test_actual = np.expm1(y_test)  # Convert actual log SalePrice back to actual values

# Evaluating the model's performance on the orignal data which is not bieng transformed
mse = mean_squared_error(y_test_actual, y_pred_actual)
mae = mean_absolute_error(y_test_actual, y_pred_actual)
r2 = r2_score(y_test_actual, y_pred_actual)

print("Model Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Saving the trained model
with open("house_price_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Training complete! Model saved as house_price_model.pkl")