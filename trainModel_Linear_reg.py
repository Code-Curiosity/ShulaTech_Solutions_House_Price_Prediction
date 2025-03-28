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

#Select relevant features for prediction
features = ['LotArea', 'BedroomAbvGr', 'Neighborhood']
target = 'SalePrice'

#Extracting relevant features and target variable
X = df[features]
y = df[target]  

# Data Preprocessing
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

# Select final features for training
X = df[["LotArea_log", "Neighborhood_encoded", "BedroomAbvGr", "LotArea"]]  # Keep LotArea for inverse transformation
y = df["SalePrice_log"]  # Training target is the log-transformed price

# Save preprocessed feature and target datasets
X.to_csv("X_train_pre_processed.csv", index=False)
y.to_csv("y_train_pre_processed.csv", index=False)

print("Preprocessing complete! Files saved: X_train_pre_processed.csv, y_train_pre_processed.csv, label_encoder.pkl, neighborhood_mapping.pkl")

