import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
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
sns.boxplot(x=df["LotArea"])
# Thsir plot showws that there are some outliers in the LotArea column.
# We can remove them using the IQR method
Q1 = df["LotArea"].quantile(0.25)  # 25th percentile
Q3 = df["LotArea"].quantile(0.75)  # 75th percentile
IQR = Q3 - Q1  # Interquartile range
# Define lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
#Identify outliers
outliers = df[(df["LotArea"] < lower_bound) | (df["LotArea"] > upper_bound)]
print("Outliers in LotArea column:")
print(outliers)
print(outliers["LotArea"])
# Remove outliers from the dataset

plt.boxplot(df["LotArea"])
plt.title("Box Plot of LotArea")
plt.ylabel("LotArea")
plt.show()

print(df[df["LotArea"] > df["LotArea"].quantile(0.99)])  # Top 1% of values
