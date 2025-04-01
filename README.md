# 🏡 House Price Prediction using Machine Learning

## 📌 Project Overview
This project predicts house prices using a **Linear Regression** model. The dataset includes key housing attributes, and the model estimates `SalePrice` based on selected features.

## 📂 Dataset
The dataset used in this project is stored in `custom_housing_data.csv`. Key features:

| Feature Name     | Description                                                  |
|------------------|--------------------------------------------------------------|
| **LotArea**      | Size of the lot (land area of the house).                    |
| **BedroomAbvGr** | Number of bedrooms above ground level.                       |
| **Neighborhood** | Location of the house (encoded as categorical values).       |
| **SalePrice**    | Target variable (price of the house, only in training data). |

## 🛠️ Technologies Used
- **Python** (Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn)
- **Machine Learning** (Linear Regression)
- **Joblib** (for model persistence)

## 🚀 Project Workflow
1. **Load Data**: Import dataset from `custom_housing_data.csv`.
2. **Feature Engineering**: Select relevant features.
3. **Data Preprocessing**: Handle missing values and apply encoding.
4. **Model Training**: Train a **Linear Regression** model (`trainModel_Linear_reg.py`).
5. **Save Model**: Store the trained model as `house_price_model.pkl`.
6. **Model Prediction**: Use `testModel.py` to predict house prices based on user input.

## 📥 Installation & Usage

### 🔹 Prerequisites
Ensure Python is installed along with required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
