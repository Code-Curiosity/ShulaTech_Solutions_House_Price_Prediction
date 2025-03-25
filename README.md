# House Price Prediction using Machine Learning

## 📌 Project Overview

This project aims to predict house prices based on essential features using a **Linear Regression** model. The dataset includes various house characteristics, and the model is trained to estimate the `SalePrice` based on selected features.

## 📂 Dataset

The dataset consists of training and test data in CSV format. The essential features used for prediction are:

| Feature Name     | Description                                                  |
| ---------------- | ------------------------------------------------------------ |
| **LotArea**      | Size of the lot (land area of the house).                    |
| **GrLivArea**    | Above-ground living area in square feet.                     |
| **BedroomAbvGr** | Number of bedrooms above ground level.                       |
| **Neighborhood** | Location of the house (converted into one-hot encoding).     |
| **SalePrice**    | Target variable (price of the house, only in training data). |

## 🛠️ Technologies Used

- **Python** (Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn)
- **Machine Learning** (Linear Regression)

## 🚀 Project Workflow

1. **Load Data**: Import training and test datasets.
2. **Feature Selection**: Keep only essential features.
3. **Data Preprocessing**: Handle missing values and apply one-hot encoding to categorical features.
4. **Train Model**: Fit a Linear Regression model on the training dataset.
5. **Make Predictions**: Predict house prices on the test dataset.
6. **Evaluate Model**: Measure performance using metrics like Mean Squared Error (MSE) and R-squared (R²).

## 📥 Installation & Usage

### 🔹 Prerequisites

Ensure you have Python installed along with the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 🔹 Running the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/Code-Curiosity/ShulaTech_Solutions_House_Price_Prediction.git
   ```
2. Run the script:
   ```bash
   python housePricePrediction.py
   ```

## 📌 Next Steps

- Improve model performance by adding more features.
- Experiment with different regression algorithms.
- Optimize hyperparameters for better accuracy.

---

💡 **Contributions are welcome!** Feel free to fork and improve the project. 😊

