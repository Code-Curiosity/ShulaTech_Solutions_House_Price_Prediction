import joblib
import numpy as np

# Code for model testting based on the data entered by user
model = joblib.load('house_price_model.pkl')

def predict_house_price():
    print("\n🔹 Welcome to the House Price Predictor 🔹\n")

    # Taking input from the user
    try:
        lot_area = float(input("📌 Enter Lot Area: "))
        neighborhood_encoded = int(input("📌 Enter Neighborhood (1-6): "))  # Mapping A-F to 1-6
        bedrooms = int(input("📌 Enter Number of Bedrooms: "))

        # Create input array
        user_data = np.array([[lot_area, neighborhood_encoded, bedrooms]])

        # Predict log sale price
        predicted_log_price = model.predict(user_data)

        # Convert back to actual sale price
        predicted_price = np.exp(predicted_log_price)[0]  # Convert log price back to normal price

        print(f"\n✅ Predicted House Price: ${predicted_price:,.2f}\n")

    except ValueError:
        print("\n⚠️ Invalid input! Please enter numerical values.")

# Run the function
if __name__ == "__main__":
    predict_house_price()