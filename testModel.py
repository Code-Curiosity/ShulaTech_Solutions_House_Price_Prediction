import joblib
import numpy as np

# Code for model testting based on the data entered by user
model = joblib.load('house_price_model.pkl')

# Load the trained model
try:
    model = joblib.load("house_price_model.pkl")
except FileNotFoundError:
    print("Error: Model file not found! Make sure 'house_price_model.pkl' exists.")
    exit()

# Function to get valid Lot Area input
def get_valid_lot_area():
    while True:
        try:
            lot_area = float(input("📌 Enter Lot Area (500 - 15,000): "))
            if 500 <= lot_area <= 15000:
                return lot_area
            else:
                print("❌ Error: Lot Area must be between 500 and 15,000. Try again.")
        except ValueError:
            print("❌ Error: Please enter a valid numeric value.")

# Get user input
print("\n🏡 Welcome to the House Price Predictor 🔹")

lot_area = get_valid_lot_area()  # Ensures valid Lot Area
neighborhood = int(input("📌 Enter Neighborhood (1-6): "))
bedrooms = int(input("📌 Enter Number of Bedrooms: "))

# Ensure input is in the correct format (as a NumPy array)
user_input = np.array([[lot_area, neighborhood, bedrooms]])

# Make prediction (log scale)
predicted_log_price = model.predict(user_input)[0]

# Convert back to actual price (Inverse Log Transformation)
predicted_price = np.exp(predicted_log_price)

# Display the predicted price
print(f"\n✅ Predicted House Price: ${predicted_price:,.2f}")