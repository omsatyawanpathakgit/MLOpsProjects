import joblib
import pandas as pd


# Load model artifact
artifact = joblib.load(
    "models/Swiggy_FoodDeliveryTimePrediction_Model.pkl"
)

model = artifact["model"]
preprocessor = artifact["preprocessor"]


# Example new order
new_data = pd.DataFrame({
    'Area': ['Vastrapur'],
    'City': ['Ahmedabad'],
    'Restaurant': ['Vanilla Sky'],
    'Price': [300.0],
    'Avg ratings': [2.9],
    'Address': ['Vastrapur'],
    'Total ratings': [220]
})


# Apply same preprocessing used during training
processed_data = preprocessor.transform(new_data)


# Prediction
prediction = model.predict(processed_data)


print(
    f"Predicted Delivery Time: {prediction[0]:.2f} minutes"
)