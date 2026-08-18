#Running Test-Cases on the Regression Model (Swiggy Delivery Time Prediction) using Github Actions

# Import required modules
import joblib
import pandas as pd
import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

#Ignore all incoming warnings (if any):
import warnings
warnings.filterwarnings("ignore")

def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def get_chrome_options():
    options = Options()

    options.add_argument("--headless=new")

    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    return options



def load_model(model_path):
    model = joblib.load(model_path)

    logging.info("Model loaded successfully.")

    return model


def prepare_input_data():
    new_data = pd.DataFrame({
        "Area": ["Vastrapur"],
        "City": ["Ahmedabad"],
        "Restaurant": ["Vanilla Sky"],
        "Price": [300.0],
        "Avg ratings": [2.9],
        "Address": ["Vastrapur"],
        "Total ratings": [220]
    })

    return new_data

    
def predict_delivery_time(model, new_data):
    prediction = model.predict(new_data)

    predicted_time = round(prediction[0])

    return predicted_time


def display_prediction(predicted_time):
    print(
        "Predicted Delivery Time:",
        predicted_time,
        "minutes"
    )

    logging.info(
        "Delivery time predicted = %s minutes",
        predicted_time
    )


def main():

    # Configure logging
    configure_logging()

    try:
        # Load model
        model = load_model(
            "Swiggy_FoodDeliveryTimePrediction_Model.pkl"
        )

        # Prepare input
        new_data = prepare_input_data()

        # Predict
        predicted_time = predict_delivery_time(
            model,
            new_data
        )

        # Display result
        display_prediction(predicted_time)

    finally:
        logging.info("Testing completes.")


if __name__ == "__main__":
    main()