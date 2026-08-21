import joblib

from zenml import step


@step
def save_model(
    model,
    preprocessor
):

    artifact = {
        "model": model,
        "preprocessor": preprocessor
    }

    joblib.dump(
        artifact,
        "models/Swiggy_FoodDeliveryTimePrediction_Model.pkl"
    )

    print(
        "Model saved successfully."
    )