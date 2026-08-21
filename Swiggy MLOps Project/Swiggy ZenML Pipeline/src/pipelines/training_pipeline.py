from zenml import pipeline

from src.steps.load_data import load_data
from src.steps.preprocess_data import preprocess_data
from src.steps.train_models import train_models
from src.steps.evaluate_models import evaluate_models
from src.steps.select_best_model import select_best_model
from src.steps.save_model import save_model


@pipeline
def swiggy_training_pipeline():

    # Step 1
    df = load_data()

    # Step 2
    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = preprocess_data(df)

    # Step 3
    models = train_models(
        X_train,
        y_train
    )

    # Step 4
    results = evaluate_models(
        models,
        X_test,
        y_test
    )

    # Step 5
    (
        best_model_name,
        best_model
    ) = select_best_model(
        models,
        results
    )

    # Step 6
    save_model(
        best_model,
        preprocessor
    )