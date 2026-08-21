from typing import Dict

import pandas as pd

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from zenml import step


@step
def evaluate_models(
    models: Dict,
    X_test,
    y_test
) -> pd.DataFrame:

    results = []

    for name, model in models.items():

        predictions = model.predict(X_test)

        r2 = r2_score(
            y_test,
            predictions
        )

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = mean_squared_error(
            y_test,
            predictions
        ) ** 0.5

        results.append({
            "model": name,
            "r2": r2,
            "mae": mae,
            "rmse": rmse
        })

    results_df = pd.DataFrame(results)

    print("\nModel Performance:")
    print(results_df)

    return results_df