from typing import Dict, Tuple, Any

from zenml import step


@step
def select_best_model(
    models: Dict,
    results
) -> Tuple[str, Any]:

    best_model_name = (
        results
        .sort_values(
            by="r2",
            ascending=False
        )
        .iloc[0]["model"]
    )

    best_model = models[best_model_name]

    print(f"Best model: {best_model_name}")

    return best_model_name, best_model