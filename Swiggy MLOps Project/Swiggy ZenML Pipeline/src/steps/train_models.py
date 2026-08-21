from typing import Dict

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    AdaBoostRegressor
)

from xgboost import XGBRegressor

from zenml import step


@step
def train_models(
    X_train,
    y_train
) -> Dict:

    models = {

        "Linear Regression":
            LinearRegression(),

        "KNN":
            KNeighborsRegressor(),

        "Decision Tree":
            DecisionTreeRegressor(
                random_state=42
            ),

        "Extra Trees":
            ExtraTreesRegressor(
                random_state=42
            ),

        "AdaBoost":
            AdaBoostRegressor(
                random_state=42
            ),

        "XGBoost":
            XGBRegressor(
                random_state=42
            )
    }

    trained_models = {}

    for name, model in models.items():

        print(f"Training {name}...")

        model.fit(
            X_train,
            y_train
        )

        trained_models[name] = model

    return trained_models