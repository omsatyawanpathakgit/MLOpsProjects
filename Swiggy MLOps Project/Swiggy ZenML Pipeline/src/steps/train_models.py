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
                n_estimators=250,max_depth=5,
                random_state=42
            ),

        "AdaBoost":
            AdaBoostRegressor(
                estimator=DecisionTreeRegressor(max_depth=7),
                n_estimators=400,
                learning_rate=0.5,
                random_state=42
            ),

        "XGBoost":
            XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
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