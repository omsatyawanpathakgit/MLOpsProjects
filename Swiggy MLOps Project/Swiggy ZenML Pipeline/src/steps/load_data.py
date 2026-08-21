import pandas as pd
from zenml import step


@step
def load_data() -> pd.DataFrame:

    df = pd.read_csv("data/swiggy.csv")

    print("Dataset shape:", df.shape)

    return df