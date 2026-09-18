import pandas as pd
from prodml.config import DATA_URL


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_URL)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # data clean
    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df["duration"] = df.duration.apply(lambda td: td.total_seconds() / 60)

    # filtering trips
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    return df
