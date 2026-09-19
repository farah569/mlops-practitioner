import pytest
import pandas as pd
from prodml.features import create_features


@pytest.mark.parametrize(
    "pu, do, expected",
    [
        (132, 264, "132_264"),
        (1, 1, "1_1"),
    ],
)
def test_create_features(pu, do, expected):
    df = pd.DataFrame([{"PULocationID": pu, "DOLocationID": do}])
    result = create_features(df)
    assert result["PU_DO"].iloc[0] == expected
