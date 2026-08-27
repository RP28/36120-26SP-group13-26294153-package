from __future__ import annotations

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

@pytest.fixture
def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "cat": ["x", "y", "x", "y", "x", "z"],
            "target": [0, 1, 0, 1, 0, 1],
        },
        index=[10, 11, 12, 13, 14, 15],
    )
