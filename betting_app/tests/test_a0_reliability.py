"""Annual training must exclude labels released beyond its declared origin."""
import pandas as pd

from src.models.a0_reliability import annual_partitions


def test_annual_partitions_respect_both_event_and_release_boundaries():
    frame = pd.DataFrame({
        "date": ["2023-06-30", "2023-06-30", "2023-07-01", "2023-12-31", "2024-01-01", "2025-01-01"],
        "result_day": ["2023-06-30", "2023-07-01", "2023-07-01", "2024-01-01", "2024-01-02", "2025-01-01"],
    })
    split = annual_partitions(frame, 2024)
    assert split["train"].tolist() == [0]
    assert split["cal"].tolist() == [2]
    assert split["validation"].tolist() == [4]
    assert not set(split["train"]) & set(split["cal"])


