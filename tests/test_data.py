import pytest

from heart.data import load_clean_data
from heart.validation import validate_dataframe


def test_clean_data_schema():
    df = load_clean_data()
    assert len(df) == 303
    assert "target" in df.columns
    assert validate_dataframe(df, is_training=True) is True

def test_target_is_binary():
    df = load_clean_data()
    unique_targets = set(df["target"].unique())
    assert unique_targets.issubset({0, 1})

def test_no_missing_values_in_clean_data():
    df = load_clean_data()
    assert df.isnull().sum().sum() == 0

def test_validation_rejects_out_of_range():
    df = load_clean_data().copy()
    df.loc[0, "age"] = 150 # Invalid age
    with pytest.raises(ValueError):
        validate_dataframe(df, is_training=True)

def test_validation_rejects_invalid_categorical():
    df = load_clean_data().copy()
    df.loc[0, "sex"] = 5 # Allowed: [0, 1]
    with pytest.raises(ValueError):
        validate_dataframe(df, is_training=True)
