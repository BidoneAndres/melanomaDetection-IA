import pandas as pd
import pytest

from src.division import dividir_train_test


def test_dividir_train_test_estratifica_y_no_separa_lesiones():
    df = pd.DataFrame(
        {
            "lesion_id": [f"lesion_{i}" for i in range(40)],
            "etiqueta": [0, 1] * 20,
        }
    )

    train, test = dividir_train_test(df)

    assert set(train["lesion_id"]).isdisjoint(test["lesion_id"])
    assert train["etiqueta"].value_counts(normalize=True).to_dict() == {
        0: 0.5,
        1: 0.5,
    }
    assert test["etiqueta"].value_counts(normalize=True).to_dict() == {
        0: 0.5,
        1: 0.5,
    }


def test_dividir_train_test_rechaza_lesiones_duplicadas():
    df = pd.DataFrame(
        {
            "lesion_id": ["lesion_a", "lesion_a", "lesion_b", "lesion_c"],
            "etiqueta": [0, 0, 1, 1],
        }
    )

    with pytest.raises(ValueError, match="una sola imagen por lesión"):
        dividir_train_test(df)
