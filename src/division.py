from sklearn.model_selection import train_test_split


def dividir_train_test(df, test_size=0.2, random_state=42):
    if df["lesion_id"].duplicated().any():
        raise ValueError("El conjunto debe contener una sola imagen por lesión.")

    return train_test_split(
        df,
        test_size=test_size,
        stratify=df["etiqueta"],
        random_state=random_state,
    )
