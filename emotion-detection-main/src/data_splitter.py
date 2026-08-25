"""
Data Pipeline: Train / Validation / Test splitting.

We split the dataset into three parts:
- train      -> the model learns from this
- validation -> checked after every training epoch, to see if the model
                is actually improving (and not just memorizing train data)
- test       -> only used once, at the very end, for the final unbiased
                evaluation
"""

from sklearn.model_selection import train_test_split


def split_data(df, text_column="clean_text", label_column="sentiment",
                val_size=0.1, test_size=0.2, random_state=42):
    """
    Split the DataFrame into train/validation/test inputs (X) and outputs (y).

    Parameters
    ----------
    df : pandas.DataFrame
    text_column : str
        Column with the cleaned tweet text (the model's input).
    label_column : str
        Column with the emotion label (what the model must predict).
    val_size : float
        Fraction of the FULL dataset to use for validation (e.g. 0.1 = 10%).
    test_size : float
        Fraction of the FULL dataset to use for testing (e.g. 0.2 = 20%).
    random_state : int
        Fixes the randomness so the split is the same every time we run it.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    X = df[text_column]
    y = df[label_column]

    # Step 1: carve out the test set first.
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,  # keep the same emotion proportions in every split
    )

    # Step 2: split what's left into train and validation.
    # val_size was defined relative to the FULL dataset, so we convert it
    # to a fraction of what remains (X_temp) before splitting again.
    relative_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=y_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
