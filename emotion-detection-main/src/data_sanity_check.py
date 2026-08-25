"""
Data Pipeline: Sanity checks.

Before we spend time training a model, we check the data for obvious
problems: missing text, missing labels, duplicate rows, or text that
becomes empty after cleaning. Catching these early avoids confusing
errors later during training.
"""


def run_sanity_checks(df, text_column="content", label_column="sentiment",
                       clean_text_column=None):
    """
    Print a short data-quality report.

    Parameters
    ----------
    df : pandas.DataFrame
    text_column : str
        Column with the raw tweet text.
    label_column : str
        Column with the emotion label.
    clean_text_column : str or None
        If given and present in df, also checks for text that became
        empty after cleaning (e.g. a tweet that was only a URL).

    Returns
    -------
    passed : bool
        True if no missing text/labels were found.
    report : dict
        The individual counts that were checked.
    """
    report = {
        "total_rows": len(df),
        "missing_text": int(df[text_column].isna().sum()),
        "missing_labels": int(df[label_column].isna().sum()),
        "duplicate_rows": int(df.duplicated(subset=[text_column, label_column]).sum()),
        "unique_labels": sorted(df[label_column].dropna().unique().tolist()),
    }

    if clean_text_column is not None and clean_text_column in df.columns:
        empty_mask = df[clean_text_column].str.strip() == ""
        report["empty_after_cleaning"] = int(empty_mask.sum())

    print("DATA SANITY CHECK")
    print("-" * 40)
    for key, value in report.items():
        print(f"{key}: {value}")
    print("-" * 40)

    passed = report["missing_text"] == 0 and report["missing_labels"] == 0
    if passed:
        print("PASSED: no missing text or labels.")
    else:
        print("WARNING: missing text or labels found. Review before training.")

    return passed, report
