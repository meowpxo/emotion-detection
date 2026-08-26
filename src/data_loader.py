"""
Data Pipeline: Loading.

This file has simple functions to read the CSV file into a pandas
DataFrame (basically a spreadsheet-like table we can work with in Python).
"""

import pandas as pd


def load_dataset(csv_path):
    """
    Read the tweet emotion CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file, e.g. "tweet_emotions.csv"

    Returns
    -------
    pandas.DataFrame
        Table with columns: tweet_id, sentiment, content
    """
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=["content", "sentiment"]).reset_index(drop=True)
    return df


def get_class_counts(df, label_column="sentiment"):
    """
    Count how many tweets belong to each emotion.

    Returns
    -------
    pandas.Series
        Emotion name -> number of tweets, sorted from most to least common.
    """
    return df[label_column].value_counts()
