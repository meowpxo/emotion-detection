"""
Data Pipeline: Text cleaning.

Tweets are messy (links, @mentions, hashtags, typos, punctuation...).
Cleaning the text first makes it much easier for the model to learn
useful patterns instead of noise.
"""

import re


def clean_text(text):
    """
    Clean a single tweet.

    What this does, in order:
    1. Lowercase everything          ("Happy" -> "happy")
    2. Remove links                  ("check http://x.com" -> "check")
    3. Remove @mentions               ("@john hi" -> "hi")
    4. Remove the '#' symbol but keep the word ("#fun" -> "fun")
    5. Remove numbers and punctuation
    6. Collapse extra spaces
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # links
    text = re.sub(r"@\w+", " ", text)                # @mentions
    text = re.sub(r"#", " ", text)                   # hashtag symbol
    text = re.sub(r"[^a-z\s]", " ", text)             # numbers/punctuation
    text = re.sub(r"\s+", " ", text).strip()          # extra whitespace
    return text


def clean_dataframe(df, text_column="content", new_column="clean_text"):
    """
    Apply clean_text() to every row of a DataFrame column and store the
    result in a new column.

    Parameters
    ----------
    df : pandas.DataFrame
    text_column : str
        Column that holds the raw tweet text.
    new_column : str
        Name of the new column to create with cleaned text.

    Returns
    -------
    pandas.DataFrame
        A copy of df with the new cleaned-text column added.
    """
    df = df.copy()
    df[new_column] = df[text_column].apply(clean_text)
    return df
