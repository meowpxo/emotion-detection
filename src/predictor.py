"""
Use the trained model to predict emotions on new text.
"""

from src.text_cleaner import clean_text


def predict_emotion(model, texts):
    """
    Predict the emotion of one or more pieces of text.

    Parameters
    ----------
    model : trained sklearn Pipeline
    texts : str or list of str
        A single sentence, or a list of sentences.

    Returns
    -------
    list
        Predicted emotion label for each input sentence.
    """
    if isinstance(texts, str):
        texts = [texts]

    cleaned_texts = [clean_text(t) for t in texts]
    predictions = model.predict(cleaned_texts)
    return list(predictions)
