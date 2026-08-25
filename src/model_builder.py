"""
Model Architecture.

The model has two parts:
1. TfidfVectorizer  - turns text into numbers (computers only understand numbers).
2. SGDClassifier    - a linear classifier trained with "log_loss", which is
   just logistic regression trained one small batch (epoch) at a time via
   partial_fit(). We use this instead of plain LogisticRegression so the
   training pipeline can check validation performance after every epoch,
   the same way you would when training a neural network.

We build the vectorizer and the classifier separately (instead of one
combined Pipeline) because the training pipeline needs to fit the
vectorizer once, then loop over the classifier's training epochs.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline


def build_vectorizer(max_features=5000, ngram_range=(1, 2)):
    """Create an untrained TF-IDF vectorizer."""
    return TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)


def build_classifier(alpha=1e-4, learning_rate="optimal", random_state=42):
    """Create an untrained linear classifier (logistic regression via SGD)."""
    return SGDClassifier(
        loss="log_loss",
        alpha=alpha,
        learning_rate=learning_rate,
        random_state=random_state,
    )


def combine_into_pipeline(vectorizer, classifier):
    """
    Bundle an already-trained vectorizer + classifier into a single
    Pipeline object. This lets the rest of the project (saving, loading,
    predicting on raw text) work with one simple object, exactly like
    before training was split into epochs.
    """
    return Pipeline([("tfidf", vectorizer), ("classifier", classifier)])
