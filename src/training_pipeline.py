"""
Training Pipeline.

Trains the classifier one epoch at a time. After every epoch, we check
accuracy and loss on BOTH the training set and the validation set. This
lets us see, epoch by epoch, whether the model is actually learning or
starting to overfit (train performance keeps improving while validation
performance stops improving or gets worse).
"""

import numpy as np
from sklearn.metrics import accuracy_score, log_loss


def train_and_validate(vectorizer, classifier, X_train, y_train, X_val, y_val, epochs=15):
    """
    Fit the vectorizer on the training text, then train the classifier
    for a number of epochs, evaluating on the validation set after each one.

    Parameters
    ----------
    vectorizer : untrained TfidfVectorizer (from model_builder.build_vectorizer)
    classifier : untrained SGDClassifier (from model_builder.build_classifier)
    X_train, y_train : training text and labels
    X_val, y_val : validation text and labels
    epochs : int
        Number of passes over the training data.

    Returns
    -------
    classifier : the trained classifier
    vectorizer : the fitted vectorizer
    history : dict
        Per-epoch train/validation accuracy and loss, useful for plotting.
    """
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    all_classes = np.unique(y_train)

    history = {
        "epoch": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(1, epochs + 1):
        classifier.partial_fit(X_train_vec, y_train, classes=all_classes)

        train_pred = classifier.predict(X_train_vec)
        val_pred = classifier.predict(X_val_vec)
        train_proba = classifier.predict_proba(X_train_vec)
        val_proba = classifier.predict_proba(X_val_vec)

        history["epoch"].append(epoch)
        history["train_accuracy"].append(accuracy_score(y_train, train_pred))
        history["val_accuracy"].append(accuracy_score(y_val, val_pred))
        history["train_loss"].append(log_loss(y_train, train_proba, labels=all_classes))
        history["val_loss"].append(log_loss(y_val, val_proba, labels=all_classes))

        print(
            f"Epoch {epoch:2d}/{epochs} - "
            f"train_acc: {history['train_accuracy'][-1]:.3f} - "
            f"val_acc: {history['val_accuracy'][-1]:.3f} - "
            f"train_loss: {history['train_loss'][-1]:.3f} - "
            f"val_loss: {history['val_loss'][-1]:.3f}"
        )

    return classifier, vectorizer, history
