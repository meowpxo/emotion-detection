"""
Evaluation & Visualization.

Functions to check how well the model performs and to visualize the
results: training curves (accuracy/loss per epoch), a final test-set
report, and a confusion matrix.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def plot_training_curves(history):
    """
    Plot how training/validation accuracy and loss changed across epochs.

    Parameters
    ----------
    history : dict
        Output of training_pipeline.train_and_validate().
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["epoch"], history["train_accuracy"], label="Train")
    axes[0].plot(history["epoch"], history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_loss"], label="Train")
    axes[1].plot(history["epoch"], history["val_loss"], label="Validation")
    axes[1].set_title("Loss per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def evaluate_model(model, X_test, y_test):
    """
    Print the accuracy and a per-emotion performance report.

    Returns
    -------
    y_pred : array
        The model's predicted emotions for X_test (used for the confusion matrix).
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    print(f"Accuracy: {accuracy:.2%}\n")
    print("Classification Report:")
    print(report)

    return y_pred


def plot_confusion_matrix(y_test, y_pred, labels):
    """
    Draw a heatmap showing which emotions the model mixes up.

    Rows = actual emotion, Columns = predicted emotion.
    A perfect model would have all its values on the diagonal.
    """
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Emotion")
    plt.ylabel("Actual Emotion")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
