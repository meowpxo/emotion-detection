"""
Save and load the trained model.

Training takes time, so we save the finished model to a file once,
then reload it later instead of retraining every time.
"""

import os
import joblib


def save_model(model, path="models/emotion_model.pkl"):
    """Save a trained model pipeline to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to: {path}")


def load_model(path="models/emotion_model.pkl"):
    """Load a previously saved model pipeline from disk."""
    return joblib.load(path)
