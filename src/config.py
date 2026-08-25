"""
Central configuration for the project.

Every file path, hyperparameter, and setting used by the pipeline lives
here. If you want to change how the model trains, change a value in this
file instead of hunting through every .py file.
"""

# ---- File paths ----
DATA_PATH = "tweet_emotions.csv"
MODEL_SAVE_PATH = "models/emotion_model.pkl"

# ---- Dataset columns ----
TEXT_COLUMN = "content"
CLEAN_TEXT_COLUMN = "clean_text"
LABEL_COLUMN = "sentiment"

# ---- Train / Validation / Test split ----
# The dataset is split three ways: train (learn), validation (check
# progress during training), and test (final, untouched evaluation).
VAL_SIZE = 0.1          # 10% of the full dataset
TEST_SIZE = 0.2         # 20% of the full dataset
RANDOM_STATE = 42       # fixes randomness so results are reproducible

# ---- Feature extraction (TF-IDF) ----
MAX_FEATURES = 5000     # maximum number of unique word/phrase features
NGRAM_RANGE = (1, 2)    # use single words and pairs of words

# ---- Model / training hyperparameters ----
ALPHA = 1e-4             # regularization strength (higher = simpler model)
LEARNING_RATE = "optimal"
EPOCHS = 15              # number of passes over the training data
