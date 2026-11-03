import os

# Project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset paths
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGES_DIR = os.path.join(BASE_DIR, "Images")
TEXT_DATA_DIR = os.path.join(BASE_DIR, "Flickr_TextData")
TOKEN_FILE = os.path.join(TEXT_DATA_DIR, "Flickr8k.token.txt")
TRAIN_IMAGES_FILE = os.path.join(TEXT_DATA_DIR, "Flickr_8k.trainImages.txt")
TEST_IMAGES_FILE = os.path.join(TEXT_DATA_DIR, "Flickr_8k.testImages.txt")

# Embeddings & Preprocessed features
GLOVE_FILE = os.path.join(BASE_DIR, "glove.6B.50d.txt")
TRAIN_FEATURES_FILE = os.path.join(BASE_DIR, "encoded_train_features.pkl")
TEST_FEATURES_FILE = os.path.join(BASE_DIR, "encoded_test_features.pkl")
VOCAB_FILE = os.path.join(BASE_DIR, "vocab.pkl")

# Model checkpoints & artifacts
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
DEFAULT_WEIGHTS = os.path.join(MODELS_DIR, "model_final.weights.h5")
KERAS_MODEL_FILE = os.path.join(BASE_DIR, "full_model.keras")

# Model & Training Hyperparameters
IMAGE_SHAPE = (224, 224, 3)
FEATURE_DIM = 2048
EMBEDDING_DIM = 50
DENSE_UNITS = 256
LSTM_UNITS = 256
DROPOUT_RATE = 0.5
VOCAB_THRESHOLD = 10

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001

START_TOKEN = "<s>"
END_TOKEN = "<e>"
