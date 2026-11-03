from typing import Dict, Union
import numpy as np
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.config import END_TOKEN, FEATURE_DIM, START_TOKEN
from src.encoder import extract_features


def predict_caption(
    photo_features: np.ndarray,
    model: Model,
    word2idx: Dict[str, int],
    idx2word: Dict[int, str],
    max_len: int,
) -> str:
    """Generates a caption greedily one word at a time given a 2048-dim feature vector.

    Args:
        photo_features: Array of shape (1, 2048) or (2048,).
        model: Trained Keras caption model.
        word2idx: Mapping of word -> index.
        idx2word: Mapping of index -> word.
        max_len: Maximum caption length.

    Returns:
        The generated caption string.
    """
    if photo_features.ndim == 1:
        photo_features = photo_features.reshape((1, FEATURE_DIM))

    in_text = START_TOKEN

    for _ in range(max_len):
        sequence = [word2idx[w] for w in in_text.split() if w in word2idx]
        sequence = pad_sequences([sequence], maxlen=max_len, padding="post")

        ypred = model.predict([photo_features, sequence], verbose=0)
        idx_pred = int(np.argmax(ypred))
        word = idx2word.get(idx_pred, "")

        if not word or word == END_TOKEN:
            break

        in_text += f" {word}"

    # Strip start token and any trailing end token
    words = [w for w in in_text.split() if w not in (START_TOKEN, END_TOKEN)]
    return " ".join(words)


def predict_from_image(
    image_input: Union[str, np.ndarray, Image.Image],
    model: Model,
    extractor: Model,
    word2idx: Dict[str, int],
    idx2word: Dict[int, str],
    max_len: int,
) -> str:
    """Extracts features from an image and generates its caption."""
    features = extract_features(image_input, extractor)
    return predict_caption(features, model, word2idx, idx2word, max_len)
