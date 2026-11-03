import os
from typing import Dict, Optional
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Embedding, Input, LSTM, add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from src.config import (
    DENSE_UNITS,
    DROPOUT_RATE,
    EMBEDDING_DIM,
    FEATURE_DIM,
    LEARNING_RATE,
    LSTM_UNITS,
)


def load_glove_embeddings(
    glove_path: str, word2idx: Dict[str, int], vocab_size: int, emb_dim: int = EMBEDDING_DIM
) -> np.ndarray:
    """Reads GloVe word vectors and constructs the embedding matrix for the vocabulary."""
    print(f"Loading GloVe embeddings from {glove_path}...")
    embeddings_index = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs

    print(f"Found {len(embeddings_index)} word vectors in GloVe.")

    embedding_matrix = np.zeros((vocab_size, emb_dim), dtype="float32")
    for word, idx in word2idx.items():
        if idx < vocab_size:
            vector = embeddings_index.get(word)
            if vector is not None:
                embedding_matrix[idx] = vector

    return embedding_matrix


def build_caption_model(
    vocab_size: int,
    max_len: int,
    emb_dim: int = EMBEDDING_DIM,
    embedding_matrix: Optional[np.ndarray] = None,
    feature_dim: int = FEATURE_DIM,
    units: int = DENSE_UNITS,
    lstm_units: int = LSTM_UNITS,
    dropout_rate: float = DROPOUT_RATE,
    learning_rate: float = LEARNING_RATE,
) -> Model:
    """Constructs the ResNet50 + LSTM Image Captioning model architecture."""

    # 1. Image feature extraction branch
    inputs_image = Input(shape=(feature_dim,), name="image_input")
    fe1 = Dropout(dropout_rate)(inputs_image)
    fe2 = Dense(units, activation="relu", name="image_dense")(fe1)

    # 2. Text sequence branch
    inputs_text = Input(shape=(max_len,), name="text_input")
    if embedding_matrix is not None:
        se1 = Embedding(
            vocab_size,
            emb_dim,
            weights=[embedding_matrix],
            trainable=False,
            mask_zero=True,
            name="caption_embedding",
        )(inputs_text)
    else:
        se1 = Embedding(
            vocab_size,
            emb_dim,
            mask_zero=True,
            name="caption_embedding",
        )(inputs_text)

    se2 = Dropout(dropout_rate)(se1)
    se3 = LSTM(lstm_units, name="caption_lstm")(se2)

    # 3. Decoder: merge both representations
    decoder1 = add([fe2, se3], name="merge_layer")
    decoder2 = Dense(units, activation="relu", name="decoder_dense")(decoder1)
    outputs = Dense(vocab_size, activation="softmax", name="output_layer")(decoder2)

    model = Model(inputs=[inputs_image, inputs_text], outputs=outputs)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )
    return model
