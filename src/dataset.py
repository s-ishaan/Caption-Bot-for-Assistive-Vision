import random
from typing import Dict, Iterator, List, Tuple
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


def data_generator(
    train_descriptions: Dict[str, List[str]],
    encoding_train: Dict[str, np.ndarray],
    word2idx: Dict[str, int],
    max_len: int,
    vocab_size: int,
    batch_size: int,
) -> Iterator[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]]:
    """Yields batches of (image_features, padded_partial_captions) -> next_word_categorical.

    Iterates endlessly for model.fit.
    """
    image_keys = list(train_descriptions.keys())

    while True:
        X1, X2, y = [], [], []
        for _ in range(batch_size):
            key = random.choice(image_keys)
            if key not in encoding_train:
                continue

            photo = encoding_train[key]
            caption = random.choice(train_descriptions[key])

            seq = [word2idx[word] for word in caption.split() if word in word2idx]

            for i in range(1, len(seq)):
                xi = seq[:i]
                yi = seq[i]

                xi_padded = pad_sequences([xi], maxlen=max_len, value=0, padding="post")[0]
                yi_cat = to_categorical(yi, num_classes=vocab_size)

                X1.append(photo)
                X2.append(xi_padded)
                y.append(yi_cat)

        if len(X1) > 0:
            yield ((np.array(X1), np.array(X2)), np.array(y))
