import collections
import pickle
import re
from typing import Dict, List, Tuple

from src.config import END_TOKEN, START_TOKEN, VOCAB_THRESHOLD


def clean_text(sentence: str) -> str:
    """Lowercases, removes non-alphabetic characters, and drops single-character words."""
    sentence = sentence.lower()
    sentence = re.sub(r"[^a-z]+", " ", sentence)
    words = [s for s in sentence.split() if len(s) > 1]
    return " ".join(words)


def load_descriptions(token_file_path: str) -> Dict[str, List[str]]:
    """Loads and cleans image descriptions from Flickr token text file.

    Returns:
        dict: Mapping of image_id -> list of cleaned captions.
    """
    with open(token_file_path, "r", encoding="utf-8") as f:
        content = f.read()

    descriptions: Dict[str, List[str]] = {}
    for line in content.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        img_name = parts[0].split(".")[0]
        caption = clean_text(parts[1])

        if img_name not in descriptions:
            descriptions[img_name] = []
        descriptions[img_name].append(caption)

    return descriptions


def build_vocabulary(
    descriptions: Dict[str, List[str]], threshold: int = VOCAB_THRESHOLD
) -> Tuple[Dict[str, int], Dict[int, str], int]:
    """Filters words by frequency threshold and assigns index mappings.

    Index 0 is reserved for padding.
    """
    total_words: List[str] = []
    for cap_list in descriptions.values():
        for cap in cap_list:
            total_words.extend(cap.split())

    counter = collections.Counter(total_words)
    frequent_words = [word for word, count in counter.items() if count > threshold]

    word2idx: Dict[str, int] = {}
    idx2word: Dict[int, str] = {}

    for i, word in enumerate(frequent_words, start=1):
        word2idx[word] = i
        idx2word[i] = word

    next_idx = len(word2idx) + 1
    word2idx[START_TOKEN] = next_idx
    idx2word[next_idx] = START_TOKEN

    next_idx += 1
    word2idx[END_TOKEN] = next_idx
    idx2word[next_idx] = END_TOKEN

    vocab_size = len(word2idx) + 1  # 0 is reserved for padding
    return word2idx, idx2word, vocab_size


def format_train_descriptions(
    descriptions: Dict[str, List[str]], train_image_ids: List[str]
) -> Tuple[Dict[str, List[str]], int]:
    """Wraps training captions with start and end tokens, and calculates max length."""
    train_descriptions: Dict[str, List[str]] = {}
    max_len = 0

    for img_id in train_image_ids:
        if img_id not in descriptions:
            continue
        train_descriptions[img_id] = []
        for cap in descriptions[img_id]:
            wrapped = f"{START_TOKEN} {cap} {END_TOKEN}"
            train_descriptions[img_id].append(wrapped)
            max_len = max(max_len, len(wrapped.split()))

    return train_descriptions, max_len


def save_vocab(
    word2idx: Dict[str, int],
    idx2word: Dict[int, str],
    max_len: int,
    vocab_size: int,
    path: str,
) -> None:
    """Serializes vocabulary and metadata to a pickle file."""
    data = {
        "word2idx": word2idx,
        "idx2word": idx2word,
        "max_len": max_len,
        "vocab_size": vocab_size,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_vocab(path: str) -> Tuple[Dict[str, int], Dict[int, str], int, int]:
    """Loads vocabulary and metadata from a pickle file."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["word2idx"], data["idx2word"], data["max_len"], data["vocab_size"]
