import argparse
import os
import sys

from src.config import (
    BATCH_SIZE,
    EPOCHS,
    GLOVE_FILE,
    IMAGES_DIR,
    MODELS_DIR,
    TEST_FEATURES_FILE,
    TEST_IMAGES_FILE,
    TOKEN_FILE,
    TRAIN_FEATURES_FILE,
    TRAIN_IMAGES_FILE,
    VOCAB_FILE,
    VOCAB_THRESHOLD,
)


def load_image_ids(file_path: str):
    """Loads image ids from a train/test list file."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")
    return [line.split(".")[0] for line in lines if line]


def main():
    parser = argparse.ArgumentParser(
        description="Train ResNet50 + LSTM Image Captioning model on Flickr8k."
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Training batch size")
    parser.add_argument("--glove", type=str, default=GLOVE_FILE, help="Path to GloVe embeddings file")
    parser.add_argument("--tokens", type=str, default=TOKEN_FILE, help="Path to Flickr8k.token.txt")
    parser.add_argument("--train_images_list", type=str, default=TRAIN_IMAGES_FILE)
    parser.add_argument("--test_images_list", type=str, default=TEST_IMAGES_FILE)
    parser.add_argument("--images_dir", type=str, default=IMAGES_DIR)
    parser.add_argument("--models_dir", type=str, default=MODELS_DIR)
    parser.add_argument("--vocab_out", type=str, default=VOCAB_FILE)
    parser.add_argument(
        "--extract_features",
        action="store_true",
        help="Force feature re-extraction even if pickle files exist",
    )

    args = parser.parse_args()

    import pickle
    from src.dataset import data_generator
    from src.encoder import batch_extract_and_save, get_feature_extractor
    from src.model import build_caption_model, load_glove_embeddings
    from src.preprocess import (
        build_vocabulary,
        format_train_descriptions,
        load_descriptions,
        save_vocab,
    )


    os.makedirs(args.models_dir, exist_ok=True)

    # 1. Check data availability
    if not os.path.exists(args.tokens):
        print(f"Error: Token file not found at '{args.tokens}'.")
        print("Please download Flickr8k dataset into the appropriate folder (see README.md).")
        sys.exit(1)

    print("Step 1: Processing text captions...")
    descriptions = load_descriptions(args.tokens)
    print(f"Loaded descriptions for {len(descriptions)} images.")

    print(f"Step 2: Building vocabulary (threshold > {VOCAB_THRESHOLD})...")
    word2idx, idx2word, vocab_size = build_vocabulary(descriptions, threshold=VOCAB_THRESHOLD)
    print(f"Vocabulary size: {vocab_size} unique tokens (including special tokens and padding).")

    train_ids = load_image_ids(args.train_images_list)
    test_ids = load_image_ids(args.test_images_list)
    print(f"Training images: {len(train_ids)} | Test images: {len(test_ids)}")

    train_descriptions, max_len = format_train_descriptions(descriptions, train_ids)
    print(f"Maximum sequence length (with start/end tokens): {max_len}")

    print(f"Saving vocabulary metadata to {args.vocab_out}...")
    save_vocab(word2idx, idx2word, max_len, vocab_size, args.vocab_out)

    # 3. Image Feature Extraction / Loading
    extractor = None
    if args.extract_features or not os.path.exists(TRAIN_FEATURES_FILE):
        print("Extracting training features...")
        extractor = get_feature_extractor()
        encoding_train = batch_extract_and_save(
            train_ids, args.images_dir, TRAIN_FEATURES_FILE, extractor
        )
    else:
        print(f"Loading cached training features from {TRAIN_FEATURES_FILE}...")
        with open(TRAIN_FEATURES_FILE, "rb") as f:
            encoding_train = pickle.load(f)

    if args.extract_features or not os.path.exists(TEST_FEATURES_FILE):
        print("Extracting test features...")
        if extractor is None:
            extractor = get_feature_extractor()
        batch_extract_and_save(test_ids, args.images_dir, TEST_FEATURES_FILE, extractor)
    else:
        print(f"Cached test features found at {TEST_FEATURES_FILE}.")

    # 4. GloVe Embeddings
    embedding_matrix = None
    if os.path.exists(args.glove):
        embedding_matrix = load_glove_embeddings(args.glove, word2idx, vocab_size)
    else:
        print(
            f"Warning: GloVe embeddings file not found at '{args.glove}'. Model will train embedding layer from scratch."
        )

    # 5. Build Model
    print("Step 4: Building model architecture...")
    model = build_caption_model(
        vocab_size=vocab_size,
        max_len=max_len,
        embedding_matrix=embedding_matrix,
    )
    model.summary()

    # 6. Training Loop
    steps = max(1, len(train_descriptions) // args.batch_size)
    print(f"\nStarting training for {args.epochs} epochs ({steps} steps per epoch)...")

    generator = data_generator(
        train_descriptions=train_descriptions,
        encoding_train=encoding_train,
        word2idx=word2idx,
        max_len=max_len,
        vocab_size=vocab_size,
        batch_size=args.batch_size,
    )

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)

        checkpoint_path = os.path.join(args.models_dir, f"model_epoch_{epoch}.weights.h5")
        model.save_weights(checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    final_path = os.path.join(args.models_dir, "model_final.weights.h5")
    model.save_weights(final_path)
    print(f"\nTraining completed! Final weights saved to {final_path}")


if __name__ == "__main__":
    main()
