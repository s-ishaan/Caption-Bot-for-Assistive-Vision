import argparse
import os
import sys

from src.config import DEFAULT_WEIGHTS, KERAS_MODEL_FILE, VOCAB_FILE


def main():
    parser = argparse.ArgumentParser(
        description="Generate natural language caption and assistive speech for an image."
    )
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        required=True,
        help="Path to the input image file (jpg, png, etc.)",
    )
    parser.add_argument(
        "--weights",
        "-w",
        type=str,
        default=DEFAULT_WEIGHTS,
        help=f"Path to trained model weights (default: {DEFAULT_WEIGHTS})",
    )
    parser.add_argument(
        "--vocab",
        "-v",
        type=str,
        default=VOCAB_FILE,
        help=f"Path to saved vocabulary pickle (default: {VOCAB_FILE})",
    )
    parser.add_argument(
        "--tts",
        action="store_true",
        help="Enable text-to-speech audio generation for assistive vision",
    )
    parser.add_argument(
        "--audio_out",
        type=str,
        default="caption_audio.mp3",
        help="Output path for audio file when --tts is enabled",
    )

    args = parser.parse_args()

    # Lazy imports for fast CLI response and graceful dependency checking
    from src.encoder import get_feature_extractor
    from src.inference import predict_from_image
    from src.model import build_caption_model
    from src.preprocess import load_vocab
    from src.tts import generate_speech


    if not os.path.exists(args.image):
        print(f"Error: Image '{args.image}' not found.")
        sys.exit(1)

    if not os.path.exists(args.vocab):
        print(
            f"Error: Vocabulary file '{args.vocab}' not found. Please train the model or provide valid vocab.pkl."
        )
        sys.exit(1)

    print("Loading vocabulary...")
    word2idx, idx2word, max_len, vocab_size = load_vocab(args.vocab)

    import keras

    if os.path.exists(KERAS_MODEL_FILE):
        print(f"Loading full model from {KERAS_MODEL_FILE}...")
        model = keras.models.load_model(KERAS_MODEL_FILE)
    else:
        print("Building model architecture...")
        model = build_caption_model(vocab_size=vocab_size, max_len=max_len)
        if os.path.exists(args.weights):
            print(f"Loading weights from {args.weights}...")
            model.load_weights(args.weights)
        else:
            print(
                f"Warning: Weights file '{args.weights}' not found. Running with uninitialized weights."
            )


    print("Initializing ResNet50 feature extractor...")
    extractor = get_feature_extractor()

    print(f"Generating caption for: {args.image}")
    caption = predict_from_image(
        image_input=args.image,
        model=model,
        extractor=extractor,
        word2idx=word2idx,
        idx2word=idx2word,
        max_len=max_len,
    )

    print("\n" + "=" * 50)
    print(f"Generated Caption: \"{caption}\"")
    print("=" * 50 + "\n")

    if args.tts:
        print("Synthesizing assistive speech...")
        audio_file = generate_speech(caption, output_path=args.audio_out)
        if audio_file:
            print(f"Audio saved to: {audio_file}")


if __name__ == "__main__":
    main()
