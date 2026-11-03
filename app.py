import os
import gradio as gr
from PIL import Image

from src.config import DEFAULT_WEIGHTS, KERAS_MODEL_FILE, VOCAB_FILE
from src.encoder import get_feature_extractor
from src.inference import predict_from_image
from src.model import build_caption_model
from src.preprocess import load_vocab
from src.tts import generate_speech

# Global model references (lazy loaded)
_EXTRACTOR = None
_CAPTION_MODEL = None
_WORD2IDX = None
_IDX2WORD = None
_MAX_LEN = None


def load_pipeline():
    global _EXTRACTOR, _CAPTION_MODEL, _WORD2IDX, _IDX2WORD, _MAX_LEN

    if _CAPTION_MODEL is not None:
        return True, ""

    if not os.path.exists(VOCAB_FILE):
        return False, (
            f"Vocabulary file not found at '{VOCAB_FILE}'. "
            "Please ensure vocab.pkl is placed in the project root."
        )

    try:
        import keras

        _WORD2IDX, _IDX2WORD, _MAX_LEN, vocab_size = load_vocab(VOCAB_FILE)

        if os.path.exists(KERAS_MODEL_FILE):
            print(f"Loading pre-trained model from {KERAS_MODEL_FILE}...")
            _CAPTION_MODEL = keras.models.load_model(KERAS_MODEL_FILE)
        else:
            print("Building model architecture and loading weights...")
            _CAPTION_MODEL = build_caption_model(vocab_size=vocab_size, max_len=_MAX_LEN)
            if os.path.exists(DEFAULT_WEIGHTS):
                _CAPTION_MODEL.load_weights(DEFAULT_WEIGHTS)
            else:
                print(f"Warning: Weights not found at '{DEFAULT_WEIGHTS}'. Running with uninitialized weights.")

        _EXTRACTOR = get_feature_extractor()
        return True, ""
    except Exception as e:
        return False, f"Failed to initialize model: {e}"



def process_image(input_image):
    if input_image is None:
        return "Please upload an image.", None

    ready, err_msg = load_pipeline()
    if not ready:
        return err_msg, None

    try:
        # Convert numpy array from Gradio to PIL Image if necessary
        if not isinstance(input_image, Image.Image):
            pil_img = Image.fromarray(input_image)
        else:
            pil_img = input_image

        caption = predict_from_image(
            image_input=pil_img,
            model=_CAPTION_MODEL,
            extractor=_EXTRACTOR,
            word2idx=_WORD2IDX,
            idx2word=_IDX2WORD,
            max_len=_MAX_LEN,
        )

        # Generate assistive voice readout
        audio_path = generate_speech(caption)
        return caption, audio_path

    except Exception as e:
        return f"Error during inference: {str(e)}", None


CUSTOM_CSS = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.header-box {
    text-align: center;
    margin-bottom: 20px;
}
"""


def build_app():
    with gr.Blocks(title="Caption Bot for Assistive Vision") as demo:
        with gr.Column(elem_classes=["header-box"]):
            gr.Markdown(
                """
                # 👁️ Caption Bot for Assistive Vision
                ### Translating visual scenes into natural spoken language for the visually impaired.
                Upload or capture an image to generate an accurate descriptive caption and hear it read aloud.
                """
            )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Input Image",
                    sources=["upload", "webcam", "clipboard"],
                )
                submit_btn = gr.Button("🔍 Describe Image & Speak", variant="primary", size="lg")

            with gr.Column(scale=1):
                caption_output = gr.Textbox(
                    label="Generated Caption",
                    placeholder="Caption will appear here...",
                    lines=3,
                )
                audio_output = gr.Audio(
                    label="Assistive Voice Readout (TTS)",
                    type="filepath",
                    autoplay=True,
                )

        submit_btn.click(
            fn=process_image,
            inputs=[image_input],
            outputs=[caption_output, audio_output],
        )

        gr.Markdown(
            """
            ---
            **Architecture:** ResNet50 (CNN Encoder) + GloVe 50d Word Embeddings + LSTM (Sequence Decoder) + Google Text-to-Speech (gTTS).
            """
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(css=CUSTOM_CSS)
