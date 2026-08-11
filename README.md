# Caption-Bot-for-Assistive-Vision

# Image Captioning with ResNet50 + LSTM

Generates natural language captions for images using a CNN encoder (ResNet50) and an LSTM decoder, trained on the Flickr8k dataset.

## How it works

- **Image encoder**: A pretrained ResNet50 (ImageNet weights) extracts a 2048-dim feature vector per image (final classification layer removed).
- **Caption encoder**: Captions are cleaned (lowercased, non-alpha stripped), tokenized, and wrapped with `<s>` / `<e>` start/end tokens. Word embeddings are initialized from pretrained GloVe (50d) and frozen during training.
- **Decoder**: Image features and the partial caption sequence (embedded + LSTM) are merged and passed through dense layers to predict the next word, trained with categorical crossentropy via a custom batch generator.
- **Inference**: Captions are generated greedily — one word at a time — until the `<e>` token or max length is reached.

## Requirements

- Python packages: `tensorflow`/`keras`, `opencv-python`, `matplotlib`, `numpy`, `pandas`, `regex`, `pillow`
- Data (not included — download separately):
  - [Flickr8k dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k) — images + `Flickr8k.token.txt`, `Flickr_8k.trainImages.txt`, `Flickr_8k.testImages.txt`
  - [GloVe 6B 50d embeddings](https://nlp.stanford.edu/projects/glove/)

## Expected directory structure

```
image_captioning/
├── Flickr_TextData/
│   ├── Flickr8k.token.txt
│   ├── Flickr_8k.trainImages.txt
│   └── Flickr_8k.testImages.txt
├── Images/
│   └── *.jpg
├── glove.6B.50d.txt
├── encoded_train_features.pkl   # generated (cached ResNet50 features)
├── encoded_test_features.pkl    # generated (cached ResNet50 features)
└── saved_models/
    └── *.weights.h5
```

## Usage

1. Mount the data (originally written for Google Colab + Google Drive; adjust paths if running elsewhere).
2. Run feature extraction once to populate `encoded_train_features.pkl` / `encoded_test_features.pkl` (the extraction loop is present but commented out in favor of loading cached pickles — uncomment for a first run).
3. Train:
   ```python
   train_model(epochs=30, model=model)
   ```
   Model weights are checkpointed per epoch to `saved_models/model_{epoch}.weights.h5`.
4. Generate a caption for a new image:
   ```python
   caption = generate_caption_from_image("path/to/image.jpg")
   ```

## Notes

- Batch size (3) and epochs (30) are set low/high respectively for demonstration on limited compute — tune for your setup.
- Word frequency threshold for vocabulary inclusion is 10 occurrences.
- Two personal test images (`kanyinion.jpeg`, `muffoo.jpeg`) are referenced at the end for informal inference checks and are not part of the dataset.
