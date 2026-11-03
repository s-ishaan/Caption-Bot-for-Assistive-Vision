import os
import pickle
from typing import Dict, List, Union
import numpy as np
from PIL import Image
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

from src.config import FEATURE_DIM, IMAGE_SHAPE


def get_feature_extractor() -> Model:
    """Instantiates ResNet50 pretrained on ImageNet and removes the classification head."""
    base_model = ResNet50(weights="imagenet", input_shape=IMAGE_SHAPE)
    feature_extractor = Model(
        inputs=base_model.input, outputs=base_model.layers[-2].output
    )
    return feature_extractor


def preprocess_image(img_input: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
    """Converts a file path, PIL image, or numpy array to a normalized batch for ResNet50."""
    if isinstance(img_input, str):
        img = image.load_img(img_input, target_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
    elif isinstance(img_input, Image.Image):
        img = img_input.resize((IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
    elif isinstance(img_input, np.ndarray):
        img = Image.fromarray(img_input).resize((IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
    else:
        raise ValueError(f"Unsupported image type: {type(img_input)}")

    img_arr = image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = preprocess_input(img_arr)
    return img_arr


def extract_features(
    img_input: Union[str, np.ndarray, Image.Image],
    extractor: Model,
) -> np.ndarray:
    """Extracts a 2048-dim feature vector for a single image."""
    preprocessed = preprocess_image(img_input)
    features = extractor.predict(preprocessed, verbose=0)
    return features.reshape((FEATURE_DIM,))


def batch_extract_and_save(
    image_ids: List[str],
    images_dir: str,
    output_pkl: str,
    extractor: Model = None,
) -> Dict[str, np.ndarray]:
    """Extracts features for all specified images and saves them to a pickle file."""
    if extractor is None:
        extractor = get_feature_extractor()

    features_dict: Dict[str, np.ndarray] = {}
    print(f"Extracting features for {len(image_ids)} images...")

    for img_id in tqdm(image_ids):
        img_path = os.path.join(images_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            continue
        try:
            features = extract_features(img_path, extractor)
            features_dict[img_id] = features
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")

    with open(output_pkl, "wb") as f:
        pickle.dump(features_dict, f)

    print(f"Saved {len(features_dict)} feature vectors to {output_pkl}")
    return features_dict
