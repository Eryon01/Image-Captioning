# Image Captioning with CNN–Transformer (TensorFlow)

This repository implements an image captioning model that combines a **CNN-based image encoder** (InceptionV3) with a **Transformer-based text decoder** in TensorFlow/Keras.

The model takes an input image and generates a natural-language description (caption) using a learned vocabulary and a trained CNN–Transformer architecture.

---

## Features

- 🧠 **Pretrained InceptionV3 Encoder**  
  Uses `tf.keras.applications.InceptionV3` (ImageNet weights) as a fixed feature extractor.

- 🔤 **Custom Text Tokenizer & Vocabulary**  
  Loads a precomputed vocabulary from `saved_models/vocab.file` and uses `TextVectorization` + `StringLookup` layers.

- 🔁 **Transformer-based Encoder–Decoder**  
  - Transformer-style encoder over CNN features  
  - Transformer-style decoder with:
    - Multi-head self-attention
    - Cross-attention over image embeddings
    - Positional and token embeddings

- 📊 **Custom Training Loop (Keras Model subclassing)**  
  - Custom `train_step` and `test_step`  
  - Masked loss and accuracy calculations (ignores padding tokens)

- 🖼️ **Easy Inference Utility**  
  - `get_caption_model()` to load the model with pretrained weights  
  - `generate_caption()` to generate captions for a given image path or NumPy array

---

## Project Structure

> This is the logical structure implied by the code. Adjust based on your actual repo.

```text
.
├── saved_models/
│   ├── vocab.file
│   └── image_captioning_transformer_weights.h5
├── image_captioning_model.py   # (this file / core model code)
└── README.md
