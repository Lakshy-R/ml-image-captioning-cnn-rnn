# Image Captioning — CNN–RNN on MS COCO

**Course:** Advanced Machine Learning, Spring 2026
**Institution:** German International University of Applied Sciences
**Assignment:** 2 — A Deep Learning Framework for Image Captioning using CNN–RNN Architecture

---

## Overview

This project builds an end-to-end deep learning pipeline that generates natural language captions for images from the [MS COCO dataset](https://www.kaggle.com/datasets/hariwh0/ms-coco-dataset).

The architecture mirrors the natural decomposition of the problem:
- A **CNN encoder** compresses each image into a fixed-length visual representation (the "eye")
- An **RNN decoder** generates a caption word-by-word conditioned on that representation (the "voice")

An auxiliary image classification objective sharpens the visual features and improves caption relevance.

---

## Architecture

```
Image → CNN Encoder → Feature Vector → RNN Decoder → Caption (word by word)
                            ↓
                 Auxiliary Classification Head
```

| Component | Role |
|---|---|
| CNN (pretrained/custom) | Visual feature extraction |
| Embedding Layer | Word representation |
| RNN Decoder (LSTM/GRU) | Sequential caption generation |
| Dense Output Layer | Vocabulary prediction at each step |

---

## Dataset

**MS COCO (Common Objects in Context)**

| Split | Images | Captions |
|---|---|---|
| Train | 118,000 | 590,000 |
| Validation | 5,000 | 25,000 |
| Test | 5,000 | 25,000 |

> At least 50% of the training data is used. Each image has 5 human-written captions, providing a rich supervisory signal.

Dataset link: [Kaggle — MS COCO](https://www.kaggle.com/datasets/hariwh0/ms-coco-dataset)

---

## Project Structure

```
ml-image-captioning-cnn-rnn/
│
├── image-captioning.ipynb          # Full implementation notebook
└── README.md
```

---

## Notebook Contents

1. **Data Preprocessing** — caption cleaning & tokenization, vocabulary construction, sequence preparation, CNN feature extraction
2. **Model Implementation** — CNN encoder, embedding layer, RNN decoder, dense output head
3. **Training** — end-to-end training with auxiliary classification loss
4. **Evaluation** — caption quality metrics, sample generated captions

---

## Setup

```bash
pip install numpy pandas matplotlib torch torchvision nltk pycocotools
jupyter notebook image-captioning.ipynb
```

> Note: Downloading MS COCO requires ~20 GB of disk space. A 50%+ subset is sufficient for training.

---

## Key Concepts

- **CNN–RNN interface:** the CNN's final feature vector is injected as the initial hidden state of the RNN decoder
- **Teacher forcing:** ground-truth tokens are fed as inputs during training for stable sequence learning
- **Auxiliary supervision:** a classification head on the CNN features provides an additional training signal
