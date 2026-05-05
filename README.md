# Human Face to Comic Face Translation using Pix2Pix GAN 🎭➡️🖌️

## 👨‍🎨 Overview

This project presents a deep learning-based system that transforms real human facial images into comic-style portraits using the **Pix2Pix conditional Generative Adversarial Network (cGAN)**. The model is trained on the **Face2Comic** dataset and demonstrated through a **Streamlit** web application for real-time image translation.

## 🚀 Live Demo
👉 [Comic Face app](https://comicface.streamlit.app/)

---

## 🧠 Author
- Utkarsh Mishra

---

## ✨ Key Features

- Converts human faces to comic-style portraits
- Trained using paired real-comic images with Pix2Pix (cGAN)
- Streamlit web app for real-time image translation
- Identity-preserving and stylized outputs

---

## 📚 Table of Contents

- [Introduction](#-introduction)
- [Related Work](#-related-work)
- [Dataset](#-dataset-face2comic)
- [Methodology](#-methodology)
- [Implementation Details](#-implementation-details)
- [Results](#-results)
- [Web Application](#-streamlit-web-application)
- [Conclusion](#-conclusion)
- [References](#-references)

---

## 🔍 Introduction

Image-to-image translation is a key task in computer vision involving the mapping of an image from one domain to another. In this project, we focus on **converting real human faces into comic-styled images** — a task relevant to animation, avatars, and entertainment industries.

We use **Pix2Pix**, a supervised cGAN, capable of learning pixel-to-pixel mappings for paired datasets, making it ideal for our task.

---

## 🔗 Related Work

- **Pix2Pix (Isola et al., 2017)**: Conditional GAN for supervised image-to-image translation.
- **CycleGAN**: Handles unpaired data for domain translation.
- **CartoonGAN / StyleGAN**: Facial stylization with more complex architectures.

Pix2Pix is chosen for its simplicity, pixel-level accuracy, and compatibility with paired data.

---

## 🧾 Dataset: Face2Comic

- Source: [Kaggle - Face2Comic](https://www.kaggle.com/datasets)
- **Total Images**: 10,000 paired (real + comic)
- **Preprocessing**:
  - Resized to `256x256`
  - Normalized to `[-1, 1]`
  - Matched using identical filenames

Diverse in terms of age, lighting, expressions, and pose.

---

## 🏗️ Methodology

### Pix2Pix Architecture

- **Generator**: U-Net with skip connections
- **Discriminator**: PatchGAN (focuses on 70×70 image patches)

### Loss Functions

- **Adversarial Loss**:

$$L_{GAN}(G,D) = \mathbb{E}_{x,y}[\log D(x, y)] + \mathbb{E}_x[\log(1 - D(x, G(x)))]$$

- **L1 Reconstruction Loss**:

$$L_{L1}(G) = \mathbb{E}_{x,y}[||y - G(x)||_1]$$

- **Final Objective**:

$$G^* = \arg\min_G \max_D \ L_{GAN}(G,D) + \lambda \cdot L_{L1}(G)$$

where `λ = 100`

---

## 🧪 Implementation Details

| Parameter     | Value                    |
|---------------|--------------------------|
| Framework     | PyTorch                  |
| Epochs        | 100                      |
| Batch Size    | 4                        |
| Optimizer     | Adam (β1=0.5, β2=0.999)  |
| Learning Rate | 2e-4                     |
| Image Size    | 256×256                  |
| Model Saving  | Best checkpoints saved   |

---

## 📈 Results

<img src="outputs/output1.png" width="600"/>
<img src="outputs/output2.png" width="600"/>
<img src="outputs/output3.png" width="600"/>

**Success:**
- High-quality comic faces for well-lit and clear input images
- Preservation of identity and facial structure

**Challenges:**
- Artifacts in occluded faces or strong shadows
- Occasionally loses fine details (e.g., glasses)

---

## 🌐 Streamlit Web Application

A user-friendly web interface to try the model in real-time.

**Features:**
- Upload a real face image
- Model converts it to comic style
- Output displayed in real-time

To run locally:
```bash
streamlit run app.py
```

---

## ✅ Conclusion

A Pix2Pix-based system was built and deployed that learns to translate real human faces into artistic comic versions. The model preserves both content and artistic style, demonstrating the power of GANs for creative applications.

---

## 📚 References

1. Isola, P., Zhu, J.Y., Zhou, T., & Efros, A.A. (2017). *Image-to-image translation with conditional adversarial networks*. CVPR.
2. Nathan Nguyen. *Face2Comic Dataset*, Kaggle.
3. Goodfellow, I. et al. (2014). *Generative Adversarial Networks*. NeurIPS.
