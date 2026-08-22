# 🧠 AI Research Hub & Deep Learning Suite
### *Comprehensive Benchmarks, Multi-Modal Architectures, Medical Diagnostics & Computer Vision*

<div align="center">

[![GitHub Repo](https://img.shields.io/badge/GitHub-ebi19912%2FAI-181717?style=for-the-badge&logo=github)](https://github.com/ebi19912/AI)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/YOLO-v8%20%7C%20Vision-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

<br/>

**[ 🇬🇧 English Documentation ](#-project-overview) | [ 🇮🇷 راهنمای کامل فارسی (Persian) ](README_FA.md)**

</div>

---

## 📌 Project Overview

This repository serves as a **comprehensive Deep Learning and Artificial Intelligence research laboratory** curated and engineered by **[Rouhalah Ebrahimi](https://www.linkedin.com/in/rouhalah-ebrahimi/)**. It contains over 50 Jupyter Notebooks featuring rigorous comparative benchmarks, custom loss implementations, multi-backbone hybrid models, unsupervised representation learning, and domain-specific AI solutions.

### 🔬 Core Research Pillars
1. **Medical Diagnostic Imaging**: Multi-label thoracic disease diagnosis on the **NIH ChestX-ray14** dataset, melanoma segmentation via **U-Net**, and hybrid **Vision Transformer (ViT)** ensembles.
2. **Explainable AI (XAI) & Layer Interpretability**: PyTorch forward hooks and Keras intermediate activation visualizers for convolutional feature map inspection.
3. **Aerial Object Detection & UAV Tracking**: State-of-the-art **YOLOv8** (Nano, Extra-Large) fine-tuned for drone surveillance and the **VisDrone2019-DET** benchmark.
4. **Generative Modeling & Image Processing**: Deep Convolutional GANs (**DCGAN**), Semi-Supervised GANs (**SGAN**), Conditional GANs (**CGAN**), and grayscale-to-RGB colorization autoencoders.
5. **Biometric Security**: Convolutional neural network-based fingerprint enrollment, feature extraction, and matching authentication pipeline (**FVC2000**).
6. **Financial Time-Series Forecasting**: Recurrent neural networks (**GRU / LSTM**) and hyperparameter optimization via **Keras Tuner** for cryptocurrency (BTC/IRT, BNB/IRT) and stock index (Dow Jones) forecasting.
7. **Cybersecurity & Tabular ML**: Password strength categorization using ensemble classifiers (**XGBoost, Random Forest, AdaBoost, Stacking**).

---

## 🏗️ Technical Architecture & Research Domains

### 1. 🫁 Medical Image Multi-Label Classification & Diagnostics

#### 🎯 Problem Statement & Technical Challenges
Thoracic pathology detection presents three major deep learning hurdles:
- **Severe Class Imbalance**: Pathologies such as Hernia (<0.5%) and Pneumonia are heavily underrepresented compared to Normal/Infiltration scans.
- **Patient Data Leakage Prevention**: Multiple X-ray scans belong to identical patients. Splitting randomly by image index introduces data contamination. We enforce patient-level isolation using `Patient_ID` partitioning:
  $$	ext{Patients}(\mathcal{D}_{	ext{train}}) \cap 	ext{Patients}(\mathcal{D}_{	ext{val}}) \cap 	ext{Patients}(\mathcal{D}_{	ext{test}}) = \emptyset$$
- **Multi-Label Co-occurrence**: Patients frequently present overlapping conditions (e.g., Effusion + Atelectasis).

#### ⚖️ Weighted Binary Cross-Entropy (WBCE)
To balance the gradient contribution between rare positive instances and abundant negative instances, a class-weighted loss is defined:
$$\mathcal{L}_{	ext{WBCE}} = -rac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} \left[ w_{p, c} \cdot y_{i, c} \log(\hat{y}_{i, c}) + w_{n, c} \cdot (1 - y_{i, c}) \log(1 - \hat{y}_{i, c}) ight]$$
Where positive and negative class weights are dynamically computed from class frequency:
$$w_{p, c} = rac{N - N_c}{N}, \quad w_{n, c} = rac{N_c}{N}$$

#### 📊 Benchmarked Architectures:
- **Modern ConvNets**: `ConvNeXt` (Tiny, Small, Base, Large, XLarge)
- **Densely Connected Networks**: `DenseNet-169`, `DenseNet-201`
- **Residual Networks**: `ResNet-50`, `ResNet-101V2`, `ResNet-152`
- **Mobile & Efficient Architectures**: `EfficientNetV2-L`, `MobileNetV1`, `MobileNetV2`, `NASNetMobile`
- **Classic Baselines**: `VGG-16`, `VGG-19`

#### 🧬 Hybrid Networks & Vision Transformer Fusion
- **`VIT + ConvNeXtBase + VGG16`**: Multi-modal fusion combining Vision Transformers (global self-attention), ConvNeXt (depthwise 7x7 spatial kernels), and VGG16 (local low-level feature extraction), trained using **PyTorch Automatic Mixed Precision (AMP)**.
- **`ConvNeXtBase + VGG16`**: Multi-scale feature concatenation model optimized for Lung Cancer and multi-label thoracic pathology.
- **`6-Model Ensemble API`**: Unified multi-backbone fusion network integrating VGG16, VGG19, ResNet50, InceptionV3, DenseNet201, and MobileNet.

---

### 2. 🔬 Medical Image Semantic Segmentation (ISIC Skin Lesion)

Implemented in [`skin_lesion_segmentation_using_unet.ipynb`](https://colab.research.google.com/github/ebi19912/AI/blob/main/skin_lesion_segmentation_using_unet.ipynb):
- **Architecture**: Custom U-Net with contracting path (encoder), bottleneck, expansive path (decoder), and skip connections.
- **Loss Function**: Custom **Jaccard Distance Loss**:
  $$J(y, \hat{y}) = rac{\sum (y \cdot \hat{y}) + \epsilon}{\sum y + \sum \hat{y} - \sum (y \cdot \hat{y}) + \epsilon}, \quad \mathcal{L}_{	ext{Jaccard}} = 1 - J(y, \hat{y})$$
- **Metrics**: IoU (Intersection over Union), Dice Similarity Coefficient (DSC), Precision, Recall, Pixel Accuracy.
- **Post-Processing**: Morphological boundary enhancement filters for lesion delineation.

---

### 3. 🔍 Explainable AI (XAI) & Layer-wise Feature Maps

Implemented in [`check_Models_Layers_For_X_ray_Images_.ipynb`](https://colab.research.google.com/github/ebi19912/AI/blob/main/check_Models_Layers_For_X_ray_Images_.ipynb):
- Evaluates feature representation depth using **PyTorch forward hooks** and **Keras intermediate models**.
- Demonstrates how hierarchical neural layers transition from edge and texture detection to anatomical structures (ribs, cardiac silhouette) and localized pathological anomalies (consolidation, nodules).

---

### 4. 🚁 Aerial Object Detection & Drone Vision

Implemented in [`VisDrone_Detection.ipynb`](https://colab.research.google.com/github/ebi19912/AI/blob/main/VisDrone_Detection.ipynb), [`YOLO_Drone_Detection.ipynb`](https://colab.research.google.com/github/ebi19912/AI/blob/main/YOLO_Drone_Detection.ipynb), and [`YOLO_Drone_Detection_2.ipynb`](https://colab.research.google.com/github/ebi19912/AI/blob/main/YOLO_Drone_Detection_2.ipynb):
- **Models**: Ultralytics **YOLOv8n** (real-time edge inference) and **YOLOv8x** (high-accuracy deep detection).
- **Datasets**: **VisDrone2019-DET** benchmark and aerial drone surveillance datasets.
- **High-Resolution Multi-Scale Training**: Image sizes up to $1500 	imes 1500$ px to effectively detect micro-scale objects from high altitudes.

---

### 5. 🎨 Generative Models (GANs) & Image Processing

Implemented in [`All_gan.ipynb`](https://colab.research.google.com/github/ebi19912/AI/blob/main/All_gan.ipynb), [`Colorized.ipynb`](https://colab.research.google.com/github/ebi19912/AI/blob/main/Colorized.ipynb), and [`Image_Proc_Autoencoder.ipynb`](https://colab.research.google.com/github/ebi19912/AI/blob/main/Image_Proc_Autoencoder.ipynb):
- **DCGAN**: Deep Convolutional Generative Adversarial Network with LeakyReLU activations and transposed convolutions.
- **SGAN**: Semi-Supervised GAN utilizing unlabeled data alongside labeled datasets to enhance classifier accuracy.
- **CGAN**: Conditional GAN for targeted class-conditioned image synthesis.
- **Colorization Autoencoders**: Encoder-decoder architectures for automatic grayscale-to-RGB colorization on CelebA, Oxford Flowers, and Landscape datasets.
- **Image Processing Autoencoders**: Denoising and latent reconstruction for license plate numbers and satellite river images.

---

### 6. 🔐 Biometric Security & Fingerprint Recognition

Implemented in [`Fingerprint_CNN_English.ipynb`](https://colab.research.google.com/github/ebi19912/AI/blob/main/Fingerprint_CNN_English.ipynb) & [`Fingerprint_CNN_Persian.ipynb`](https://colab.research.google.com/github/ebi19912/AI/blob/main/Fingerprint_CNN_Persian.ipynb):
- **Benchmark**: FVC2000 DB4_B Fingerprint Dataset.
- **Pipeline**: Image normalization $	o$ Convolutional feature extraction $	o$ Embedding registration (`register_fingerprint`) $	o$ Similarity-based authentication (`authenticate_fingerprint`).
- Full bilingual English and Persian documentation and explanations.

---

### 7. 📈 Financial Time-Series Forecasting

Implemented in [`BTCIRT_Predict.ipynb`](https://colab.research.google.com/github/ebi19912/AI/blob/main/BTCIRT_Predict.ipynb), [`BNBIRT_Predict.ipynb`](https://colab.research.google.com/github/ebi19912/AI/blob/main/BNBIRT_Predict.ipynb), and [`DJPrediction.ipynb`](https://colab.research.google.com/github/ebi19912/AI/blob/main/DJPrediction.ipynb):
- **BTC/IRT & BNB/IRT**: Time-series sequential price prediction using Gated Recurrent Units (GRU).
- **Hyperparameter Optimization**: Automated tuning of GRU layer depth, unit count, and learning rates using **Keras Tuner**.
- **Dow Jones Industrial Average**: Multi-step stock price forecasting with LSTM and GRU networks.

---

### 8. 🛡️ Tabular ML & Cybersecurity

Implemented in [`iPassword_(fin).ipynb`](https://colab.research.google.com/github/ebi19912/AI/blob/main/iPassword_(fin).ipynb):
- Feature engineering from password character entropy, length, digit/symbol ratios.
- Comparative evaluation across **XGBoost, Random Forest, AdaBoost, Decision Tree, Logistic Regression, KNN, and Stacking Ensembles**.

---

## 📂 Complete File Catalog & Google Colab Links

| Category | Notebook / File | Primary Architecture / Focus | Benchmark Dataset | Google Colab |
| :--- | :--- | :--- | :--- | :---: |
| **Medical Vision** | `ConvNeXtBase_ChestClassification_wbce.ipynb` | ConvNeXt-Base + WBCE | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/ConvNeXtBase_ChestClassification_wbce.ipynb) |
| **Medical Vision** | `ConvNeXtTiny_ChestClassification_wbce.ipynb` | ConvNeXt-Tiny + WBCE | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/ConvNeXtTiny_ChestClassification_wbce.ipynb) |
| **Medical Vision** | `ConvNeXtSmall_ChestClassification_wbce.ipynb` | ConvNeXt-Small + WBCE | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/ConvNeXtSmall_ChestClassification_wbce.ipynb) |
| **Medical Vision** | `ConvNeXtLarge_ChestClassification_wbce.ipynb` | ConvNeXt-Large + WBCE | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/ConvNeXtLarge_ChestClassification_wbce.ipynb) |
| **Medical Vision** | `ConvNeXtXLarge_ChestClassification_wbce.ipynb` | ConvNeXt-XLarge + WBCE | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/ConvNeXtXLarge_ChestClassification_wbce.ipynb) |
| **Medical Vision** | `DenseNet169_ChestClassification_wbce.ipynb` | DenseNet-169 + WBCE | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/DenseNet169_ChestClassification_wbce.ipynb) |
| **Medical Vision** | `DenseNet201_ChestClassification_wbce.ipynb` | DenseNet-201 + WBCE | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/DenseNet201_ChestClassification_wbce.ipynb) |
| **Medical Vision** | `DenseNet201__ChestClassification_wbce.ipynb` | DenseNet-201 (run 2) + WBCE | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/DenseNet201__ChestClassification_wbce.ipynb) |
| **Medical Vision** | `ResNet101V2_ChestClassification_wbce.ipynb` | ResNet-101V2 + WBCE | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/ResNet101V2_ChestClassification_wbce.ipynb) |
| **Medical Vision** | `ResNet152_ChestClassification_wbce.ipynb` | ResNet-152 + WBCE | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/ResNet152_ChestClassification_wbce.ipynb) |
| **Medical Vision** | `EfficientNetV2L_ChestClassification_wbce.ipynb` | EfficientNetV2-L + WBCE | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/EfficientNetV2L_ChestClassification_wbce.ipynb) |
| **Medical Vision** | `MobileNet_ChestClassification_wbce.ipynb` | MobileNetV1 + WBCE | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/MobileNet_ChestClassification_wbce.ipynb) |
| **Medical Vision** | `MobileNetV2_ChestClassification_wbce.ipynb` | MobileNetV2 + WBCE | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/MobileNetV2_ChestClassification_wbce.ipynb) |
| **Medical Vision** | `NASNetMobile_ChestClassification_wbce.ipynb` | NASNetMobile + WBCE | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/NASNetMobile_ChestClassification_wbce.ipynb) |
| **Medical Vision** | `VGG16_ChestClassification_wbce.ipynb` | VGG-16 + WBCE | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/VGG16_ChestClassification_wbce.ipynb) |
| **Hybrids & Fusion** | `VIT+ConvNeXtBase+VGG16_Chest_final.ipynb` | ViT + ConvNeXt + VGG16 (PyTorch AMP) | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/VIT%2BConvNeXtBase%2BVGG16_Chest_final.ipynb) |
| **Hybrids & Fusion** | `ConvNeXtBase+VGG16_ChestClassification_wbce_Full.ipynb` | ConvNeXt-Base + VGG16 Dual Backbone | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/ConvNeXtBase%2BVGG16_ChestClassification_wbce_Full.ipynb) |
| **Hybrids & Fusion** | `Lung_Cancer_ConvNeXtBase+VGG16_ChestClassification_wbce_Full.ipynb` | ConvNeXt + VGG16 Lung Cancer | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/Lung_Cancer_ConvNeXtBase%2BVGG16_ChestClassification_wbce_Full.ipynb) |
| **Hybrids & Fusion** | `Copy_of_Lung_Cancer_ConvNeXtBase+VGG16_ChestClassification_wbce_Full.ipynb` | ConvNeXt + VGG16 Lung Cancer (Copy) | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/Copy_of_Lung_Cancer_ConvNeXtBase%2BVGG16_ChestClassification_wbce_Full.ipynb) |
| **Hybrids & Fusion** | `MobileNet+VGG16_ChestClassification_wbce.ipynb` | MobileNet + VGG16 Fusion | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/MobileNet%2BVGG16_ChestClassification_wbce.ipynb) |
| **Hybrids & Fusion** | `API_model_with_six_models_Copy.ipynb` | 6-Model Ensemble Pipeline | Chest X-Ray | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/API_model_with_six_models_Copy.ipynb) |
| **Segmentation** | `skin_lesion_segmentation_using_unet.ipynb` | U-Net Semantic Segmentation (Jaccard Loss) | ISIC Skin Lesion | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/skin_lesion_segmentation_using_unet.ipynb) |
| **Explainable AI** | `check_Models_Layers_For_X_ray_Images_.ipynb` | Layer-wise Feature Maps & Activation Hooks | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/check_Models_Layers_For_X_ray_Images_.ipynb) |
| **Explainable AI** | `Copy_of_check_Models_Layers_For_X_ray_Images_.ipynb` | Layer-wise Feature Maps (Copy) | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/Copy_of_check_Models_Layers_For_X_ray_Images_.ipynb) |
| **Autoencoders** | `AutoEncoder_ConvNeXtBase_ChestClassification_wbce.ipynb` | ConvNeXt-Base Autoencoder | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/AutoEncoder_ConvNeXtBase_ChestClassification_wbce.ipynb) |
| **Autoencoders** | `AutoEncoder_Vgg16_ChestXray_NIH.ipynb` | VGG-16 Autoencoder Reconstruction | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/AutoEncoder_Vgg16_ChestXray_NIH.ipynb) |
| **Autoencoders** | `AutoEncoder_vgg19_ChestClassification_wbce.ipynb` | VGG-19 Autoencoder Reconstruction | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/AutoEncoder_vgg19_ChestClassification_wbce.ipynb) |
| **Autoencoders** | `autoencoder_vgg19_chestxray_nih.ipynb` | VGG-19 Autoencoder NIH | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/autoencoder_vgg19_chestxray_nih.ipynb) |
| **Autoencoders** | `vgg19_autoencoder_chest_xray_auc.ipynb` | VGG-19 Autoencoder AUC Analysis | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/vgg19_autoencoder_chest_xray_auc.ipynb) |
| **Autoencoders** | `Autoencoder_chest_xray_Full.ipynb` | Full Chest X-Ray Autoencoder | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/Autoencoder_chest_xray_Full.ipynb) |
| **Autoencoders** | `Copy_of_chest_xray_Autoencoder.ipynb` | Chest X-Ray Autoencoder (Copy) | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/Copy_of_chest_xray_Autoencoder.ipynb) |
| **Autoencoders** | `MobileNetVGG16Autoencoder_DffrntLoss_ChestMultiLBL.ipynb` | MobileNet+VGG16 Dual Autoencoder | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/MobileNetVGG16Autoencoder_DffrntLoss_ChestMultiLBL.ipynb) |
| **Drone Detection** | `VisDrone_Detection.ipynb` | YOLOv8n Aerial Surveillance | VisDrone2019-DET | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/VisDrone_Detection.ipynb) |
| **Drone Detection** | `YOLO_Drone_Detection.ipynb` | YOLOv8n Custom Drone Tracking | Aerial Drone Dataset | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/YOLO_Drone_Detection.ipynb) |
| **Drone Detection** | `YOLO_Drone_Detection_2.ipynb` | YOLOv8x High-Res Drone Detection | Aerial Drone Dataset | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/YOLO_Drone_Detection_2.ipynb) |
| **GANs & Processing** | `All_gan.ipynb` | DCGAN, SGAN, CGAN Generative Models | MNIST / Synthetic | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/All_gan.ipynb) |
| **GANs & Processing** | `Colorized.ipynb` | Image Colorization Autoencoder | CelebA / Landscapes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/Colorized.ipynb) |
| **GANs & Processing** | `colorization_autoencoder.ipynb` | Grayscale to RGB Autoencoder | Celebrity Faces | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/colorization_autoencoder.ipynb) |
| **GANs & Processing** | `colorized_image.ipynb` | Flowers Colorization CNN | Flowers Dataset | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/colorized_image.ipynb) |
| **GANs & Processing** | `Image_Proc_Autoencoder.ipynb` | License Plates & River Denoising | Multimodal | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/Image_Proc_Autoencoder.ipynb) |
| **GANs & Processing** | `VGG19_Autoencoder.ipynb` | VGG-19 Latent Autoencoder | Custom Image Data | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/VGG19_Autoencoder.ipynb) |
| **Biometrics** | `Fingerprint_CNN_English.ipynb` | CNN Fingerprint Authentication (EN) | FVC2000 DB4_B | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/Fingerprint_CNN_English.ipynb) |
| **Biometrics** | `Fingerprint_CNN_Persian.ipynb` | CNN Fingerprint Authentication (FA) | FVC2000 DB4_B | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/Fingerprint_CNN_Persian.ipynb) |
| **Signal Processing** | `PatternRecognition4th_2009_6_Chapter_Final.ipynb` | DCT, DST & Haar Wavelet Transform | Image Benchmarks | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/PatternRecognition4th_2009_6_Chapter_Final.ipynb) |
| **Time-Series** | `BTCIRT_Predict.ipynb` | GRU Bitcoin Price Forecasting | BTC/IRT Historical | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/BTCIRT_Predict.ipynb) |
| **Time-Series** | `BNBIRT_Predict.ipynb` | GRU + Keras Tuner Hyperparameter Optimization | BNB/IRT Historical | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/BNBIRT_Predict.ipynb) |
| **Time-Series** | `DJPrediction.ipynb` | LSTM & GRU Stock Index Forecasting | Dow Jones Index | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/DJPrediction.ipynb) |
| **Cybersecurity** | `iPassword_(fin).ipynb` | XGBoost & Stacking Classifier | Password Strength Data | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/iPassword_(fin).ipynb) |
| **Cybersecurity** | `Copy_of_Password0.ipynb` | Password Strength Classifier (Copy) | Password Strength Data | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/Copy_of_Password0.ipynb) |
| **GUI & Simulation** | `DiceGameSimulatorGUI.ipynb` | Tkinter Monte Carlo Simulation GUI | Synthetic Simulation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/DiceGameSimulatorGUI.ipynb) |
| **Comparative Analytics** | `Untitled1.ipynb` | Chest X-Ray Pipeline Exploration | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/Untitled1.ipynb) |
| **Comparative Analytics** | `Untitled2.ipynb` | Model Comparison Across Patient Cohorts | NIH ChestX-ray14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/Untitled2.ipynb) |
| **Comparative Analytics** | `Untitled4.ipynb` | F1-Score Comparative Visualization | Model Benchmark Metrics | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ebi19912/AI/blob/main/Untitled4.ipynb) |

---

## 🚀 Getting Started

### ☁️ Option 1: Execute on Google Colab (Recommended)
1. Select any notebook from the table above and click its **Open in Colab** badge.
2. In Google Colab, select **Runtime > Change runtime type > GPU (T4 / A100)**.
3. If downloading datasets from Kaggle, upload your `kaggle.json` API credential key when prompted.

### 💻 Option 2: Local Environment Setup

```bash
# 1. Clone the repository
git clone https://github.com/ebi19912/AI.git
cd AI

# 2. Set up a virtual environment
python -m venv venv

# Windows:
.\venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

# 3. Install core dependencies
pip install tensorflow torch torchvision torchaudio ultralytics opencv-python scikit-learn pandas numpy matplotlib seaborn jupyter keras-tuner xgboost
```

---

## 👤 Author & Contact

**Rouhalah Ebrahimi**
- **GitHub**: [@ebi19912](https://github.com/ebi19912)
- **LinkedIn**: [Rouhalah Ebrahimi](https://www.linkedin.com/in/rouhalah-ebrahimi/)
- **Repository Link**: [https://github.com/ebi19912/AI](https://github.com/ebi19912/AI)

---

## 📜 License
This project is open-source and released under the [MIT License](LICENSE).
