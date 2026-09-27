nse <div align="center">

# 🌿 Mulberry Leaf Cultivar Classifier

**Identify 10 mulberry cultivars from a single leaf photo, and see _why_ the model decided, with Grad-CAM and LIME.**

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mulberry-leaf-cultivar-classifier-10.streamlit.app/)
[![XAI](https://img.shields.io/badge/XAI-Grad--CAM%20%7C%20LIME-8A2BE2)](https://github.com/jacobgil/pytorch-grad-cam)
[![Best accuracy](https://img.shields.io/badge/best%20accuracy-99.20%25-brightgreen)](#-results)
[![Kaggle](https://img.shields.io/badge/Kaggle-notebooks-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/samirhossain2001/code)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

A Streamlit web app and training notebooks for classifying mulberry leaf cultivars with four CNNs: a custom 5-layer CNN plus fine-tuned ResNet-50, EfficientNet-B2 and VGG16. Upload a leaf image, pick a model, and get the top-3 predictions alongside five explainability visualizations.

**🚀 [Try the live demo →](https://mulberry-leaf-cultivar-classifier-10.streamlit.app/)**

## ✨ Features

- **Four models, one UI.** Switch between CustomCNN, ResNet-50, EfficientNet-B2 and VGG16 from the sidebar.
- **Top-3 predictions** with softmax confidence for every uploaded image.
- **Explainable AI.** See Grad-CAM, Grad-CAM++, Eigen-CAM and Ablation-CAM heatmaps, plus a LIME superpixel explanation.
- **One-click export.** Download all visualizations as a ZIP of PNGs.
- **Architecture viewer.** Inspect the full layer structure of the selected model.
- **Reproducible training.** The complete Kaggle notebooks for every model are in [`notebooks/`](notebooks/).

## 📊 Results

Test-split metrics as reported by each training notebook (weighted averages):

| Model               | Accuracy   | Precision | Recall | F1     | Epochs (early-stopped) | Notebook                                                                        |
| ------------------- | ---------- | --------- | ------ | ------ | ---------------------- | ------------------------------------------------------------------------------- |
| **EfficientNet-B2** | **99.20%** | 0.9921    | 0.9920 | 0.9920 | 20                     | [Kaggle](https://www.kaggle.com/code/samirhossain2001/mulberry-efficientnet-b2) |
| ResNet-50           | 98.67%     | 0.9869    | 0.9867 | 0.9867 | 16                     | [Kaggle](https://www.kaggle.com/code/samirhossain2001/mulberry-resnet50)        |
| CustomCNN           | 94.93%     | 0.9503    | 0.9493 | 0.9493 | 31                     | [Kaggle](https://www.kaggle.com/code/samirhossain2001/mulberry-customcnn)       |
| VGG16               | 92.33%     | 0.9239    | 0.9233 | 0.9229 | 25                     | [Kaggle](https://www.kaggle.com/code/samirhossain2001/mulberry-vgg16)           |

## 🗂️ Dataset & Training

The models are trained on the **Mulberry Leaf Dataset** (the `Original` split), which has 5,262 images across 10 cultivars. <!-- TODO: add a link to the dataset source -->

<details>
<summary>Class list and image counts</summary>

| Label | Cultivar           | Images |
| ----- | ------------------ | ------ |
| 0     | ChiangMai60        | 500    |
| 1     | RedKing            | 350    |
| 2     | WhiteKing          | 541    |
| 3     | BlackOodTurkey     | 500    |
| 4     | TaiwanStraberry    | 488    |
| 5     | BlackAustralia     | 637    |
| 6     | Buriram60          | 345    |
| 7     | Kamphaengsaeng42   | 500    |
| 8     | TaiwanMeacho       | 640    |
| 9     | ChiangMaiBuriram60 | 761    |

</details>

**Pipeline** (shared by all four notebooks):

- **Balancing:** each class is oversampled to 1,500 images, then split randomly into train/val/test (70/20/10).
- **Input:** images resized to 224 × 224 and normalized with mean/std 0.5. The notebooks also define an augmentation pipeline (`transform_train`).
- **Optimization:** Adam (lr = 1e-3), cross-entropy loss, mixed precision (AMP), up to 50 epochs with early stopping (patience 5).
- **Transfer learning:** ResNet-50, EfficientNet-B2 and VGG16 start from ImageNet weights, and their final classifier layer is replaced with a 10-class head.
- **Hardware:** Kaggle GPU runtime (Python 3.11).

## 📁 Project Structure

```
.
├── app.py              # Streamlit app: inference, CAM/LIME explanations, ZIP export
├── requirements.txt    # Pinned Python dependencies (CPU PyTorch)
├── packages.txt        # System packages for Streamlit Community Cloud (OpenCV needs libGL)
├── notebooks/          # Training + evaluation + XAI notebooks (one per model)
│   ├── mulberry-customcnn.ipynb
│   ├── mulberry-resnet50.ipynb
│   ├── mulberry-efficientnet-b2.ipynb
│   └── mulberry-vgg16.ipynb
└── LICENSE
```

## 🚀 Installation

**1. Clone the repository**

```bash
git clone https://github.com/SamirHossain2001/CSE366_Group_A_Streamlit_App.git
cd CSE366_Group_A_Streamlit_App
```

**2. Create and activate a virtual environment** (Python 3.11 or 3.12)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

> `requirements.txt` installs CPU-only PyTorch wheels. If you have an NVIDIA GPU, install the CUDA build of `torch`/`torchvision` from [pytorch.org](https://pytorch.org/get-started/locally/) instead.

**4. Model weights (automatic)**

The trained weights are hosted on the Hugging Face Hub at [2001Samir/mulberry-leaf-models](https://huggingface.co/2001Samir/mulberry-leaf-models). The app downloads the selected model the first time you use it and caches it locally, so there's nothing to set up.

To work offline, download the files (from the Hub or [Google Drive](https://drive.google.com/drive/folders/1sx0k4mZHHgPslZla67TbtffHHPeLSpBd?usp=drive_link)) and put them next to `app.py`. Local files take priority over the Hub.

| File                                    | Model           | Size   |
| --------------------------------------- | --------------- | ------ |
| `custom_cnn_model.pth`                  | CustomCNN       | 118 MB |
| `transfer_learning_resnet50.pth`        | ResNet-50       | 94 MB  |
| `transfer_learning_efficientnet_b2.pth` | EfficientNet-B2 | 31 MB  |
| `transfer_learning_vgg16.pth`           | VGG16           | 537 MB |

## ▶️ Usage

```bash
streamlit run app.py
```

1. Choose a model in the sidebar.
2. Upload a leaf image (`.jpg`, `.jpeg` or `.png`).
3. Review the predicted cultivar, the top-3 probabilities and the CAM/LIME heatmaps.
4. Click **Download ZIP of Visualizations** to save all explanation images.

> The app runs on CPU if no CUDA GPU is available. Ablation-CAM and LIME are the slowest steps on CPU.

## 🔁 Reproducing Training

Each notebook in [`notebooks/`](notebooks/) is self-contained and was written for Kaggle. It expects the dataset at `/kaggle/input/mulberry-leaf-dataset/Mulberry Leaf Dataset/Original` and a GPU. Running a notebook end to end trains the model, saves its `.pth` file, plots loss curves and a confusion matrix, and produces the CAM/LIME explanations.

Maintained by [@SamirHossain2001](https://github.com/SamirHossain2001).

## 📄 License

This project is licensed under the [MIT License](LICENSE).
