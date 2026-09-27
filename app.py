import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import zipfile
import io
import os
from huggingface_hub import hf_hub_download
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus as GradCAMpp, EigenCAM, AblationCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from lime import lime_image
from skimage.segmentation import mark_boundaries
from torch.cuda.amp import autocast

# Define CLASS_NAMES globally
# Order must match torchvision ImageFolder's sorted folder order used in training ("01 ..." to "10 ...")
CLASS_NAMES = [
    "ChiangMai60",
    "RedKing",
    "WhiteKing",
    "BlackOodTurkey",
    "TaiwanStraberry",
    "BlackAustralia",
    "Buriram60",
    "Kamphaengsaeng42",
    "TaiwanMeacho",
    "ChiangMaiBuriram60"
]

# Hugging Face Hub repo that hosts the .pth checkpoints
HF_REPO_ID = "2001Samir/mulberry-leaf-models"

# Function to resolve a checkpoint: local copy next to app.py if present, otherwise download (cached) from the Hub
def get_checkpoint(filename):
    if os.path.exists(filename):
        return filename
    try:
        return hf_hub_download(repo_id=HF_REPO_ID, filename=filename)
    except Exception as e:
        st.error(f"Could not download {filename} from Hugging Face repo {HF_REPO_ID}: {e}")
        st.stop()

# CustomCNN definition
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

        self.convs = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self._conv_output_size = self._get_conv_output_size()

        self.fc1 = nn.Linear(self._conv_output_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, len(CLASS_NAMES))
        self.dropout = nn.Dropout(0.1)

    def _get_conv_output_size(self):
        with autocast():
            o = self.convs(torch.randn(1, 3, 224, 224))
            return int(np.prod(o.size()[1:]))

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._conv_output_size)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.dropout(torch.relu(self.fc3(x)))
        x = self.dropout(torch.relu(self.fc4(x)))
        x = self.fc5(x)
        return x

def load_custom_cnn(path, num_classes, device):
    model = CustomCNN().to(device)
    try:
        state_dict = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        st.error(f"Error loading CustomCNN checkpoint: {e}. Please ensure the checkpoint matches the 10-class model.")
        st.stop()
    model.eval()
    return model

def load_resnet50(path, num_classes, device):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    try:
        state_dict = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        st.error(f"Error loading ResNet-50 checkpoint: {e}")
        st.stop()
    model.to(device).eval()
    return model

def load_efficientnet_b2(path, num_classes, device):
    model = models.efficientnet_b2(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    try:
        state_dict = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        st.error(f"Error loading EfficientNet-B2 checkpoint: {e}")
        st.stop()
    model.to(device).eval()
    return model

def load_vgg16(path, num_classes, device):
    model = models.vgg16(pretrained=False)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    try:
        state_dict = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        st.error(f"Error loading VGG16 checkpoint: {e}")
        st.stop()
    model.to(device).eval()
    return model

# Function to get model architecture as string
def get_model_architecture(model):
    return str(model)

# Function to generate CAM heatmaps
def generate_cam(model, tensor, target_class, method, target_layers):
    if method == 'GradCAM':
        cam_extractor = GradCAM(model=model, target_layers=target_layers)
    elif method == 'GradCAMpp':
        cam_extractor = GradCAMpp(model=model, target_layers=target_layers)
    elif method == 'EigenCAM':
        cam_extractor = EigenCAM(model=model, target_layers=target_layers)
    elif method == 'AblationCAM':
        cam_extractor = AblationCAM(model=model, target_layers=target_layers)
    else:
        raise ValueError("Invalid CAM method")
    
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam_extractor(input_tensor=tensor, targets=targets)
    return grayscale_cam[0]

# Function to overlay heatmap on image
def overlay_heatmap(img, heatmap):
    heatmap_pil = Image.fromarray(np.uint8(heatmap * 255), 'L').resize(img.size, Image.BILINEAR)
    heatmap_resized = np.float32(np.array(heatmap_pil)) / 255
    rgb_img = np.float32(img) / 255
    result = show_cam_on_image(rgb_img, heatmap_resized, use_rgb=True)
    return Image.fromarray(np.uint8(result * 255))

# Function for LIME explanation
def lime_explain(model, img_array, num_samples=100):
    explainer = lime_image.LimeImageExplainer()
    def predict_fn(images):
        pil_images = [Image.fromarray(img).convert("RGB") for img in images]
        transformed_images = torch.stack([transform(img) for img in pil_images]).to(DEVICE)
        with torch.no_grad():
            return torch.softmax(model(transformed_images), dim=1).cpu().numpy()
    
    img_array = np.array(Image.fromarray(img_array).resize((224, 224)))
    explanation = explainer.explain_instance(img_array, predict_fn, top_labels=1, hide_color=0, num_samples=num_samples)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    lime_img = mark_boundaries(temp / 255.0, mask, color=(1, 1, 1))
    return Image.fromarray((lime_img * 255).astype(np.uint8))

# Streamlit app
st.set_page_config(page_title="Mulberry Leaf Classifier", layout="wide", initial_sidebar_state="expanded")

st.title("🌿 Mulberry Leaf Cultivar Classifier")
st.markdown("Upload an image of a mulberry leaf to classify its cultivar using advanced CNN models. Explore explainability with CAM methods and LIME!")

# Sidebar
with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox("Choose Model", ("Custom CNN", "ResNet-50", "EfficientNet-B2", "VGG16"))
    uploaded_file = st.file_uploader("Upload a Leaf Image", type=["jpg", "jpeg", "png"])
    st.markdown("---")
    st.subheader("Model Metadata")
    NUM_CLASSES = len(CLASS_NAMES)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    INPUT_SIZE = "224x224"
    if model_choice == "Custom CNN":
        CHECKPOINT_PATH = "custom_cnn_model.pth"
    elif model_choice == "ResNet-50":
        CHECKPOINT_PATH = "transfer_learning_resnet50.pth"
    elif model_choice == "EfficientNet-B2":
        CHECKPOINT_PATH = "transfer_learning_efficientnet_b2.pth"
    else:  # VGG16
        CHECKPOINT_PATH = "transfer_learning_vgg16.pth"
    st.write(f"**Architecture:** {model_choice}")
    st.write(f"**Input Size:** {INPUT_SIZE}")
    st.write(f"**Classes:** {', '.join(CLASS_NAMES)}")
    st.write(f"**Checkpoint Path:** {CHECKPOINT_PATH}")

MODEL_LOADERS = {
    "Custom CNN": load_custom_cnn,
    "ResNet-50": load_resnet50,
    "EfficientNet-B2": load_efficientnet_b2,
    "VGG16": load_vgg16
}

# Keep only the selected model in memory (all four together exceed free-tier RAM)
@st.cache_resource(max_entries=1, show_spinner="Loading model weights...")
def load_model(name, checkpoint_path):
    return MODEL_LOADERS[name](get_checkpoint(checkpoint_path), NUM_CLASSES, DEVICE)

model = load_model(model_choice, CHECKPOINT_PATH)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

if uploaded_file:
    # 1. Load & show original image
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)  # For LIME
    st.image(img, caption="Input Image", width=400)
    
    # 2. Preprocess
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    # 3. Inference with spinner
    if model_choice == "Custom CNN":
        target_layers = [model.convs[16]]  # Last Conv2d layer in convs
    elif model_choice == "ResNet-50":
        target_layers = [model.layer4[-1]]  # Last layer in ResNet-50's layer4
    elif model_choice == "EfficientNet-B2":
        target_layers = [model.features[-1]]  # Last conv layer in EfficientNet-B2
    else:  # VGG16
        target_layers = [model.features[-3]]  # Last conv layer in VGG16 (before maxpool)
    with st.spinner("Performing model inference..."):
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    # 4. Predictions
    top_idxs = np.argsort(probs)[::-1][:3]
    predicted_class = CLASS_NAMES[top_idxs[0]]
    st.subheader("Predictions")
    st.success(f"**Final Predicted Label:** {predicted_class} ({probs[top_idxs[0]]*100:.2f}%)")
    st.markdown("**Top-3 Classes:**")
    for idx in top_idxs:
        st.write(f"{CLASS_NAMES[idx]}: {probs[idx]*100:.2f}%")
    
    # 5. Model Architecture Details
    with st.expander("View Model Architecture"):
        st.text(get_model_architecture(model))
    
    # 6. Explainability Visualizations with spinner
    st.subheader("Explainability Visualizations")
    st.markdown("Heatmaps overlaid on the original image for interpretability.")
    
    target_class = top_idxs[0]  # Use top predicted class for CAMs
    
    methods = ['GradCAM', 'GradCAMpp', 'EigenCAM', 'AblationCAM']
    vis_images = {}
    
    # Row 1: CAM Methods with spinner
    with st.spinner("Generating CAM visualizations..."):
        cols = st.columns(4)  # Four columns for CAM methods
        for i, method in enumerate(methods):
            with cols[i]:
                heatmap = generate_cam(model, tensor, target_class, method, target_layers)
                overlaid = overlay_heatmap(img, heatmap)
                st.image(overlaid, caption=method, use_container_width=True)
                vis_images[method] = overlaid
    
    # Row 2: LIME with spinner
    st.markdown("---")  # Separator between rows
    with st.spinner("Generating LIME explanation..."):
        lime_img = lime_explain(model, img_array)
        st.image(lime_img, caption="LIME", width=400)
        vis_images['LIME'] = lime_img
    
    # 7. Download option
    st.subheader("Download Visualizations")
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for name, vis_img in vis_images.items():
            img_buffer = io.BytesIO()
            vis_img.save(img_buffer, format="PNG")
            zip_file.writestr(f"{name}.png", img_buffer.getvalue())
    zip_buffer.seek(0)
    st.download_button(
        label="Download ZIP of Visualizations",
        data=zip_buffer,
        file_name="visualizations.zip",
        mime="application/zip"
    )