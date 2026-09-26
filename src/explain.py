import os
import cv2
import numpy as np
import torch
import argparse
import random
import glob
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from model import SwinDRModel

def apply_clahe_rgb(image):
    # Convert RGB to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    # Merge back and convert to RGB
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def load_image(img_path):
    rgb_img = cv2.imread(img_path, 1)
    if rgb_img is None:
        raise FileNotFoundError(f"Could not load image at {img_path}")
    rgb_img = rgb_img[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255.0
    
    # Apply CLAHE
    rgb_img_clahe = apply_clahe_rgb(np.uint8(255 * rgb_img))
    rgb_img_clahe = np.float32(rgb_img_clahe) / 255.0
    
    # Match training preprocessing: [0, 1] scaled tensor without ImageNet z-normalization
    input_tensor = torch.from_numpy(rgb_img_clahe.transpose((2, 0, 1))).unsqueeze(0).float()
    return rgb_img_clahe, input_tensor

def swin_reshape_transform(x):
    # timm's Swin Transformer actually keeps the spatial dimensions as 2D!
    # The shape is (B, H, W, C), for example (1, 7, 7, 1024).
    # Grad-CAM expects standard CNN format: (B, C, H, W).
    if x.ndim == 4:
        return x.permute(0, 3, 1, 2)
    
    # Fallback just in case a different layer outputs a flattened (B, L, C) sequence
    result = x.reshape(x.size(0),
                       int(np.sqrt(x.size(1))),
                       int(np.sqrt(x.size(1))),
                       x.size(2))
    return result.permute(0, 3, 1, 2)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # 1. Load the trained model
    model = SwinDRModel(pretrained=False).to(device)
    model_path = os.path.join(os.path.dirname(__file__), "best_swin_model.pth")
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded trained weights from {model_path}")
    else:
        print(f"Warning: {model_path} not found. Using untrained model for demonstration.")
    
    model.eval()

    # 2. Target layer for Swin Transformer in timm
    # We target the normalization layer of the final block in the last layer
    target_layers = [model.backbone.layers[-1].blocks[-1].norm1]

    # 3. Initialize Grad-CAM (Now with the reshape transform!)
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=swin_reshape_transform)
    
    # Target class (1 = Diabetic Retinopathy)
    targets = [ClassifierOutputTarget(1)]

    # 4. Process a sample image
    default_sample = os.path.join(project_root, "data", "raw", "train_images", "1b329a127307.png")
    parser = argparse.ArgumentParser(description="Explain DR Detection Model")
    parser.add_argument('--image', type=str, default=default_sample,
                        help='Path to the image you want to test')
    parser.add_argument('--random', action='store_true', help='Pick a random image from the validation set')
    args = parser.parse_args()
    
    if args.random:
        # Get a list of all images in the validation directory
        val_dir = os.path.join(project_root, "data", "raw", "val_images")
        all_images = glob.glob(os.path.join(val_dir, "*.png")) + glob.glob(os.path.join(val_dir, "*.jpeg"))
        if len(all_images) > 0:
            sample_img_path = random.choice(all_images)
            print(f"[Random Mode] Selected image: {sample_img_path}")
        else:
            print(f"Could not find any images in {val_dir}")
            return
    else:
        sample_img_path = args.image 
    
    if not os.path.exists(sample_img_path):
        print(f"Sample image not found: {sample_img_path}. Please provide a valid path.")
        return

    rgb_img, input_tensor = load_image(sample_img_path)
    input_tensor = input_tensor.to(device)

    # 4.5 Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = torch.argmax(output, dim=1).item()
        class_name = "DR (Diabetic Retinopathy)" if pred_idx == 1 else "No DR"
        print(f"\n================================")
        print(f"Predicted Class: {class_name}")
        print(f"================================\n")

    # 5. Generate heatmap
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # Superimpose heatmap on original image
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # 6. Save/Display the result
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original (CLAHE)")
    plt.imshow(rgb_img)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Grad-CAM (DR Features)")
    plt.imshow(visualization)
    plt.axis('off')
    
    output_path = os.path.join(os.path.dirname(__file__), "gradcam_result.png")
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Successfully saved heatmap to {output_path}")

if __name__ == '__main__':
    main()
