import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                             recall_score, roc_auc_score, confusion_matrix, 
                             cohen_kappa_score, matthews_corrcoef, roc_curve)
from tqdm import tqdm
from torch.cuda.amp import autocast

from model import SwinDRModel
from data_module import get_dataloaders

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # 1. Load Data
    valid_csv = "../data/raw/valid.csv"
    valid_img_dir = "../data/raw/val_images/"
    
    if not os.path.exists(valid_csv):
        print(f"Error: Could not find validation CSV at {valid_csv}")
        return
        
    valid_loader = get_dataloaders(valid_csv, valid_img_dir, batch_size=32)

    # 2. Load Model
    model = SwinDRModel(pretrained=False).to(device)
    model_path = "best_swin_model.pth"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded trained weights from {model_path}")
    else:
        print(f"Warning: {model_path} not found. Ensure you have trained the model first!")
        return

    model.eval()
    
    all_preds = []
    all_probs = []
    all_labels = []

    # 3. Get Predictions
    with torch.no_grad():
        for images, labels in tqdm(valid_loader, desc="Evaluating"):
            images = images.to(device)
            
            with autocast():
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)[:, 1] # Probability of Class 1 (DR)
                preds = torch.argmax(outputs, dim=1)
                
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 4. Calculate Metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    auc = roc_auc_score(all_labels, all_probs)
    kappa = cohen_kappa_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)

    print("\n" + "="*30)
    print("🏆 FINAL EVALUATION METRICS 🏆")
    print("="*30)
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"AUC:       {auc:.4f}")
    print(f"Kappa:     {kappa:.4f}")
    print(f"MCC:       {mcc:.4f}")
    print("="*30)

    # 5. Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No DR', 'DR'], yticklabels=['No DR', 'DR'])
    plt.title('Confusion Matrix - Swin Transformer')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig("confusion_matrix.png", bbox_inches='tight')
    plt.close()

    # 6. Plot ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png", bbox_inches='tight')
    plt.close()

    print("\nVisualizations saved: 'confusion_matrix.png' and 'roc_curve.png'")

if __name__ == "__main__":
    evaluate_model()
