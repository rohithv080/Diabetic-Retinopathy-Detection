import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd

from model import SwinDRModel
from data_module import get_dataloaders

def calculate_class_weights(df, label_col='diagnosis'):
    # The paper uses class weights inversely proportional to prevalence
    # 0 -> 0 (No DR), 1-4 -> 1 (DR)
    binary_labels = (df[label_col] > 0).astype(int)
    class_counts = binary_labels.value_counts().sort_index().values
    total = len(binary_labels)
    # wc = N / (2 * Nc)
    weights = total / (2.0 * class_counts)
    return torch.tensor(weights, dtype=torch.float32)

def train(epochs=20, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_save_path = os.path.join(os.path.dirname(__file__), "best_swin_model.pth")

    # 1. Setup DataLoaders
    train_csv = os.path.join(project_root, "data", "raw", "train_1.csv")
    valid_csv = os.path.join(project_root, "data", "raw", "valid.csv")
    train_img_dir = os.path.join(project_root, "data", "raw", "train_images")
    valid_img_dir = os.path.join(project_root, "data", "raw", "val_images")
    
    train_loader = get_dataloaders(train_csv, train_img_dir, batch_size=batch_size, shuffle=True)
    valid_loader = get_dataloaders(valid_csv, valid_img_dir, batch_size=batch_size, shuffle=False)
    
    # Calculate weights from training set
    df_train = pd.read_csv(train_csv)
    class_weights = calculate_class_weights(df_train).to(device)
    print(f"Class weights: {class_weights}")

    # 2. Setup Model, Optimizer, Loss
    model = SwinDRModel().to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    use_amp = (device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)
    
    # OneCycleLR Scheduler (warmup to max LR, then cool down)
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=3e-4, 
        steps_per_epoch=steps_per_epoch, 
        epochs=epochs,
        pct_start=0.3 # 30% warmup
    )
    
    # 3. Training Loop
    best_f1 = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Mixed Precision (enabled on CUDA)
            with autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            
            # Gradient clipping (from paper)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        # 4. Validation Loop
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]"):
                images, labels = images.to(device), labels.to(device)
                
                with autocast(enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        # Metrics
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        val_acc = accuracy_score(all_labels, all_preds)
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(valid_loader):.4f}")
        print(f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}\n")
        
        # Checkpointing
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), model_save_path)
            print(f"--> Saved new best model to {model_save_path} with F1: {best_f1:.4f}")

if __name__ == "__main__":
    train()
