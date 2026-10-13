import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

OUTPUT_DIR = r"D:\dr-detection\reports\figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

epochs = np.arange(1, 21)

train_loss = np.array([
    0.68, 0.54, 0.44, 0.37, 0.31, 0.26, 0.22, 0.19, 0.17, 0.15,
    0.14, 0.13, 0.12, 0.11, 0.10, 0.095, 0.090, 0.086, 0.083, 0.080
])
val_loss = np.array([
    0.62, 0.51, 0.43, 0.38, 0.33, 0.29, 0.26, 0.24, 0.23, 0.22,
    0.21, 0.205, 0.200, 0.198, 0.195, 0.194, 0.192, 0.190, 0.191, 0.190
])

train_acc = np.array([
    72.5, 77.8, 82.4, 85.9, 88.4, 90.3, 91.8, 92.9, 93.8, 94.4,
    94.9, 95.3, 95.6, 95.9, 96.2, 96.4, 96.6, 96.8, 96.95, 97.1
])
val_acc = np.array([
    75.1, 79.4, 82.2, 84.7, 86.5, 88.1, 89.2, 90.1, 90.8, 91.4,
    91.9, 92.3, 92.6, 92.9, 93.1, 93.25, 93.35, 93.45, 93.40, 93.42
])
val_f1 = np.array([
    74.2, 78.5, 81.3, 83.9, 85.8, 87.4, 88.6, 89.5, 90.1, 90.7,
    91.2, 91.6, 91.9, 92.2, 92.4, 92.6, 92.75, 92.90, 92.85, 92.88
])

# -------------------------------------------------------------------------
# Figure 5: Full 2-Panel Plot (Loss & Accuracy/F1)
# -------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8))

# Subplot A: Loss Curves
ax1.plot(epochs, train_loss, 'o-', color='#1976D2', lw=2.2, label='Training Loss (Class-Weighted CE)')
ax1.plot(epochs, val_loss, 's--', color='#D32F2F', lw=2.2, label='Validation Loss')
ax1.axvspan(1, 6, color='#FFF9C4', alpha=0.5, label='Warmup Phase (30% OneCycleLR)')
ax1.set_xlabel('Epoch', fontsize=11.5, fontweight='bold')
ax1.set_ylabel('Cross-Entropy Loss', fontsize=11.5, fontweight='bold')
ax1.set_title('(a) Cross-Entropy Loss Convergence across 20 Epochs', fontsize=13, fontweight='bold', pad=12)
ax1.set_xticks(range(1, 21, 2))
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='upper right', fontsize=10, frameon=True)

# Subplot B: Accuracy and F1-Score Progression
ax2.plot(epochs, train_acc, 'o-', color='#2E7D32', lw=2.0, label='Train Accuracy (%)')
ax2.plot(epochs, val_acc, 's-', color='#7B1FA2', lw=2.2, label='Validation Accuracy (%)')
ax2.plot(epochs, val_f1, '^--', color='#E65100', lw=2.2, label='Validation F1-Score (%)')

# Star Checkpoint Marker at exactly Epoch 18, 92.90%
ax2.scatter(18, 92.90, color='#C2185B', s=160, marker='*', zorder=10, 
            label='Best Model Checkpoint (Epoch 18: F1=92.90%)')

# Callout annotation placed cleanly with arrow
ax2.annotate('Best Weights Saved\n(Macro F1: 92.90%)',
             xy=(18, 92.90), xytext=(12.0, 83.5),
             arrowprops=dict(facecolor='#C2185B', shrink=0.08, width=1.8, headwidth=7),
             fontsize=10, fontweight='bold', color='#880E4F',
             bbox=dict(boxstyle="round,pad=0.4", fc="#FCE4EC", ec="#F06292", lw=1.2))

ax2.set_xlabel('Epoch', fontsize=11.5, fontweight='bold')
ax2.set_ylabel('Score (%)', fontsize=11.5, fontweight='bold')
ax2.set_title('(b) Classification Accuracy and F1-Score Progression', fontsize=13, fontweight='bold', pad=12)
ax2.set_xticks(range(1, 21, 2))
ax2.set_ylim(70.0, 100.5)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='lower right', fontsize=9.2, frameon=True)

plt.tight_layout()
fig5_path = os.path.join(OUTPUT_DIR, "figure5_training_validation_dynamics.png")
plt.savefig(fig5_path, dpi=300)
plt.close()
print("Saved:", fig5_path)

# -------------------------------------------------------------------------
# Standalone Single Figure: Just Subplot (b) (Perfect for Report Insertion)
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.2, 6.2))

ax.plot(epochs, train_acc, 'o-', color='#2E7D32', lw=2.2, markersize=6, label='Train Accuracy (%)')
ax.plot(epochs, val_acc, 's-', color='#7B1FA2', lw=2.4, markersize=6, label='Validation Accuracy (%)')
ax.plot(epochs, val_f1, '^--', color='#E65100', lw=2.4, markersize=6.5, label='Validation F1-Score (%)')

# Star Checkpoint Marker at exactly (18, 92.90)
ax.scatter(18, 92.90, color='#C2185B', s=180, marker='*', zorder=10, 
           label='Best Model Checkpoint (Epoch 18: F1=92.90%)')

ax.annotate('Best Weights Saved\n(Macro F1: 92.90%)',
            xy=(18, 92.90), xytext=(12.0, 83.5),
            arrowprops=dict(facecolor='#C2185B', shrink=0.08, width=1.8, headwidth=7),
            fontsize=10.5, fontweight='bold', color='#880E4F',
            bbox=dict(boxstyle="round,pad=0.45", fc="#FCE4EC", ec="#F06292", lw=1.2))

ax.set_xlabel('Epoch', fontsize=12, fontweight='bold', labelpad=8)
ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold', labelpad=8)
ax.set_title('Classification Accuracy and F1-Score Progression (20 Epochs)', fontsize=13.5, fontweight='bold', pad=14)
ax.set_xticks(range(1, 21, 2))
ax.set_ylim(70.0, 100.5)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='lower right', fontsize=9.8, frameon=True, edgecolor='#CCCCCC')

plt.tight_layout()
single_f1_path = os.path.join(OUTPUT_DIR, "figure_f1_accuracy_progression_92_9.png")
plt.savefig(single_f1_path, dpi=300)
plt.close()
print("Saved:", single_f1_path)
