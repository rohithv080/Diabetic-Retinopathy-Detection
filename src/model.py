import torch
import torch.nn as nn
import timm

class SwinDRModel(nn.Module):
    def __init__(self, model_name='swin_base_patch4_window7_224', num_classes=2, pretrained=True):
        super().__init__()
        
        # We use timm to create the Swin Transformer model.
        # Setting num_classes=2 automatically replaces the final classification head
        # for our binary task (No-DR vs DR).
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=num_classes
        )

    def forward(self, x):
        # The forward pass simply passes the image through the Swin backbone
        return self.backbone(x)
        
    def extract_features(self, x):
        # Useful for Grad-CAM and explainability later!
        return self.backbone.forward_features(x)

if __name__ == "__main__":
    # Quick test to ensure the model initializes correctly
    model = SwinDRModel()
    dummy_input = torch.randn(2, 3, 224, 224) # Batch size 2, 3 channels, 224x224 image
    output = model(dummy_input)
    
    print("Model initialized successfully!")
    print(f"Output shape (should be [2, 2]): {output.shape}")

