from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch.nn as nn
import torch

class EfficientNet(nn.Module):
    def __init__(self, num_classes=4) -> None:
        super().__init__()
        self.model = efficientnet_b0(weights=("pretrained", EfficientNet_B0_Weights))
        self.model.classifier[-1] = torch.nn.Linear(1280, num_classes, bias=True)
        
    def forward(self, x):
        return self.model(x)
    
if __name__ == "__main__":
    from torchsummary import summary
    
    model = EfficientNet().to("cuda")
    summary(model, (3, 512, 512))