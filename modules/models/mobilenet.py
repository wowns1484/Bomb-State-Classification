from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import torch.nn as nn
import torch

class MobileNet(nn.Module):
    def __init__(self, num_classes=4) -> None:
        super().__init__()
        self.model = mobilenet_v2(weights=("pretrained", MobileNet_V2_Weights))
        self.model.classifier[-1] = torch.nn.Linear(1280, num_classes, bias=True)
        
    def forward(self, x):
        return self.model(x)
    
if __name__ == "__main__":
    from torchsummary import summary
    
    model = MobileNet().to("cuda")
    summary(model, (3, 512, 512))