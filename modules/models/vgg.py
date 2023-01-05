from torchvision.models import vgg11, VGG11_Weights
import torch.nn as nn
import torch

class VGG(nn.Module):
    def __init__(self, num_classes=4) -> None:
        super().__init__()
        self.model = vgg11(weights=("pretrained", VGG11_Weights))
        self.model.classifier[-1] = torch.nn.Linear(4096, num_classes, bias=True)
        
    def forward(self, x):
        return self.model(x) 
    
if __name__ == "__main__":
    from torchsummary import summary
    
    model = VGG().to("cuda")
    summary(model, (3, 512, 512))