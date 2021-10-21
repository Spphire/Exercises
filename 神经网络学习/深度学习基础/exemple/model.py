import torch
import torch.nn as nn
from copy import deepcopy


class FcModelMan(nn.Module):
    def __init__(self):
        super(FcModelMan, self).__init__()

        self.f = nn.Flatten()
        self.l1 = nn.Linear(28*28, 256)
        self.s1 = nn.SiLU()
        self.l2 = nn.Linear(256, 512)
        self.s2 = nn.SiLU()
        self.l3 = nn.Linear(512, 256)
        self.s3 = nn.SiLU()
        self.l4 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.f(x)
        x = self.l1(x)
        x = self.s1(x)
        x = self.l2(x)
        x = self.s2(x)
        x = self.l3(x)
        x = self.s3(x)
        x = self.l4(x)
        return x


class FcModel(nn.Module):
    def __init__(self):
        super(FcModel, self).__init__()
        self.model = nn.Sequential(             # Bx1x28x28
            nn.Flatten(),                       # Bx784
            nn.Linear(28*28, 256), nn.SiLU(),   # Bx256
            nn.Linear(256, 512),   nn.SiLU(),   # Bx512
            nn.Linear(512, 256),   nn.SiLU(),   # Bx256
            nn.Linear(256, 10),                 # Bx10
        )
    
    def forward(self, x):
        return self.model(x)


class CnnModel(nn.Module):
    def __init__(self):
        super(CnnModel, self).__init__()
        self.model = nn.Sequential(                 # Bx1x28x28
            nn.Conv2d(1, 8, 3, 1, 1), nn.SiLU(),    # Bx8x28x28
            nn.MaxPool2d(2, 2),                     # Bx8x14x14
            nn.Conv2d(8, 32, 3, 1, 1), nn.SiLU(),   # Bx32x14x14
            nn.MaxPool2d(2, 2),                     # Bx32x7x7
            nn.Conv2d(32, 64, 3, 1, 1), nn.SiLU(),  # Bx64x7x7
            nn.MaxPool2d(7, 7),                     # Bx64x1x1
            nn.Conv2d(64, 10, 1, 1),                # Bx10x1x1
            nn.Flatten(),                           # Bx10
        )

    def forward(self, x):
        return self.model(x)


class EmaModel(nn.Module):
    def __init__(self, model, alpha=0.999):
        super(EmaModel, self).__init__()
        self.model: nn.Module = deepcopy(model)
        self.alpha = alpha
    
    def update(self, model):
        assert type(self.model) == type(model)
        for n, p in self.model.named_parameters():
            p.data = model.get_parameter(n).data * self.alpha + p.data * (1 - self.alpha)
            
    
    def forward(self, x):
        return self.model(x)
            

def init_model(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if hasattr(m, "bias"):
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if hasattr(m, "bias"):
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

