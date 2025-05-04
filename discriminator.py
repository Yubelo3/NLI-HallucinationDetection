import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model=nn.ModuleList([
            nn.Linear(768,128),
            nn.Dropout(0.1),
            nn.Linear(128,3),
        ])
        self.softmax=nn.Softmax()

    def forward(self,x:torch.Tensor):
        for layer in self.model:
            x=layer(x)
            x=torch.relu(x)
        return self.softmax(x)