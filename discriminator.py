import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model=nn.ModuleList([
            nn.Linear(768,128),
            nn.Dropout(0.1),
            nn.Linear(128,3)
        ])


class BaseDiscriminator(object):
    def __init__(self) -> None:
        pass

    def get_similarity(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError()


class L2Discriminator(BaseDiscriminator):
    def __init__(self) -> None:
        pass

    def get_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_normalized = torch.nn.functional.normalize(x, 2, dim=-1)
        y_normalized = torch.nn.functional.normalize(y, 2, dim=-1)
        return (x_normalized*y_normalized).sum(dim=-1)
