import torch 
import torch.nn as nn
from pdb import set_trace as stop


class BaseModel(nn.Module):
    def __init__(self, base_model):
        """ postprocess method for model
        Args:
            base_model(nn.Module): initial model
        """
        super(BaseModel, self).__init__()
        self.base_model = base_model

    def forward(self, x):
        bs, nc, c, h, w = x.size()
        out_crop = self.base_model(x.view(-1, c, h, w))
        out_crop = out_crop.softmax(dim=1)
        out = out_crop.view(bs, nc, -1).mean(dim=1)

        return out

