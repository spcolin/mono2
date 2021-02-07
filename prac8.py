import torch
import torchvision
from lib.models.refine_module2 import *

depth=torch.randn(4,1,385,385)
image=torch.randn(4,3,385,385)

res=Residual(load_pretrained=False)
sam=Resample(load_pretrained=False)

res(torch.randn(4,4,385,385))
sam(torch.randn(4,4,385,385))


