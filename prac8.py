import torch
import torchvision
from lib.models.refine_module3 import *

depth=torch.randn(4,1,385,385)
image=torch.randn(4,3,385,385)
features=torch.randn(4,2048,13,13)

res=Residual()
sam=Resample()

res(features)
# sam(features)


