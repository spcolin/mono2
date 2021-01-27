import torch
import torch.nn as nn
import numpy as np
from lib.models.RD_loss5 import RD_loss5



a=RD_loss5()

depth=torch.randn(4,1,385,385)

a(depth,depth)

print(a.pred_list.shape)

