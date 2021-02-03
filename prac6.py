import torch
from lib.models.RD_loss9 import RD_loss9



input =torch.randn(4,1,385,385)

rd_loss=RD_loss9()

rd_loss(input,input)