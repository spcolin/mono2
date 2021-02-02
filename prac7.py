import torch
from lib.models.RD_loss7 import RD_loss7
from lib.models.RD_loss6 import RD_loss6


rd_loss=RD_loss7(span=20)
rd_loss2=RD_loss6()

input=torch.randn(4,1,385,385)

rd_loss(input,input)