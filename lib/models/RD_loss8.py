import torch
import torch.nn as nn
import numpy as np




class RD_loss8(nn.Module):

    def __init__(self,repeat=5):
        super(RD_loss8, self).__init__()
        self.repeat=repeat



    def compute_rd(self,tensor,seq_list):

        B=tensor.shape[0]

        reshaped_tensor=tensor.view(B,-1)

        shuffled_tensor=[]
        for i in range(B):
            shuffled_tensor.append(reshaped_tensor[i,seq_list[i]])

        shuffled_tensor=torch.stack(shuffled_tensor,0)

        rd=reshaped_tensor-shuffled_tensor

        return rd





    def forward(self,pred,gt):
        """
        compute the difference of relative depth map between predicted depth map and ground truth depth map
        :param pred: predicted depth map,B*1*H*W
        :param gt: ground truth depth map,B*1*H*W
        :return: difference of relative depth map between pred and gt
        """

        B,C,H,W=pred.shape

        map_size=H*W

        pred_rd=[]
        gt_rd=[]
        for i in range(self.repeat):
            seq_list = [torch.randperm(map_size) for i in range(B)]
            pred_rd.append(self.compute_rd(pred,seq_list))
            gt_rd.append(self.compute_rd(gt,seq_list))

        pred_rd=torch.stack(pred_rd,2)
        gt_rd=torch.stack(gt_rd,2)

        pred_norm = torch.norm(pred_rd, 2, dim=2, keepdim=True)
        gt_norm = torch.norm(gt_rd, 2, dim=2, keepdim=True)

        pred_mask = pred_norm == 0
        gt_mask = gt_norm == 0

        pred_norm = pred_norm.masked_fill(pred_mask, value=1.0)
        gt_norm = gt_norm.masked_fill(gt_mask, value=1.0)

        pred_rd = pred_rd / pred_norm
        gt_rd = gt_rd / gt_norm

        loss_fn = torch.nn.L1Loss(reduction='mean')

        loss = loss_fn(pred_rd, gt_rd)

        return loss





