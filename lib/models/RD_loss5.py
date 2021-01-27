import torch
import torch.nn as nn
import numpy as np
import math



class RD_loss5(nn.Module):

    def __init__(self):
        super(RD_loss5, self).__init__()
        x_pos = np.random.randint(30,350,size=1000)
        y_pos = np.random.randint(30,350,size=1000)

        self.point_pos=[[x_pos[i],y_pos[i]] for i in range(1000)]



    def compute_rd_list(self,pred,gt):

        pred_list = []
        gt_list = []

        for i in self.point_pos:
            pred_list.append(self.compute_rd(pred,i))
            gt_list.append(self.compute_rd(gt,i))

        return pred_list,gt_list

    def compute_rd(self,depth,pos):

        rd_top=depth[:,:,pos[1],pos[0]]-depth[:,:,pos[1]-1,pos[0]]
        rd_bottom=depth[:,:,pos[1],pos[0]]-depth[:,:,pos[1]+1,pos[0]]
        rd_left=depth[:,:,pos[1],pos[0]]-depth[:,:,pos[1],pos[0]-1]
        rd_right=depth[:,:,pos[1],pos[0]]-depth[:,:,pos[1],pos[0]+1]

        rd_left_top=depth[:,:,pos[1],pos[0]]-depth[:,:,pos[1]-1,pos[0]-1]
        rd_right_top=depth[:,:,pos[1],pos[0]]-depth[:,:,pos[1]-1,pos[0]+1]
        rd_left_bottom=depth[:,:,pos[1],pos[0]]-depth[:,:,pos[1]+1,pos[0]-1]
        rd_right_bottom=depth[:,:,pos[1],pos[0]]-depth[:,:,pos[1]+1,pos[0]+1]

        return torch.cat([rd_top,rd_bottom,rd_left,rd_right,rd_left_top,rd_right_top,rd_left_bottom,rd_right_bottom],1)

    def forward(self,pred,gt):
        """
        compute the difference of relative depth map between predicted depth map and ground truth depth map
        :param pred: predicted depth map,B*1*H*W
        :param gt: ground truth depth map,B*1*H*W
        :return: difference of relative depth map between pred and gt
        """

        pred_list,gt_list=self.compute_rd_list(pred,gt)

        pred_list=torch.cat(pred_list,0)
        gt_list=torch.cat(gt_list,0)

        pred_norm=torch.norm(pred_list,2,dim=1,keepdim=True)
        gt_norm=torch.norm(gt_list,2,dim=1,keepdim=True)

        pred_mask=pred_norm==0
        gt_mask=gt_norm==0

        pred_norm=pred_norm.masked_fill(pred_mask,value=1.0)
        gt_norm=gt_norm.masked_fill(gt_mask,value=1.0)

        pred_list=pred_list/pred_norm
        gt_list=gt_list/gt_norm

        loss_fn=torch.nn.L1Loss(reduction='mean')

        loss=loss_fn(pred_list,gt_list)

        return loss





