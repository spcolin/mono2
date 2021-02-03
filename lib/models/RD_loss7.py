import torch
import torch.nn as nn
import numpy as np




class RD_loss7(nn.Module):

    def __init__(self,span=30,repeat=5):
        super(RD_loss7, self).__init__()
        self.span=span
        self.repeat=repeat


    def sub_block(self,tensor,pos):

        block=tensor[:,:,pos[0]:pos[1],pos[2]:pos[3]]

        return block

    def compute_rd(self,tensor):

        base_block = self.sub_block(tensor, self.base_block_pos)

        top_block = self.sub_block(tensor, self.top_block_pos)
        bottom_block = self.sub_block(tensor, self.bottom_block_pos)
        left_block = self.sub_block(tensor, self.left_block_pos)
        right_block = self.sub_block(tensor, self.right_block_pos)
        top_left_block = self.sub_block(tensor, self.top_left_block_pos)
        top_right_block = self.sub_block(tensor, self.top_right_block_pos)
        bottom_left_block = self.sub_block(tensor, self.bottom_left_block_pos)
        bottom_right_block = self.sub_block(tensor, self.bottom_right_block_pos)

        rd_top=base_block-top_block
        rd_bottom=base_block-bottom_block
        rd_left=base_block-left_block
        rd_right=base_block-right_block
        rd_top_left=base_block-top_left_block
        rd_top_right=base_block-top_right_block
        rd_bottom_left=base_block-bottom_left_block
        rd_bottom_right=base_block-bottom_right_block

        rd=torch.cat([rd_top,rd_bottom,rd_left,rd_right,rd_top_left,rd_top_right,rd_bottom_left,rd_bottom_right],1)

        return rd.permute(0,2,3,1)


    def forward(self,pred,gt):
        """
        compute the difference of relative depth map between predicted depth map and ground truth depth map
        :param pred: predicted depth map,B*1*H*W
        :param gt: ground truth depth map,B*1*H*W
        :return: difference of relative depth map between pred and gt
        """

        # the sequence of computing relative depth map:
        # top,bottom,left,right,top_left,top_right,bottom_left,bottom_right
        self.span_list = np.random.randint(1, self.span + 1, size=8)

        B,C,H,W=pred.shape

        # top,bottom,left,right
        self.base_block_pos=[self.span,H-self.span,self.span,W-self.span]

        self.top_block_pos=[self.base_block_pos[0]-self.span_list[0],self.base_block_pos[1]-self.span_list[0],
                            self.base_block_pos[2],self.base_block_pos[3]]
        self.bottom_block_pos=[self.base_block_pos[0]+self.span_list[1],self.base_block_pos[1]+self.span_list[1],
                               self.base_block_pos[2],self.base_block_pos[3]]
        self.left_block_pos=[self.base_block_pos[0],self.base_block_pos[1],
                             self.base_block_pos[2]-self.span_list[2],self.base_block_pos[3]-self.span_list[2]]
        self.right_block_pos=[self.base_block_pos[0],self.base_block_pos[1],
                              self.base_block_pos[2]+self.span_list[3],self.base_block_pos[3]+self.span_list[3]]
        self.top_left_block_pos=[self.base_block_pos[0]-self.span_list[4],self.base_block_pos[1]-self.span_list[4],
                                 self.base_block_pos[2]-self.span_list[4],self.base_block_pos[3]-self.span_list[4]]
        self.top_right_block_pos=[self.base_block_pos[0]-self.span_list[5],self.base_block_pos[1]-self.span_list[5],
                                  self.base_block_pos[2]+self.span_list[5],self.base_block_pos[3]+self.span_list[5]]
        self.bottom_left_block_pos=[self.base_block_pos[0]+self.span_list[6],self.base_block_pos[1]+self.span_list[6],
                                    self.base_block_pos[2]-self.span_list[6],self.base_block_pos[3]-self.span_list[6]]
        self.bottom_right_block_pos=[self.base_block_pos[0]+self.span_list[7],self.base_block_pos[1]+self.span_list[7]
                                ,self.base_block_pos[2]+self.span_list[7],self.base_block_pos[3]+self.span_list[7]]

        pred_rd=self.compute_rd(pred)
        gt_rd=self.compute_rd(gt)

        pred_norm=torch.norm(pred_rd,2,dim=3,keepdim=True)
        gt_norm=torch.norm(gt_rd,2,dim=3,keepdim=True)

        pred_mask = pred_norm == 0
        gt_mask = gt_norm == 0

        pred_norm = pred_norm.masked_fill(pred_mask, value=1.0)
        gt_norm = gt_norm.masked_fill(gt_mask, value=1.0)

        pred_rd=pred_rd/pred_norm
        gt_rd=gt_rd/gt_norm

        loss_fn=torch.nn.L1Loss(reduction='mean')

        loss = loss_fn(pred_rd, gt_rd)



        return loss





