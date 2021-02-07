import torch
import torch.nn as nn
import numpy as np


class RD_loss10(nn.Module):

    def __init__(self, span_list=[20,50,70], repeat=100):
        super(RD_loss10, self).__init__()
        self.span_list=span_list
        self.repeat = repeat

    def sub_block(self, tensor, pos):
        block = tensor[:, :, pos[0]:pos[1], pos[2]:pos[3]]

        return block

    def compute_rd(self, tensor,base_pos,rd_pos):

        base_block = self.sub_block(tensor, base_pos)
        rd_block=self.sub_block(tensor,rd_pos)

        return base_block-rd_block

    def forward(self, pred, gt):


        B, C, H, W = pred.shape

        loss = 0
        loss_fn = torch.nn.L1Loss(reduction='mean')

        for span in self.span_list:

            y_base = np.random.randint(0,span*2)
            x_base = np.random.randint(0,span*2)

            h_base=H-2*span
            w_base=W-2*span

            base_block_pos=[y_base,y_base+h_base,x_base,x_base+w_base]

            pred_rd = []
            gt_rd = []

            for i in range(80):
                # the sequence of computing relative depth map:
                # top,bottom,left,right,top_left,top_right,bottom_left,bottom_right

                y_rd = np.random.randint(0, span * 2)
                x_rd = np.random.randint(0, span * 2)

                # top,bottom,left,right
                rd_block_pos = [y_rd,
                                y_rd+h_base,
                                x_rd,
                                x_rd+w_base]

                pred_rd.append(self.compute_rd(pred, base_block_pos, rd_block_pos))
                gt_rd.append(self.compute_rd(gt, base_block_pos, rd_block_pos))

            pred_rd = torch.cat(pred_rd, 1).permute(0, 2, 3, 1)
            gt_rd = torch.cat(gt_rd, 1).permute(0, 2, 3, 1)

            pred_norm = torch.norm(pred_rd, 2, dim=3, keepdim=True)
            gt_norm = torch.norm(gt_rd, 2, dim=3, keepdim=True)

            pred_mask = pred_norm == 0
            gt_mask = gt_norm == 0

            pred_norm = pred_norm.masked_fill(pred_mask, value=1.0)
            gt_norm = gt_norm.masked_fill(gt_mask, value=1.0)

            pred_rd = pred_rd / pred_norm
            gt_rd = gt_rd / gt_norm

            loss = loss + loss_fn(pred_rd, gt_rd)


        return loss





