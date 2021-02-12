import torch
import torch.nn as nn



class Refine_loss2(nn.Module):

    def __init__(self):
        super(Refine_loss2, self).__init__()


    def forward(self, pred, gt):

        gt_mask=gt<0.0001
        pred_mask=pred<0.0001

        mask=gt_mask|pred_mask

        masked_gt=gt.masked_fill(mask,value=torch.tensor(0.1))
        masked_pred=pred.masked_fill(mask,value=torch.tensor(0.1))


        d = torch.log(masked_pred) - torch.log(masked_gt)
        loss= torch.sqrt((d ** 2).mean() - 0.85 * (d.mean() ** 2)) * 10.0


        return loss





