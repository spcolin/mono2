import torch
import torch.nn as nn
import numpy as np



class RD_loss3(nn.Module):

    def __init__(self,range=3):
        super(RD_loss3, self).__init__()
        self.range=range

    def compute_left(self,tensor):
        """
        get the 0 to n-2 cols of tensor(n cols totally)
        :param tensor: B*1*H*W size
        :return: the 0 to n-2 cols of tensor,B*1*H*(W-1)
        """
        left=tensor[:,:,:,:-1]
        return left

    def compute_right(self,tensor):
        """
        get the 1 to n-1 cols of tensor(n cols totally)
        :param tensor: B*1*H*W size
        :return: the 1 to n-1 cols of tensor,B*1*H*(W-1)
        """
        right=tensor[:,:,:,1:]
        return right

    def compute_top(self,tensor):
        """
        get the 0 to n-2 rows of tensor(n rows totally)
        :param tensor: B*1*H*W size
        :return: the 0 to n-2 rows of tensor,B*1*(H-1)*W
        """
        top=tensor[:,:,:-1,:]
        return top

    def compute_bottom(self,tensor):
        """
        get the 1 to n-1 rows of tensor(n rows totally)
        :param tensor: B*1*H*W size
        :return: the 1 to n-1 rows of tensor,B*1*(H-1)*W
        """
        bottom=tensor[:,:,1:,:]
        return bottom

    def compute_left_top(self,tensor):
        """
        get the [0 to n-2]*[0 to n-2] of tensor
        :param tensor: B*1*H*W size
        :return: the [0 to n-2]*[0 to n-2] of tensor,B*1*(H-1)*(W-1)
        """
        left_top=tensor[:,:,:,:-1][:,:,:-1,:]
        return left_top

    def compute_right_top(self,tensor):
        """
        get the [0 to n-2]*[1 to n-1] of tensor
        :param tensor: B*1*H*W size
        :return: the [0 to n-2]*[1 to n-1] of tensor,B*1*(H-1)*(W-1)
        """
        right_top=tensor[:,:,:,1:][:,:,:-1,:]
        return right_top

    def compute_bottom_left(self,tensor):
        """
        get the [1 to n-1]*[0 to n-2] of tensor
        :param tensor: B*1*H*W size
        :return: [1 to n-1]*[0 to n-2] of tensor,B*1*(H-1)*(W-1)
        """
        bottom_left=tensor[:,:,1:,:][:,:,:,:-1]
        return bottom_left

    def compute_bottom_right(self,tensor):
        """
        get the [1 to n-1]*[1 to n-1] of tensor
        :param tensor: B*1*H*W size
        :return: [1 to n-1]*[1 to n-1] of tensor,B*1*(H-1)*(W-1)
        """
        bottom_right=tensor[:,:,1:,:][:,:,:,1:]
        return bottom_right


    def compute_rd_top_pred(self,b_tensor,t_tensor):
        """
        compute the relative depth between position[x,y] and position[x,y-1]
        :param b_tensor:bottom tensor,B*1*(H-1)*W
        :param t_tensor: top tensor,B*1*(H-1)*W
        :return: the relative depth map between position[x,y] and position[x,y-1]
        """
        depth_res_map=torch.abs(b_tensor-t_tensor)
        # added_depth_map = b_tensor + t_tensor
        # scaled_relative_depth_map = depth_res_map / added_depth_map

        # return scaled_relative_depth_map
        return depth_res_map

    def compute_rd_bottom_pred(self,t_tensor,b_tensor):
        """
        compute the relative depth between position[x,y] and position[x,y+1]
        :param t_tensor: top tensor,B*1*(H-1)*W
        :param b_tensor: bottom tensor,B*1*(H-1)*W
        :return: the relative depth map between position[x,y] and position[x,y+1]
        """
        depth_res_map=torch.abs(t_tensor-b_tensor)
        # added_depth_map = t_tensor + b_tensor
        # scaled_relative_depth_map = depth_res_map / added_depth_map

        # return scaled_relative_depth_map
        return depth_res_map

    def compute_rd_left_pred(self,l_tensor,r_tensor):
        """
        compute the relative depth between position[x,y] and position[x-1,y]
        :param l_tensor: left tensor,B*1*H*(W-1)
        :param r_tensor: right tensor,B*1*H*(W-1)
        :return: the relative depth map between position[x,y] and position[x-1,y]
        """
        depth_res_map=torch.abs(r_tensor-l_tensor)
        # added_depth_map = r_tensor + l_tensor
        # scaled_relative_depth_map = depth_res_map / added_depth_map

        # return scaled_relative_depth_map
        return depth_res_map

    def compute_rd_right_pred(self,l_tensor,r_tensor):
        """
        compute the relative depth between position[x,y] and position[x+1,y]
        :param l_tensor: left tensor,B*1*H*(W-1)
        :param r_tensor: right tensor,B*1*H*(W-1)
        :return: the relative depth map between position[x,y] and position[x+1,y]
        """
        depth_res_map=torch.abs(l_tensor-r_tensor)
        # added_depth_map = l_tensor + r_tensor
        # scaled_relative_depth_map = depth_res_map / added_depth_map

        # return scaled_relative_depth_map
        return depth_res_map

    def compute_rd_left_top_pred(self,l_t_tensor,b_r_tensor):
        """
        compute the relative depth between position[x,y] and position[x-1,y-1]
        :param l_t_tensor: left top tensor,B*1*(H-1)*(W-1)
        :param b_r_tensor: bottom right tensor,B*1*(H-1)*(W-1)
        :return: the relative depth map between position[x,y] and position[x-1,y-1]
        """
        depth_res_map=torch.abs(b_r_tensor-l_t_tensor)
        # added_depth_map = b_r_tensor + l_t_tensor
        # scaled_relative_depth_map = depth_res_map / added_depth_map

        # return scaled_relative_depth_map
        return depth_res_map

    def compute_rd_right_top_pred(self,r_t_tensor,b_l_tensor):
        """
        compute the relative depth between position[x,y] and position[x+1,y-1]
        :param r_t_tensor: right top tensor,B*1*(H-1)*(W-1)
        :param b_l_tensor: bottom left tensor,B*1*(H-1)*(W-1)
        :return: the relative depth map between position[x,y] and position[x+1,y-1]
        """
        depth_res_map=torch.abs(b_l_tensor-r_t_tensor)
        # added_depth_map = r_t_tensor + b_l_tensor
        # scaled_relative_depth = depth_res_map / added_depth_map

        # return scaled_relative_depth
        return depth_res_map

    def compute_rd_bottom_left_pred(self,b_l_tensor,r_t_tensor):
        """
        compute the relative depth between position[x,y] and position[x-1,y+1]
        :param b_l_tensor: bottom left tensor,B*1*(H-1)*(W-1)
        :param r_t_tensor: right top tensor,B*1*(H-1)*(W-1)
        :return: the relative depth map between position[x,y] and position[x-1,y+1]
        """
        depth_res_map=torch.abs(r_t_tensor-b_l_tensor)
        # added_depth_map = r_t_tensor + b_l_tensor
        # scaled_relative_depth = depth_res_map / added_depth_map

        # return scaled_relative_depth
        return depth_res_map

    def compute_rd_bottom_right_pred(self,b_r_tensor,l_t_tensor):
        """
        compute the relative depth between position[x,y] and position[x+1,y+1]
        :param b_r_tensor: bottom right tensor,B*1*(H-1)*(W-1)
        :param l_t_tensor: left top tensor,B*1*(H-1)*(W-1)
        :return: the relative depth map between position[x,y] and position[x+1,y+1]
        """
        depth_res_map=torch.abs(l_t_tensor-b_r_tensor)
        # added_depth_map = l_t_tensor + b_r_tensor
        # scaled_relative_depth = depth_res_map / added_depth_map

        # return scaled_relative_depth
        return depth_res_map


    def compute_rd_top_gt(self,b_tensor,t_tensor):
        """
        compute the relative depth between position[x,y] and position[x,y-1]
        :param b_tensor:bottom tensor,B*1*(H-1)*W
        :param t_tensor: top tensor,B*1*(H-1)*W
        :return: the relative depth map between position[x,y] and position[x,y-1]
        """
        depth_res_map = torch.abs(b_tensor - t_tensor)
        # added_depth_map = b_tensor + t_tensor
        # scaled_relative_depth_map = depth_res_map / added_depth_map

        # return scaled_relative_depth_map
        return depth_res_map

    def compute_rd_bottom_gt(self,t_tensor,b_tensor):
        """
        compute the relative depth between position[x,y] and position[x,y+1]
        :param t_tensor: top tensor,B*1*(H-1)*W
        :param b_tensor: bottom tensor,B*1*(H-1)*W
        :return: the relative depth map between position[x,y] and position[x,y+1]
        """
        depth_res_map = torch.abs(t_tensor - b_tensor)
        # added_depth_map = t_tensor + b_tensor
        # scaled_relative_depth_map = depth_res_map / added_depth_map

        # return scaled_relative_depth_map
        return depth_res_map

    def compute_rd_left_gt(self,l_tensor,r_tensor):
        """
        compute the relative depth between position[x,y] and position[x-1,y]
        :param l_tensor: left tensor,B*1*H*(W-1)
        :param r_tensor: right tensor,B*1*H*(W-1)
        :return: the relative depth map between position[x,y] and position[x-1,y]
        """
        # depth_res_map=torch.abs(r_tensor-l_tensor)
        depth_res_map = torch.abs(r_tensor - l_tensor)
        # added_depth_map = r_tensor + l_tensor
        # scaled_relative_depth_map = depth_res_map / added_depth_map

        # return scaled_relative_depth_map
        return depth_res_map

    def compute_rd_right_gt(self,l_tensor,r_tensor):
        """
        compute the relative depth between position[x,y] and position[x+1,y]
        :param l_tensor: left tensor,B*1*H*(W-1)
        :param r_tensor: right tensor,B*1*H*(W-1)
        :return: the relative depth map between position[x,y] and position[x+1,y]
        """
        # depth_res_map=torch.abs(l_tensor-r_tensor)
        depth_res_map = torch.abs(l_tensor - r_tensor)
        # added_depth_map = l_tensor + r_tensor
        # scaled_relative_depth_map = depth_res_map / added_depth_map

        # return scaled_relative_depth_map
        return depth_res_map

    def compute_rd_left_top_gt(self,l_t_tensor,b_r_tensor):
        """
        compute the relative depth between position[x,y] and position[x-1,y-1]
        :param l_t_tensor: left top tensor,B*1*(H-1)*(W-1)
        :param b_r_tensor: bottom right tensor,B*1*(H-1)*(W-1)
        :return: the relative depth map between position[x,y] and position[x-1,y-1]
        """
        # depth_res_map=torch.abs(b_r_tensor-l_t_tensor)
        depth_res_map = torch.abs(b_r_tensor - l_t_tensor)
        # added_depth_map = b_r_tensor + l_t_tensor
        # scaled_relative_depth_map = depth_res_map / added_depth_map

        # return scaled_relative_depth_map
        return depth_res_map

    def compute_rd_right_top_gt(self,r_t_tensor,b_l_tensor):
        """
        compute the relative depth between position[x,y] and position[x+1,y-1]
        :param r_t_tensor: right top tensor,B*1*(H-1)*(W-1)
        :param b_l_tensor: bottom left tensor,B*1*(H-1)*(W-1)
        :return: the relative depth map between position[x,y] and position[x+1,y-1]
        """
        # depth_res_map=torch.abs(b_l_tensor-r_t_tensor)
        depth_res_map = torch.abs(b_l_tensor - r_t_tensor)
        # added_depth_map = r_t_tensor + b_l_tensor
        # scaled_relative_depth = depth_res_map / added_depth_map

        # return scaled_relative_depth
        return depth_res_map

    def compute_rd_bottom_left_gt(self,b_l_tensor,r_t_tensor):
        """
        compute the relative depth between position[x,y] and position[x-1,y+1]
        :param b_l_tensor: bottom left tensor,B*1*(H-1)*(W-1)
        :param r_t_tensor: right top tensor,B*1*(H-1)*(W-1)
        :return: the relative depth map between position[x,y] and position[x-1,y+1]
        """
        # depth_res_map=torch.abs(r_t_tensor-b_l_tensor)
        depth_res_map = torch.abs(r_t_tensor - b_l_tensor)
        # added_depth_map = r_t_tensor + b_l_tensor
        # scaled_relative_depth = depth_res_map / added_depth_map

        # return scaled_relative_depth
        return depth_res_map

    def compute_rd_bottom_right_gt(self,b_r_tensor,l_t_tensor):
        """
        compute the relative depth between position[x,y] and position[x+1,y+1]
        :param b_r_tensor: bottom right tensor,B*1*(H-1)*(W-1)
        :param l_t_tensor: left top tensor,B*1*(H-1)*(W-1)
        :return: the relative depth map between position[x,y] and position[x+1,y+1]
        """
        # depth_res_map=torch.abs(l_t_tensor-b_r_tensor)
        depth_res_map = torch.abs(l_t_tensor - b_r_tensor)
        # added_depth_map = l_t_tensor + b_r_tensor
        # scaled_relative_depth = depth_res_map / added_depth_map

        # return scaled_relative_depth
        return depth_res_map


    def compute_rd_map_list_pred(self,depth_tensor):
        """
        compute all the relative depth map of depth_tensor
        :param depth_tensor: the original depth map,B*1*H*W
        :return: a list containing all the relative depth map of depth tensor,in the order of [top,right,bottom,left,left top,right top,bottom right,bottom left]
        """
        top = self.compute_top(depth_tensor)
        bottom = self.compute_bottom(depth_tensor)
        right = self.compute_right(depth_tensor)
        left = self.compute_left(depth_tensor)
        left_top = self.compute_left_top(depth_tensor)
        bottom_right = self.compute_bottom_right(depth_tensor)
        right_top = self.compute_right_top(depth_tensor)
        bottom_left = self.compute_bottom_left(depth_tensor)


        rd_top_pred = self.compute_rd_top_pred(bottom, top)
        rd_bottom_pred = self.compute_rd_bottom_pred(top, bottom)

        rd_right_pred = self.compute_rd_right_pred(left, right)
        rd_left_pred = self.compute_rd_left_pred(left, right)

        rd_left_top_pred = self.compute_rd_left_top_pred(left_top, bottom_right)
        rd_bottom_right_pred = self.compute_rd_bottom_right_pred(bottom_right, left_top)

        rd_right_top_pred = self.compute_rd_right_top_pred(right_top, bottom_left)
        rd_bottom_left_pred = self.compute_rd_bottom_left_pred(bottom_left, right_top)

        pred_rd_list=[rd_top_pred,rd_right_pred,rd_bottom_pred,rd_left_pred,rd_left_top_pred,rd_right_top_pred,rd_bottom_right_pred,rd_bottom_left_pred]


        return pred_rd_list

    def compute_rd_map_list_gt(self, depth_tensor):
        """
        compute all the relative depth map of depth_tensor
        :param depth_tensor: the original depth map,B*1*H*W
        :return: a list containing all the relative depth map of depth tensor,in the order of [top,right,bottom,left,left top,right top,bottom right,bottom left]
        """
        top = self.compute_top(depth_tensor)
        bottom = self.compute_bottom(depth_tensor)
        right = self.compute_right(depth_tensor)
        left = self.compute_left(depth_tensor)
        left_top = self.compute_left_top(depth_tensor)
        bottom_right = self.compute_bottom_right(depth_tensor)
        right_top = self.compute_right_top(depth_tensor)
        bottom_left = self.compute_bottom_left(depth_tensor)


        rd_top_gt = self.compute_rd_top_gt(bottom, top)
        rd_bottom_gt = self.compute_rd_bottom_gt(top, bottom)

        rd_right_gt = self.compute_rd_right_gt(left, right)
        rd_left_gt = self.compute_rd_left_gt(left, right)

        rd_left_top_gt = self.compute_rd_left_top_gt(left_top, bottom_right)
        rd_bottom_right_gt = self.compute_rd_bottom_right_gt(bottom_right, left_top)

        rd_right_top_gt = self.compute_rd_right_top_gt(right_top, bottom_left)
        rd_bottom_left_gt = self.compute_rd_bottom_left_gt(bottom_left, right_top)


        gt_rd_list = [rd_top_gt, rd_right_gt, rd_bottom_gt, rd_left_gt, rd_left_top_gt, rd_right_top_gt,
                      rd_bottom_right_gt, rd_bottom_left_gt]

        return gt_rd_list


    def forward(self,pred,gt):
        """
        compute the difference of relative depth map between predicted depth map and ground truth depth map
        :param pred: predicted depth map,B*1*H*W
        :param gt: ground truth depth map,B*1*H*W
        :return: difference of relative depth map between pred and gt
        """

        mean_pred = pred.mean(2, True).mean(3, True)
        mean_dt = gt.mean(2, True).mean(3, True)
        norm_pred = pred / mean_pred
        norm_gt = gt / mean_dt

        pred_rd_list = self.compute_rd_map_list_pred(norm_pred)
        gt_rd_list = self.compute_rd_map_list_gt(norm_gt)

        loss_fn = torch.nn.L1Loss(reduction='mean')
        # loss_fn=torch.nn.SmoothL1Loss(reduction='mean')
        # loss_fn=torch.nn.MSELoss(reduction='mean')
        loss=0
        for i in range(len(pred_rd_list)):
            rd=pred_rd_list[i]*torch.exp(-gt_rd_list[i])
            loss=loss+rd.mean()

        return loss





