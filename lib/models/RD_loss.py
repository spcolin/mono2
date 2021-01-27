import torch
import torch.nn as nn
import numpy as np



class RD_loss(nn.Module):

    def __init__(self,range=3):
        super(RD_loss, self).__init__()
        self.range=range


    def compute_left(self,tensor,range):
        """
        get the 0 to n-2 cols of tensor(n cols totally)
        :param tensor: B*1*H*W size
        :return: the 0 to n-2 cols of tensor,B*1*H*(W-1)
        """
        left=tensor[:,:,:,:-range]
        return left

    def compute_right(self,tensor,range):
        """
        get the 1 to n-1 cols of tensor(n cols totally)
        :param tensor: B*1*H*W size
        :return: the 1 to n-1 cols of tensor,B*1*H*(W-1)
        """
        right=tensor[:,:,:,range:]
        return right

    def compute_top(self,tensor,range):
        """
        get the 0 to n-2 rows of tensor(n rows totally)
        :param tensor: B*1*H*W size
        :return: the 0 to n-2 rows of tensor,B*1*(H-1)*W
        """
        top=tensor[:,:,:-range,:]
        return top

    def compute_bottom(self,tensor,range):
        """
        get the 1 to n-1 rows of tensor(n rows totally)
        :param tensor: B*1*H*W size
        :return: the 1 to n-1 rows of tensor,B*1*(H-1)*W
        """
        bottom=tensor[:,:,range:,:]
        return bottom

    def compute_left_top(self,tensor,range):
        """
        get the [0 to n-2]*[0 to n-2] of tensor
        :param tensor: B*1*H*W size
        :return: the [0 to n-2]*[0 to n-2] of tensor,B*1*(H-1)*(W-1)
        """
        left_top=tensor[:,:,:,:-range][:,:,:-range,:]
        return left_top

    def compute_right_top(self,tensor,range):
        """
        get the [0 to n-2]*[1 to n-1] of tensor
        :param tensor: B*1*H*W size
        :return: the [0 to n-2]*[1 to n-1] of tensor,B*1*(H-1)*(W-1)
        """
        right_top=tensor[:,:,:,range:][:,:,:-range,:]
        return right_top

    def compute_bottom_left(self,tensor,range):
        """
        get the [1 to n-1]*[0 to n-2] of tensor
        :param tensor: B*1*H*W size
        :return: [1 to n-1]*[0 to n-2] of tensor,B*1*(H-1)*(W-1)
        """
        bottom_left=tensor[:,:,range:,:][:,:,:,:-range]
        return bottom_left

    def compute_bottom_right(self,tensor,range):
        """
        get the [1 to n-1]*[1 to n-1] of tensor
        :param tensor: B*1*H*W size
        :return: [1 to n-1]*[1 to n-1] of tensor,B*1*(H-1)*(W-1)
        """
        bottom_right=tensor[:,:,range:,:][:,:,:,range:]
        return bottom_right

    def compute_rd_top(self,b_tensor,t_tensor):
        """
        compute the relative depth between position[x,y] and position[x,y-1]
        :param b_tensor:bottom tensor,B*1*(H-1)*W
        :param t_tensor: top tensor,B*1*(H-1)*W
        :return: the relative depth map between position[x,y] and position[x,y-1]
        """
        depth_res_map=b_tensor-t_tensor
        added_depth_map=b_tensor+t_tensor
        scaled_relative_depth_map=depth_res_map/added_depth_map

        return scaled_relative_depth_map
        # return depth_res_map

    def compute_rd_bottom(self,t_tensor,b_tensor):
        """
        compute the relative depth between position[x,y] and position[x,y+1]
        :param t_tensor: top tensor,B*1*(H-1)*W
        :param b_tensor: bottom tensor,B*1*(H-1)*W
        :return: the relative depth map between position[x,y] and position[x,y+1]
        """
        depth_res_map=t_tensor-b_tensor
        added_depth_map=t_tensor+b_tensor
        scaled_relative_depth_map=depth_res_map/added_depth_map

        return scaled_relative_depth_map
        # return depth_res_map

    def compute_rd_left(self,l_tensor,r_tensor):
        """
        compute the relative depth between position[x,y] and position[x-1,y]
        :param l_tensor: left tensor,B*1*H*(W-1)
        :param r_tensor: right tensor,B*1*H*(W-1)
        :return: the relative depth map between position[x,y] and position[x-1,y]
        """
        depth_res_map=r_tensor-l_tensor
        added_depth_map=r_tensor+l_tensor
        scaled_relative_depth_map=depth_res_map/added_depth_map

        return scaled_relative_depth_map
        # return depth_res_map

    def compute_rd_right(self,l_tensor,r_tensor):
        """
        compute the relative depth between position[x,y] and position[x+1,y]
        :param l_tensor: left tensor,B*1*H*(W-1)
        :param r_tensor: right tensor,B*1*H*(W-1)
        :return: the relative depth map between position[x,y] and position[x+1,y]
        """
        depth_res_map=l_tensor-r_tensor
        added_depth_map=l_tensor+r_tensor
        scaled_relative_depth_map=depth_res_map/added_depth_map

        return scaled_relative_depth_map
        # return depth_res_map

    def compute_rd_left_top(self,l_t_tensor,b_r_tensor):
        """
        compute the relative depth between position[x,y] and position[x-1,y-1]
        :param l_t_tensor: left top tensor,B*1*(H-1)*(W-1)
        :param b_r_tensor: bottom right tensor,B*1*(H-1)*(W-1)
        :return: the relative depth map between position[x,y] and position[x-1,y-1]
        """
        depth_res_map=b_r_tensor-l_t_tensor
        added_depth_map=b_r_tensor+l_t_tensor
        scaled_relative_depth_map=depth_res_map/added_depth_map

        return scaled_relative_depth_map
        # return depth_res_map

    def compute_rd_right_top(self,r_t_tensor,b_l_tensor):
        """
        compute the relative depth between position[x,y] and position[x+1,y-1]
        :param r_t_tensor: right top tensor,B*1*(H-1)*(W-1)
        :param b_l_tensor: bottom left tensor,B*1*(H-1)*(W-1)
        :return: the relative depth map between position[x,y] and position[x+1,y-1]
        """
        depth_res_map=b_l_tensor-r_t_tensor
        added_depth_map=r_t_tensor+b_l_tensor
        scaled_relative_depth=depth_res_map/added_depth_map

        return scaled_relative_depth
        # return depth_res_map

    def compute_rd_bottom_left(self,b_l_tensor,r_t_tensor):
        """
        compute the relative depth between position[x,y] and position[x-1,y+1]
        :param b_l_tensor: bottom left tensor,B*1*(H-1)*(W-1)
        :param r_t_tensor: right top tensor,B*1*(H-1)*(W-1)
        :return: the relative depth map between position[x,y] and position[x-1,y+1]
        """
        depth_res_map=r_t_tensor-b_l_tensor
        added_depth_map=r_t_tensor+b_l_tensor
        scaled_relative_depth=depth_res_map/added_depth_map

        return scaled_relative_depth
        # return depth_res_map

    def compute_rd_bottom_right(self,b_r_tensor,l_t_tensor):
        """
        compute the relative depth between position[x,y] and position[x+1,y+1]
        :param b_r_tensor: bottom right tensor,B*1*(H-1)*(W-1)
        :param l_t_tensor: left top tensor,B*1*(H-1)*(W-1)
        :return: the relative depth map between position[x,y] and position[x+1,y+1]
        """
        depth_res_map=l_t_tensor-b_r_tensor
        added_depth_map=l_t_tensor+b_r_tensor
        scaled_relative_depth=depth_res_map/added_depth_map

        return scaled_relative_depth
        # return depth_res_map

    def compute_rd_map_list(self,depth_tensor,range_list):
        """
        compute all the relative depth map of depth_tensor
        :param depth_tensor: the original depth map,B*1*H*W
        :return: a list containing all the relative depth map of depth tensor,in the order of [top,right,bottom,left,left top,right top,bottom right,bottom left]
        """
        top=self.compute_top(depth_tensor,range_list[0])
        bottom=self.compute_bottom(depth_tensor,range_list[0])
        rd_top = self.compute_rd_top(bottom, top)
        # print(range_list[0])
        # print(top)
        # print(bottom)
        # print(rd_top)
        # print("---------------------------")
        top2 = self.compute_top(depth_tensor, range_list[1])
        bottom2 = self.compute_bottom(depth_tensor, range_list[1])
        rd_bottom = self.compute_rd_bottom(top2, bottom2)
        # print(range_list[1])
        # print(top2)
        # print(bottom2)
        # print(rd_bottom)
        # print("---------------------------")
        right=self.compute_right(depth_tensor,range_list[2])
        left=self.compute_left(depth_tensor,range_list[2])
        rd_right = self.compute_rd_right(left, right)
        # print(range_list[2])
        # print(right)
        # print(left)
        # print(rd_right)
        # print("---------------------------")
        right2 = self.compute_right(depth_tensor, range_list[3])
        left2 = self.compute_left(depth_tensor, range_list[3])
        rd_left = self.compute_rd_left(left2, right2)
        # print(range_list[3])
        # print(right2)
        # print(left2)
        # print(rd_left)
        # print("---------------------------")
        left_top=self.compute_left_top(depth_tensor,range_list[4])
        bottom_right = self.compute_bottom_right(depth_tensor, range_list[4])
        rd_left_top = self.compute_rd_left_top(left_top, bottom_right)
        # print(range_list[4])
        # print(left_top)
        # print(bottom_right)
        # print(rd_left_top)
        #
        # print("---------------------------")
        left_top2 = self.compute_left_top(depth_tensor, range_list[5])
        bottom_right2 = self.compute_bottom_right(depth_tensor, range_list[5])
        rd_bottom_right=self.compute_rd_bottom_right(bottom_right2,left_top2)
        # print(range_list[5])
        # print(left_top2)
        # print(bottom_right2)
        # print(rd_bottom_right)
        # print("---------------------------")
        right_top = self.compute_right_top(depth_tensor, range_list[6])
        bottom_left=self.compute_bottom_left(depth_tensor,range_list[6])
        rd_right_top=self.compute_rd_right_top(right_top,bottom_left)
        # print(range_list[6])
        # print(right_top)
        # print(bottom_left)
        # print(rd_right_top)
        #
        # print("---------------------------")
        right_top2 = self.compute_right_top(depth_tensor, range_list[7])
        bottom_left2 = self.compute_bottom_left(depth_tensor, range_list[7])
        rd_bottom_left=self.compute_rd_bottom_left(bottom_left2,right_top2)
        # print(range_list[7])
        # print(right_top2)
        # print(bottom_left2)
        # print(rd_bottom_left)

        relative_depth_list=[rd_top,rd_right,rd_bottom,rd_left,rd_left_top,rd_right_top,rd_bottom_right,rd_bottom_left]

        return relative_depth_list

    def forward(self,pred,gt):
        """
        compute the difference of relative depth map between predicted depth map and ground truth depth map
        :param pred: predicted depth map,B*1*H*W
        :param gt: ground truth depth map,B*1*H*W
        :return: difference of relative depth map between pred and gt
        """

        # range_list=np.random.randint(1,self.range+1,size=8)

        range_list=[30]*8

        mean_pred=pred.mean(2,True).mean(3,True)
        mean_dt=gt.mean(2,True).mean(3,True)
        norm_pred=pred/mean_pred
        norm_gt=gt/mean_dt

        pred_rd_list=self.compute_rd_map_list(norm_pred,range_list)
        gt_rd_list=self.compute_rd_map_list(norm_gt,range_list)

        loss_fn=torch.nn.L1Loss(reduction='mean')
        # loss_fn=torch.nn.SmoothL1Loss(reduction='mean')
        # loss_fn=torch.nn.MSELoss(reduction='mean')

        loss=0
        for i in range(len(pred_rd_list)):
            loss=loss+loss_fn(pred_rd_list[i],gt_rd_list[i])

        return loss





