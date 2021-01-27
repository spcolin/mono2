import torch


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """

    mean_disp = disp.mean(2, True).mean(3, True)
    mean_img = img.mean(2, True).mean(3, True)
    norm_disp = disp / mean_disp
    norm_img = img / mean_img

    grad_disp_x = torch.abs(norm_disp[:, :, :, :-1] - norm_disp[:, :, :, 1:])
    grad_disp_y = torch.abs(norm_disp[:, :, :-1, :] - norm_disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(norm_img[:, :, :, :-1] - norm_img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(norm_img[:, :, :-1, :] - norm_img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()