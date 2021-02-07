import torch, torchvision, collections
import torch.nn.functional as F


def conv_layer(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
        torch.nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True),
        torch.nn.LeakyReLU(negative_slope=0.01)

    )


class Residual(torch.nn.Module):

    def __init__(self):
        super(Residual, self).__init__()

        # for upsample
        self.up_layer1 = conv_layer(2048, 1024)
        self.up_layer2 = conv_layer(1024, 512)

        self.up_layer3 = conv_layer(516, 256)
        self.up_layer4 = conv_layer(256, 64)

        self.up_layer5 = conv_layer(68, 16)
        self.up_layer6 = torch.nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1, bias=False)
        # prediction layer
        self.predict = torch.nn.Tanh()

    def forward(self, input,IandD):
        # input:B*2048*13*13

        up1 = F.interpolate(input, scale_factor=2, mode='bilinear', align_corners=True)  # B*2048*26*26
        up1 = self.up_layer1(up1)  # B*1024*26*26
        up1 = F.interpolate(up1, scale_factor=2, mode='bilinear', align_corners=True)  # B*1024*52*52
        up1 = self.up_layer2(up1)  # B*512*52*52

        cat1 = torch.cat([up1, F.interpolate(IandD, size=up1.shape[2:])], 1)  # B*516*52*52

        up2 = F.interpolate(cat1, scale_factor=2, mode='bilinear', align_corners=True)  # B*516*104*104
        up2 = self.up_layer3(up2)  # B*256*104*104
        up2 = F.interpolate(up2, scale_factor=2, mode='bilinear', align_corners=True)  # B*256*208*208
        up2 = self.up_layer4(up2)  # B*64*208*208

        cat2 = torch.cat([up2, F.interpolate(IandD, size=up2.shape[2:])], 1)  # B*68*208*208

        up = F.interpolate(cat2, scale_factor=2, mode='bilinear', align_corners=True)  # B*68*416*416
        up = self.up_layer5(up)  # B*16*416*416
        up = F.interpolate(up, size=(385, 385), mode='bilinear', align_corners=True)  # B*16*385*385
        prediction = self.up_layer6(up)  # B*1*385*385

        # scale the depth residual between -0.01 and 0.01
        prediction = self.predict(prediction) * 0.01  # B*1*385*385

        return prediction


class Resample(torch.nn.Module):

    def __init__(self):
        super(Resample, self).__init__()

        # for upsample
        self.up_layer1 = conv_layer(2048, 1024)
        self.up_layer2 = conv_layer(1024, 512)

        self.up_layer3 = conv_layer(516, 256)
        self.up_layer4 = conv_layer(256, 64)

        self.up_layer5 = conv_layer(68, 16)
        self.up_layer6 = torch.nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1, bias=False)
        # prediction layer
        self.predict = torch.nn.Sigmoid()


    def forward(self, input,IandD):

        up1 = F.interpolate(input, scale_factor=2, mode='bilinear', align_corners=True)  # B*512*26*26
        up1 = self.up_layer1(up1)  # B*256*26*26
        up1 = F.interpolate(up1, scale_factor=2, mode='bilinear', align_corners=True)  # B*256*52*52
        up1 = self.up_layer2(up1)  # B*128*52*52

        cat1 = torch.cat([up1, F.interpolate(IandD, size=up1.shape[2:])], 1)  # B*132*52*52

        up2 = F.interpolate(cat1, scale_factor=2, mode='bilinear', align_corners=True)  # B*132*104*104
        up2 = self.up_layer3(up2)  # B*64*104*104
        up2 = F.interpolate(up2, scale_factor=2, mode='bilinear', align_corners=True)  # B*64*208*208
        up2 = self.up_layer4(up2)  # B*32*208*208

        cat2 = torch.cat([up2, F.interpolate(IandD, size=up2.shape[2:])], 1)  # B*36*208*208

        up = F.interpolate(cat2, scale_factor=2, mode='bilinear', align_corners=True)  # B*32*416*416
        up = self.up_layer5(up)  # B*16*416*416
        up = F.interpolate(up, size=(385, 385), mode='bilinear', align_corners=True)  # B*16*385*385
        prediction = self.up_layer6(up)  # B*1*385*385

        # scale need to specify
        prediction = self.predict(prediction)  # B*1*385*385

        # the range size for sampling windows is 0 to 10
        return prediction.permute(0, 2, 3, 1) * 10


class Refine_module3(torch.nn.Module):

    def __init__(self, load_pretrained=True, pretrained_resnet50_path=None):
        super(Refine_module3, self).__init__()
        backbone_model = torchvision.models.resnet50(pretrained=False)

        # -----------------resnet backbone part--------------------
        self.backbone = torch.nn.Sequential(collections.OrderedDict(
            [(k, v) for k, v in backbone_model.named_children()][:-2]
        )
        )

        if load_pretrained == True:
            self.load_pretriained_backbone(pretrained_resnet50_path)

        # adjust the first layer of backbone to have 4 channels(3 for rgb image and 1 for depth map)
        self.backbone.conv1 = torch.nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # -----------------backbone end-----------------------------


        self.res_module=Residual()
        self.sam_module=Resample()


    def load_pretriained_backbone(self, path):

        pretrained_dict = torch.load(path)
        model_dict = self.backbone.state_dict()
        # print(pretrained_dict.keys())
        # print(model_dict.keys())
        to_load_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(to_load_dict)
        self.backbone.load_state_dict(model_dict)

        print("-----pretrained resnet loaded-----")

    def sample(self, grid, tensor):
        batch_size = tensor.shape[0]
        theta = torch.Tensor([[1, 0, 0], [0, 1, 0]])
        theta = theta.expand(batch_size, 2, 3)

        # values between -1 and 1
        base_grid = torch.nn.functional.affine_grid(theta, size=tensor.shape).to(tensor.device)

        final_grid = base_grid + grid / 385.0

        resampled_depth = torch.nn.functional.grid_sample(tensor, final_grid)

        return resampled_depth

    def forward(self, depth_tensor, img_tensor):
        input = torch.cat((img_tensor, depth_tensor), 1)
        features=self.backbone(input)

        res = self.res_module(features,input)
        resample_grid = self.sam_module(features,input)

        compensate_res_depth = depth_tensor + res
        resampled_depth = self.sample(resample_grid, compensate_res_depth)

        return compensate_res_depth, resampled_depth

# pretrained_path="/home/colin/pretrained/resnet18-5c106cde.pth"
#
# refine_net=Refine_module(pretrained_resnet18_path=pretrained_path)
#
# dp=torch.randn(4,1,385,385)
# rgb=torch.randn(4,3,385,385)
#
# output=refine_net(dp,rgb)






