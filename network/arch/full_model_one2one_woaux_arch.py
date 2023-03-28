import torch
import torch.nn as nn
import torch.nn.functional as F


conv_s2 = 4
pad0 = 1
align_corners=True


def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def std_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)
    

class DirectAwareAtt(nn.Module):
    def __init__(self, channels=144, reduction=16):
        super().__init__()
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.std = std_channels

        self.avg_att_module = nn.Sequential(
             nn.Conv2d(channels, channels // reduction, 1, 1, 0),
             nn.ReLU(True),
             nn.Conv2d(channels // reduction, channels, 1, 1, 0),
             nn.Sigmoid()
        )
        self.std_att_module = nn.Sequential(
             nn.Conv2d(channels, channels // reduction, 1, 1, 0),
             nn.ReLU(True),
             nn.Conv2d(channels // reduction, channels, 1, 1, 0),
             nn.Sigmoid()
        )


    def forward(self, x):
        avg_std = self.std(x)
        avg = self.avg_pool(x)
        att = self.avg_att_module(avg) + self.std_att_module(avg_std)
        att = att / 2
        return att * x


class DilateBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv3x3_d1 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, 1))
        self.conv3x3_d2 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 2, 2))
        self.conv3x3_d3 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 3, 3))
        self.att_module = DirectAwareAtt(channels*3, 16)

        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d) or isinstance(p, nn.ConvTranspose2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)

    def forward(self, x):
        feat = torch.cat([self.conv3x3_d1(x), self.conv3x3_d2(x), self.conv3x3_d3(x)], 1)
        return self.att_module(feat)

class DilateResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            DilateBlock(channels),
            nn.ReLU(True),
            nn.Conv2d(channels*3, channels, 3, 1, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d) or isinstance(p, nn.ConvTranspose2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)

    def forward(self, x):
        x = x + self.block(x)
        return x


class UpConv(nn.Module):
    def __init__(self, inc, outc, scale=2):
        super(UpConv, self).__init__()
        self.scale = scale
        self.conv = nn.Conv2d(inc, outc, 3, stride=1, padding=1)


        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d) or isinstance(p, nn.ConvTranspose2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True))
        

class RGBEncoder(nn.Module):
    def __init__(self, in_channels, channels, filter_size):
        super(RGBEncoder, self).__init__()

        padding = int((filter_size - 1) / 2)

        self.init = nn.Sequential(nn.Conv2d(in_channels, channels, filter_size, stride=1, padding=padding),
                                  nn.ReLU(True),
                                  nn.Conv2d(channels, channels, filter_size, stride=1, padding=padding))

        self.enc1 = nn.Sequential(nn.ReLU(True),
                                  nn.Conv2d(channels, channels, conv_s2, stride=2, padding=pad0),
                                  nn.ReLU(True),
                                  nn.Conv2d(channels, channels, filter_size, stride=1, padding=padding),
                                  )

        self.enc2 = nn.Sequential(nn.ReLU(True),
                                  nn.Conv2d(channels, channels, conv_s2, stride=2, padding=pad0),
                                  nn.ReLU(True),
                                  nn.Conv2d(channels, channels, filter_size, stride=1, padding=padding),
                                  )

        self.bottleneck = nn.Sequential(*[DilateResBlock(channels) for _ in range(4)])

        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d) or isinstance(p, nn.ConvTranspose2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)

    def forward(self, input, scale=2, c_x0=None, c_x1=None, c_x2=None, pre_x2=None, pre_x3=None, pre_x4=None):
        ### input

        x0 = self.init(input)
        x0 = x0# + c_x0
        if pre_x4 is not None:
            x0 = x0 + F.interpolate(pre_x4, scale_factor=scale, mode='bilinear', align_corners=align_corners)

        x1 = self.enc1(x0) #1/2 input size
        x1 = x1# + c_x1
        if pre_x3 is not None: 
            x1 = x1 + F.interpolate(pre_x3, scale_factor=scale, mode='bilinear', align_corners=align_corners)

        x2 = self.enc2(x1) # 1/4 input size
        x2 = x2# + c_x2
        if pre_x2 is not None:
            x2 = x2 + F.interpolate(pre_x2, scale_factor=scale, mode='bilinear', align_corners=align_corners)

        x2 = self.bottleneck(x2)
        return x0, x1, x2


class CondEncoder(nn.Module):
    def __init__(self, in_channels, channels, filter_size):
        super(CondEncoder, self).__init__()
        in_channels += 2
        padding = int((filter_size - 1) / 2)

        self.init = nn.Sequential(nn.Conv2d(in_channels, channels, filter_size, stride=1, padding=padding),
                                  nn.ReLU(True),
                                  nn.Conv2d(channels, channels, filter_size, stride=1, padding=padding))

        self.enc1 = nn.Sequential(nn.ReLU(True),
                                  nn.Conv2d(channels, channels, conv_s2, stride=2, padding=pad0),
                                  nn.ReLU(True),
                                  nn.Conv2d(channels, channels, filter_size, stride=1, padding=padding), )

        self.enc2 = nn.Sequential(nn.ReLU(True),
                                  nn.Conv2d(channels, channels, conv_s2, stride=2, padding=pad0),
                                  nn.ReLU(True),
                                  nn.Conv2d(channels, channels, filter_size, stride=1, padding=padding), )

        self.enc3 = nn.Sequential(nn.ReLU(True),
                                  nn.Conv2d(channels, channels, conv_s2, stride=2, padding=pad0),
                                  nn.ReLU(True),
                                  nn.Conv2d(channels, channels, filter_size, stride=1, padding=padding), )

        self.enc4 = nn.Sequential(nn.ReLU(True),
                                  nn.Conv2d(channels, channels, conv_s2, stride=2, padding=pad0),
                                  nn.ReLU(True),
                                  nn.Conv2d(channels, channels, filter_size, stride=1, padding=padding), )

        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d) or isinstance(p, nn.ConvTranspose2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)

    def forward(self, input, scale=2, pre_x=None):
        ### input
        b, _, h, w = input.shape
        x_code = torch.Tensor([float(x)/(w-1) for x in range(w)]).float().cuda() * 2 - 1
        y_code = torch.Tensor([float(y)/(h-1) for y in range(h)]).float().cuda() * 2 - 1
        grid_y, grid_x = torch.meshgrid(y_code, x_code)

        grid_y = grid_y.view(1,1,h,w).expand(b,1,h,w)
        grid_x = grid_x.view(1,1,h,w).expand(b,1,h,w)

        input = torch.cat([grid_x, grid_y], 1)
        x0 = self.init(input)

        if pre_x is not None:
            x0 = x0 + F.interpolate(pre_x, scale_factor=scale, mode='bilinear', align_corners=align_corners)

        x1 = self.enc1(x0)  # 1/2 input size
        x2 = self.enc2(x1)  # 1/4 input size
        x3 = self.enc3(x2)  # 1/8 input size
        x4 = self.enc4(x3)  # 1/16 input size

        # return the pre-activated features
        return x0, x1, x2, x3, x4


class RGBDecoder(nn.Module):
    def __init__(self, channels, filter_size):
        super(RGBDecoder, self).__init__()
        padding = int((filter_size-1)/2)
 
        self.dec2 = nn.Sequential(nn.ReLU(True),
                                  nn.Conv2d(channels//2, channels//2, filter_size, stride=1, padding=padding),
                                  nn.ReLU(True),
                                  UpConv(channels//2, channels//2),
                                  nn.ReLU(True),
                                  nn.Conv2d(channels//2, channels//2, filter_size, stride=1, padding=padding),
                                  )

        self.dec1 = nn.Sequential(nn.ReLU(True),
                                  nn.Conv2d(channels//2, channels//2, filter_size, stride=1, padding=padding),
                                  nn.ReLU(True),
                                  UpConv(channels//2, channels//2),
                                  nn.ReLU(True),
                                  nn.Conv2d(channels//2, channels//2, filter_size, stride=1, padding=padding),
                                  )

        self.prdct = nn.Sequential(nn.ReLU(True),
                                   nn.Conv2d(channels//2, channels//2, filter_size, stride=1, padding=padding),
                                   nn.ReLU(True),
                                   nn.Conv2d(channels//2, 3, filter_size, stride=1, padding=padding))

        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d) or isinstance(p, nn.ConvTranspose2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)

    def forward(self, pre_rgbx, pre_x2 = None, pre_x3 = None, pre_x4 = None):

        x2 = pre_rgbx[2]
        x1 = pre_rgbx[1]
        x0 = pre_rgbx[0]

        if pre_x2 != None:
            x2 = x2 + F.interpolate(pre_x2, scale_factor=2, mode='bilinear', align_corners=align_corners)

        x3 = self.dec2(x2) # 1/2 input size
        if pre_x3 != None:
            x3 = x3 + F.interpolate(pre_x3, scale_factor=2, mode='bilinear', align_corners=align_corners)
        
        x4 = self.dec1(x1+x3) #1/1 input size
        if pre_x4 != None:
            x4 = x4 + F.interpolate(pre_x4, scale_factor=2, mode='bilinear', align_corners=align_corners)
        ### prediction
        output_rgb = self.prdct(x4 + x0)

        return x2, x3, x4, output_rgb


class One2One_noaux(nn.Module):
    def __init__(self, in_channels=3, short_connection=True):
        super(One2One_noaux, self).__init__()

        self.short_connection = short_connection
        self.in_channels = in_channels
        denc_channels = 48
        cenc_channels = 48
        ddcd_channels = denc_channels+cenc_channels

        # self.cond_encoder = CondEncoder(0, cenc_channels, 3)

        self.rgb_encoder1 = RGBEncoder(in_channels, denc_channels, 3)
        self.rgb_decoder1 = RGBDecoder(ddcd_channels, 3)

        self.rgb_encoder2 = RGBEncoder(2*in_channels, denc_channels, 3)
        self.rgb_decoder2 = RGBDecoder(ddcd_channels, 3)

        self.rgb_encoder3 = RGBEncoder(2*in_channels, denc_channels, 3)
        self.rgb_decoder3 = RGBDecoder(ddcd_channels, 3)


    def forward(self, x, enable_layers=None):
        x = x[0]
        input_rgb=x
        ## for the 1/4 res
        input_rgb14 = F.interpolate(input_rgb, scale_factor=0.25, mode='bilinear',align_corners=align_corners)
        # print(enc_c[2].shape)
        enc_rgb14 = self.rgb_encoder1(input_rgb14, 2)  # enc_rgb [larger -> smaller size] 
        dcd_rgb14 = self.rgb_decoder1(enc_rgb14) # dec_rgb [smaller -> larger size] 

        ## for the 1/2 res
        input_rgb12 = F.interpolate(input_rgb, scale_factor=0.5, mode='bilinear', align_corners=align_corners)
        ori_pred_rgb14 = dcd_rgb14[3]
        if self.short_connection:
            ori_pred_rgb14 += input_rgb14
        predict_rgb12 = F.interpolate(ori_pred_rgb14, scale_factor=2, mode='bilinear', align_corners=align_corners)
        input_12 = torch.cat((input_rgb12, predict_rgb12), 1)

        enc_rgb12 = self.rgb_encoder2(input_12, 2) 
        dcd_rgb12 = self.rgb_decoder2(enc_rgb12, dcd_rgb14[0], dcd_rgb14[1], dcd_rgb14[2])

        ## for the 1/1 res
        ori_pred_rgb12 = dcd_rgb12[3]
        if self.short_connection:
            ori_pred_rgb12 += input_rgb12
        predict_rgb11 = F.interpolate(ori_pred_rgb12, scale_factor=2, mode='bilinear', align_corners=align_corners)
        input_11 = torch.cat((input_rgb, predict_rgb11), 1)

        enc_rgb11 = self.rgb_encoder3(input_11, 2)
        dcd_rgb11 = self.rgb_decoder3(enc_rgb11, dcd_rgb12[0], dcd_rgb12[1], dcd_rgb12[2])

        output_rgb11 = dcd_rgb11[3]
        if self.short_connection:
            output_rgb11 += input_rgb

        return output_rgb11, ori_pred_rgb12, ori_pred_rgb14

