
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyCDNet(nn.Module):
    def __init__(
        self,
        n_channels,
        n_classes,
        kernels_per_layer=2,
        bilinear=False,
        reduction_ratio=16,
    ):
        super(MyCDNet, self).__init__()
        self.n_classes = n_classes
        kernels_per_layer = kernels_per_layer
        self.bilinear = bilinear
        reduction_ratio = reduction_ratio
        self.input_nbr_1, self.input_nbr_2 = n_channels

        self.inc = DoubleConvDS(self.input_nbr_1, 16, kernels_per_layer=kernels_per_layer)
        self.cbam1 = CBAM(16, reduction_ratio=reduction_ratio)
        self.down1 = DownDS(16, 32, kernels_per_layer=kernels_per_layer)
        self.cbam2 = CBAM(32, reduction_ratio=reduction_ratio)
        self.down2 = DownDS(32, 64, kernels_per_layer=kernels_per_layer)
        self.cbam3 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down3 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam4 = CBAM(128, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1

        self.inc_2 = DoubleConvDS(self.input_nbr_2, 16, kernels_per_layer=kernels_per_layer)
        self.cbam1_2 = CBAM(16, reduction_ratio=reduction_ratio)
        self.down1_2 = DownDS(16, 32, kernels_per_layer=kernels_per_layer)
        self.cbam2_2 = CBAM(32, reduction_ratio=reduction_ratio)
        self.down2_2 = DownDS(32, 64, kernels_per_layer=kernels_per_layer)
        self.cbam3_2 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down3_2 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam4_2 = CBAM(128, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1

        self.sa = TDAM()
        self.up2 = UpDS(128*2, 64*2 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(64*2, 32*2 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(32*2, 16*2, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(16*2, self.n_classes)

    def forward(self, s2_1, s2_2, s1_1, s1_2):

        x1 = self.inc(s2_1)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3)
        x4Att = self.cbam4(x4)

        x1 = self.inc(s2_2)
        x1Att_2 = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att_2 = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att_2 = self.cbam3(x3)
        x4 = self.down3(x3)
        x4Att_2 = self.cbam4(x4)

        x1_b = self.inc_2(s1_1)
        x1Att_b = self.cbam1_2(x1_b)
        x2_b = self.down1_2(x1_b)
        x2Att_b = self.cbam2_2(x2_b)
        x3_b = self.down2_2(x2_b)
        x3Att_b = self.cbam3_2(x3_b)
        x4_b = self.down3_2(x3_b)
        x4Att_b = self.cbam4_2(x4_b)

        x1_b = self.inc_2(s1_2)
        x1Att_b_2 = self.cbam1_2(x1_b)
        x2_b = self.down1_2(x1_b)
        x2Att_b_2 = self.cbam2_2(x2_b)
        x3_b = self.down2_2(x2_b)
        x3Att_b_2 = self.cbam3_2(x3_b)
        x4_b = self.down3_2(x3_b)
        x4Att_b_2 = self.cbam4_2(x4_b)

        x4Att_two = torch.cat((x4Att-x4Att_2, x4Att_b-x4Att_b_2), 1)
        x4Att_two_weight = self.sa(x4Att_two)

        x = self.up2(x4Att_two_weight, x3Att - x3Att_2, x3Att_b - x3Att_b_2)
        x = self.up3(x, x2Att-x2Att_2, x2Att_b-x2Att_b_2)
        x = self.up4(x, x1Att-x1Att_2, x1Att_b-x1Att_b_2)
        logits = self.outc(x)
        return logits

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.conv2 = nn.Conv2d(16, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv2(self.conv1(x))


class DoubleConvDS(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None, kernels_per_layer=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(
                mid_channels,
                out_channels,
                kernel_size=3,
                kernels_per_layer=kernels_per_layer,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)



class DownDS(nn.Module):

    def __init__(self, in_channels, out_channels, kernels_per_layer=1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvDS(in_channels, out_channels, kernels_per_layer=kernels_per_layer),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpDS(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=False, kernels_per_layer=1):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConvDS(
                in_channels,
                out_channels,
                in_channels // 2,
                kernels_per_layer=kernels_per_layer,
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvDS(in_channels, out_channels, kernels_per_layer=kernels_per_layer)

    def forward(self, x1, x2, x3):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x1, x2, x3], dim=1)
        return self.conv(x)

class ChannelAttention(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.input_channels = input_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.MLP = nn.Sequential(
            Flatten(),
            nn.Linear(input_channels, input_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(input_channels // reduction_ratio, input_channels),
        )

    def forward(self, x):
        avg_values = self.avg_pool(x)
        max_values = self.max_pool(x)
        out = self.MLP(avg_values) + self.MLP(max_values)
        scale = x * torch.sigmoid(out).unsqueeze(2).unsqueeze(3).expand_as(x)
        return scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.bn(out)
        scale = x * torch.sigmoid(out)
        return scale


class CBAM(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(input_channels, reduction_ratio=reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        out = self.channel_att(x)
        out = self.spatial_att(out)
        return out

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_size, padding=0, kernels_per_layer=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels * kernels_per_layer,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(in_channels * kernels_per_layer, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class TDAM(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-5, var1=8, var2=0.6):
        super(TDAM, self).__init__()

        self.activaton = nn.Sigmoid()

        # The selection of values for `e_lambda`, `var1`, and `var2` will affect the effectiveness of feature fusion
        # What we provide here are just some examples
        self.e_lambda = e_lambda
        self.var1 = var1
        self.var2 = var2

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "tdam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (self.var1 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + self.var2

        return x * self.activaton(y)
