import torch
import torch.nn as nn
from timm.models.registry import register_model
from ptflops import get_model_complexity_info


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block，用 Conv1x1 替代 Linear"""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y.expand_as(x)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, in_channels, out_channels, stride=1, act=True):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class OptimizedLiteBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=False):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(inplanes, planes, stride, act=True)
        self.conv2 = DepthwiseSeparableConv(planes, planes, act=False)
        self.se = SEBlock(planes) if use_se else nn.Identity()
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class InfNet(nn.Module):
    def __init__(self, block=OptimizedLiteBlock, layers=(3, 4, 6, 3), 
                 channels=(64, 128, 256, 512), 
                 width_mult=1.0, use_se=False, dropout=0.2, head_type="conv",num_features=256):
        """
        head_type: 
            "conv"     -> Conv3×3 + BN + GAP + FC → 256
            "dropout"  -> GAP + Dropout + FC → 256
        """
        super().__init__()
        channels = tuple(max(8, int(c * width_mult)) for c in channels)
        self.inplanes = channels[0]

        # Stem
        self.conv1 = nn.Conv2d(3, channels[0], 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.stem_dw = DepthwiseSeparableConv(channels[0], channels[0], stride=1, act=True)

        # Stages
        self.layer1 = self._make_layer(block, channels[0], layers[0], stride=1, use_se=use_se)
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=2, use_se=use_se)
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2, use_se=use_se)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2, use_se=use_se)

        # Head
        self.head_type = head_type
        if head_type == "conv":
            self.head_conv = nn.Conv2d(channels[3], 256, 3, 1, 0, bias=False)
            self.head_bn = nn.BatchNorm2d(256)
            self.fc = nn.Linear(256, 256)
        elif head_type == "dropout":
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(channels[3], 256)
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")

        self._init_weights()

    def _make_layer(self, block, planes, blocks, stride=1, use_se=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample, use_se=use_se)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=use_se))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.stem_dw(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.head_type == "conv":
            x = self.head_conv(x)
            x = self.head_bn(x)
            x = nn.AdaptiveAvgPool2d((1, 1))(x).flatten(1)
            x = self.fc(x)
        elif self.head_type == "dropout":
            x = self.global_pool(x).flatten(1)
            x = self.dropout(x)
            x = self.fc(x)

        return x


@register_model
def inf_net(**kwargs):
    return InfNet(**kwargs)




# ----------------- 测试 -----------------
if __name__ == "__main__":
    net = inf_net().eval()
    # from torchsummary import summary
    # summary(net, (3, 128, 128), device='cpu')
    flops, params = get_model_complexity_info(net, (3, 128, 128), as_strings=True, print_per_layer_stat=False)
    print(f"[InfNet] FLOPs: {flops} | Params: {params}")