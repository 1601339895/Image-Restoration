## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange



# 保留原有的基础工具类（autopad、Conv、Channel、Spatial）
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    
    default_act = nn.GELU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, bias=False):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=bias)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class Channel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.Apt = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x2 = self.dwconv(x)
        x5 = self.Apt(x2)
        x6 = self.sigmoid(x5)
        return x6

class Spatial(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, 1, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        x5 = self.bn(x1)
        x6 = self.sigmoid(x5)
        return x6

# 轻量化版FCM_FFN：拆分为1/4 + 1/4 + 2/4，2/4用恒等/轻量卷积
class FCM_FFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2., bias=False, part3_strategy='identity'):
        """
        Args:
            dim: 输入通道维度
            ffn_expansion_factor: 扩张因子（可适当降低，如1.5或1.0）
            bias: 偏置
            part3_strategy: part3的处理策略，可选['identity', '1x1conv', 'dwconv3x3']
        """
        super().__init__()
        # 1. 通道拆分：1/4 + 1/4 + 2/4（核心修改）
        self.part1 = dim // 4  # 1/4通道
        self.part2 = dim // 4  # 1/4通道
        self.part3 = dim - self.part1 - self.part2  # 剩余2/4（即1/2）通道

        # 2. 扩张因子：对part1/part2采用扩张（可降低扩张因子减少计算量）
        self.hidden1 = int(self.part1 * ffn_expansion_factor)
        self.hidden2 = int(self.part2 * ffn_expansion_factor)

        # 3. part1/part2的轻量流程（简化原流程：去掉部分细化卷积，或保留核心）
        self.project_in1 = Conv(self.part1, self.hidden1, k=1, act=True, bias=bias)
        self.dwconv1 = Conv(self.hidden1, self.hidden1, k=3, g=self.hidden1, act=True, bias=bias)  # 深度卷积
        self.unify1 = Conv(self.hidden1, dim, k=1, act=True, bias=bias)  # 映射到dim

        self.project_in2 = Conv(self.part2, self.hidden2, k=1, act=True, bias=bias)
        self.dwconv2 = Conv(self.hidden2, self.hidden2, k=3, g=self.hidden2, act=True, bias=bias)
        self.unify2 = Conv(self.hidden2, dim, k=1, act=True, bias=bias)  # 映射到dim

        # 4. part3的处理策略（恒等/1x1卷积/3x3深度卷积，核心轻量化点）
        self.part3_strategy = part3_strategy
        if part3_strategy == '1x1conv':
            self.part3_process = Conv(self.part3, dim, k=1, act=True, bias=bias)  # 1x1卷积映射到dim
        elif part3_strategy == 'dwconv3x3':
            self.part3_process = Conv(self.part3, dim, k=3, g=self.part3, act=True, bias=bias)  # 深度卷积映射到dim
        elif part3_strategy == 'identity':
            # 恒等映射：若part3 != dim，用1x1卷积适配维度（无激活，轻量）
            if self.part3 != dim:
                self.part3_process = Conv(self.part3, dim, k=1, act=False, bias=bias)
            else:
                self.part3_process = nn.Identity()
        else:
            raise ValueError(f"part3_strategy must be in ['identity', '1x1conv', 'dwconv3x3'], got {part3_strategy}")

        # 5. 保留FCM的空间/通道注意力（输入为dim，与原一致）
        self.spatial = Spatial(dim)
        self.channel = Channel(dim)

        # 6. 输出层（轻量版，去掉冗余卷积）
        self.project_out = Conv(dim, dim, k=1, act=False, bias=bias)

    def forward(self, x):
        # Step1: 通道拆分（1/4, 1/4, 2/4）
        x1, x2, x3 = torch.split(x, [self.part1, self.part2, self.part3], dim=1)

        # Step2: part1/part2的轻量流程
        x1 = self.project_in1(x1)
        x1 = self.dwconv1(x1)
        x1 = self.unify1(x1)

        x2 = self.project_in2(x2)
        x2 = self.dwconv2(x2)
        x2 = self.unify2(x2)

        # Step3: part3的轻量化处理
        x3 = self.part3_process(x3)

        # Step4: 复刻FCM的注意力交互（可简化为直接融合，进一步轻量化）
        # 方案1：保留原注意力交互
        x33 = self.spatial(x2) * x1  # 空间注意力加权part1
        x44 = self.channel(x1) * x2  # 通道注意力加权part2
        x_fuse = x33 + x44 + x3  # 融合part1/part2/part3

        # pdb.set_trace()
        # x_fuse = torch.cat([x3, x33, x44], dim=1)

        # 方案2：直接拼接融合（更轻量，可选）
        # x_fuse = x1 + x2 + x3

        # Step5: 输出层
        out = self.project_out(x_fuse)

        return out


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class GatedMDTA(nn.Module):
    def __init__(self, dim, num_heads, bias, gate_type=None):
        super(GatedMDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.gate_type = gate_type
        self.dim = dim
        self.bias = bias

        if gate_type is None:
            self.qkv_out_channels = dim * 3
        elif gate_type == 'headwise':
            self.qkv_out_channels = dim * 3 + num_heads
        elif gate_type == 'elementwise':
            self.qkv_out_channels = dim * 4

        self.qkv = nn.Conv2d(dim, self.qkv_out_channels, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(self.qkv_out_channels, self.qkv_out_channels, kernel_size=3, stride=1, padding=1,
                                    groups=self.qkv_out_channels, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        head_dim = c // self.num_heads
        qkv_with_gate = self.qkv_dwconv(self.qkv(x))

        if self.gate_type is None:
            q, k, v = qkv_with_gate.chunk(3, dim=1)
            gate_score = None
        elif self.gate_type == 'headwise':
            q_with_gate, k, v = qkv_with_gate.split([self.dim + self.num_heads, self.dim, self.dim], dim=1)
            q, gate_score = q_with_gate.split([self.dim, self.num_heads], dim=1)
        elif self.gate_type == 'elementwise':
            q_with_gate, k, v = qkv_with_gate.split([self.dim * 2, self.dim, self.dim], dim=1)
            q, gate_score = q_with_gate.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads, c=head_dim)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads, c=head_dim)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads, c=head_dim)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v

        if self.gate_type is not None:
            if self.gate_type == 'headwise':
                gate_score = rearrange(gate_score, 'b head h w -> b head 1 (h w)', head=self.num_heads)
            elif self.gate_type == 'elementwise':
                gate_score = rearrange(gate_score, 'b (head c) h w -> b head c (h w)', head=self.num_heads, c=head_dim)
            out = out * torch.sigmoid(gate_score)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, gate_type=None, part3_strategy='identity'):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = GatedMDTA(dim, num_heads, bias, gate_type=gate_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FCM_FFN(dim, ffn_expansion_factor, bias, part3_strategy=part3_strategy)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
##---------- Model_IR -----------------------
class Model_IR(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        gate_type=None,  ## Other option "elementwise" "headwise"
        part3_strategy='dwconv3x3',  # Other option 'dwconv3x3' '1x1conv' 'identity'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(Model_IR, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, gate_type=gate_type, part3_strategy=part3_strategy) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, gate_type=gate_type, part3_strategy=part3_strategy) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, gate_type=gate_type, part3_strategy=part3_strategy) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, gate_type=gate_type, part3_strategy=part3_strategy) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, gate_type=gate_type, part3_strategy=part3_strategy) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, gate_type=gate_type, part3_strategy=part3_strategy) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, gate_type=gate_type, part3_strategy=part3_strategy) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, gate_type=gate_type, part3_strategy=part3_strategy) for i in range(num_refinement_blocks)])
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img


        return out_dec_level1
    

if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    # test
    model = Model_IR(
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',
        gate_type= 'elementwise',
        part3_strategy='identity'  
    ).cuda()

    # 1x1conv: model  7.8M    37.5G  Max Memery : 6853.125 [M]
    # dwconv3x3: model  7.442M  36.187G Max Memery : 6851.764 [M]
    # 计算模型参数量
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {model_params / 1e6:.3f} M")

    x = torch.randn(1, 3, 224, 224).cuda()
    y = model(x)
    print(y.shape)
    # Memory usage  
    print('{:>16s} : {:<.3f} [M]'.format('Max Memery',
                                         torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1024 ** 2))

    # FLOPS and PARAMS
    flops = FlopCountAnalysis(model, (x))
    print(flop_count_table(flops))

