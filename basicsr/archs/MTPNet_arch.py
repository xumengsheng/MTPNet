import math
import numbers

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec

from torch.fft import fft2, fftshift, ifft2, ifftshift
from basicsr.archs.arch_util2 import LayerNorm2d
from functools import partial
from typing import Optional, Callable
from basicsr.utils.registry import ARCH_REGISTRY
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat
from fvcore.nn import FlopCountAnalysis, flop_count_table


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


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
        return x / torch.sqrt(sigma + 1e-5) * self.weight


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
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class DepthWiseConv(nn.Module):

    def __init__(self, dim, kernel_size):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size,
                      padding=padding, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

    def forward(self, x):
        return self.dwconv(x)

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()

        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        # 这个卷积层接收2个通道的输入（这里指的是后面通过torch.cat拼接后的max和mean特征图），输出1个通道的特征图。卷积核大小为7x7，使用了3的填充（padding=3），以确保输出特征图的空间维度与输入相同

    def forward(self, x):
        max = torch.max(x, 1, keepdim=True)[0]
        # 沿着输入x的第二个维度（假设维度顺序为[batch_size, channels, height, width]，则第二维是通道维）计算最大值，并保持输出维度不变（keepdim=True）。
        # torch.max返回的是一个元组，这里只取第一个元素（即最大值）赋值给变量max。
        mean = torch.mean(x, 1, keepdim=True)  # 沿着输入x的第二个维度计算均值，并保持输出维度不变。
        scale = torch.cat((max, mean), dim=1)
        # 将计算得到的最大值max和均值mean沿着第二个维度（通道维）拼接起来，形成一个新的特征图scale，其通道数为2（因为max和mean各占一个通道）。
        scale = self.spatial(scale)  # 将拼接后的特征图scale通过之前定义的卷积层self.spatial进行卷积操作，以生成一个空间门控图。
        scale = F.sigmoid(scale)  # 使用Sigmoid激活函数对卷积后的scale进行激活，将其值映射到(0, 1)区间内，作为空间门控的权重。
        return scale

class ChannelGate(nn.Module):

    def __init__(self, dim, reduction=8):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1),
            nn.GELU(),
            nn.Conv2d(dim // reduction, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.gate(x)


class BrightnessSpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, brightness):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out, brightness], dim=1)
        spatial_weights = self.conv(combined)
        spatial_weights = self.sigmoid(spatial_weights)
        return x * spatial_weights


class FrequencyDomainProcessor(nn.Module):

    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.in_channels = in_channels

        # 频域特征提取
        self.freq_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.size()

        # 傅里叶变换
        x_fft = torch.fft.rfft2(x, norm='ortho')

        # 分离幅度谱和相位谱
        magnitude = torch.abs(x_fft)
        phase = torch.angle(x_fft)

        # 对幅度谱进行对数变换增强细节
        magnitude_log = torch.log1p(magnitude)

        # 合并频域信息
        freq_feature = torch.cat([magnitude_log, phase], dim=1)

        # 生成频域注意力权重
        freq_weights = self.freq_conv(freq_feature)

        # 应用频域权重
        return x * freq_weights

class mamba(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod  # 初始化一个线性层（nn.Linear），并特别关注于其权重和偏置的初始化方式。
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization,计算权重初始化的标准差 dt_init_std
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant": # 将权重初始化为常数
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random": # 使用均匀分布初始化权重
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std) # 范围在 [-dt_init_std, dt_init_std] 之间。
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min) # 使用均匀分布生成一个在 [log(dt_min), log(dt_max)] 范围内的随机数。通过指数函数 torch.exp 将其转换为 [dt_min, dt_max] 范围内的值。
        ).clamp(min=dt_init_floor) # 使用 clamp 确保偏置值不小于 dt_init_floor。
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt)) # 使用公式 inv_dt = dt + torch.log(-torch.expm1(-dt)) 计算 dt 的逆 softplus 值。这个公式用于确保在反向传播时，通过 F.softplus 得到的值在 [dt_min, dt_max] 范围内。
        with torch.no_grad(): # 使用 torch.no_grad() 上下文管理器，确保在设置偏置时不记录梯度。
            dt_proj.bias.copy_(inv_dt) # 将计算得到的逆 softplus 值复制到线性层的偏置中。
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True # 设置偏置的 _no_reinit 属性为 True，防止在后续的重新初始化过程中覆盖此偏置。

        return dt_proj

    @staticmethod # 定义了一个名为 A_log_init 的函数，用于初始化一个对数形式的参数张量 A_log。
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True): # d_state: 状态维度的大小，决定初始张量的行数。d_inner: 内部维度的大小，决定初始张量的列数（通过重复扩展）。
        # S4D real initialization                                          copies: 要复制的次数，用于生成更高维度的张量。device: 张量应放置的设备（如 'cpu' 或 'cuda'）。merge: 如果 copies > 1，是否将复制的张量合并为一个大的张量。
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device), # 使用 torch.arange 生成一个从 1 到 d_state 的序列。
        # 使用 repeat 函数（假设这是一个自定义或来自库的函数，类似于 einops.repeat）将序列重复 d_inner 次，形成一个 d_inner x d_state 的张量。
            "n -> d n",  # "n -> d n" 表示将维度 n 张量扩展为 d x n 的张量，其中 d 是 d_inner。
            d=d_inner,
        ).contiguous() # .contiguous() 确保张量在内存中连续存储，提高访问效率。
        A_log = torch.log(A)  # Keep A_log in fp32 。对生成的张量 A 中的每个元素取自然对数，得到对数形式的张量 A_log。
        if copies > 1: # 如果 copies > 1，则将 A_log 重复 copies 次，生成一个更高维的张量。
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:  # 如果 merge 为 True，则将重复后的张量在第一个维度上展平（合并），形成一个连续的张量。
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log) # 将 A_log 张量包装为 nn.Parameter，以便在优化过程中不应用权重衰减。
        A_log._no_weight_decay = True
        return A_log

    @staticmethod # 初始化一个名为 D 的张量参数，通常在深度学习模型中作为某种"跳跃"（skip）参数使用。
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device) # 使用 torch.ones 生成一个全为 1 的张量，其形状为 (d_inner,)。
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W # L: 特征图的总像素数，即 H * W。
        K = 4 # K: 常量，表示某种分组或投影的数量。
        x_hwwh = torch.stack([x.view(B, -1, L), # 将输入张量 x 重塑为 (B, C, L)。
                              torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L) # 交换 x 的高度和宽度维度，然后重塑为 (B, C, L)。
                # torch.stack([...], dim=1): 将上述两个张量在新的维度上堆叠，形成 (B, 2, C, L)。view(B, 2, -1, L)进一步调整形状，可能是为了与后续操作兼容。
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)
        # torch.flip(x_hwwh, dims=[-1]): 在最后一个维度（即像素维度）上翻转 x_hwwh。
        # torch.cat([...], dim=1): 将原始的 x_hwwh 和翻转后的 x_hwwh 在新的维度上拼接，形成 (B, 4, C, L)。

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # xs.view(B, K, -1, L): 将 xs 重塑为 (B, K, C, L)。
        # torch.einsum: 使用爱因斯坦求和约定进行张量乘法，将 xs 投影到新的特征空间，形状变为 (B, K, C_out, L)，其中 C_out 由 self.x_proj_weight 的形状决定。

        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        # torch.split: 将 x_dbl 在特征维度上拆分为三个张量 dts、Bs 和 Cs，其大小分别为 self.dt_rank、self.d_state 和 self.d_state。

        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts.view(B, K, -1, L): 将 dts 重塑为适合与 self.dt_projs_weight 相乘的形状。
        # torch.einsum: 再次使用爱因斯坦求和约定进行张量乘法，将 dts 投影到新的特征空间。

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)
        # 将所有相关张量转换为浮点数，并调整形状以适应后续操作。
        # Ds、As 和 dt_projs_bias 从类属性中获取，并调整形状。

        # 执行选择性扫描操作
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L) # 提取 out_y 的第 2 和第 3 个通道，翻转后重塑。
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L) # 提取 out_y 的第 1 个通道，交换高度和宽度维度后重塑。
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L) #  提取 inv_y 的第 1 个通道，交换高度和宽度维度后重塑。

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y # 返回 out_y 的第 0 个通道、inv_y 的第 0 个通道、wh_y 和 invwh_y。

    def forward(self, x: torch.Tensor, **kwargs):
        x = x.permute(0, 2, 3, 1)  # (B,C,H,W) → (B,H,W,C)
        B, H, W, C = x.shape

        xz = self.in_proj(x) # self.in_proj 是一个线性层（ nn.Linear）将输入 x 投影到一个新的空间。输出张量的形状可能是 (B, H, W, 2*C')，其中 C' 是投影后的通道数。

        x, z = xz.chunk(2, dim=-1) # 将 xz 在最后一个维度（通道维度）上平均分成两部分，分别赋值给 x 和 z。x 和 z 的形状均为 (B, H, W, C')。
        x = x.permute(0, 3, 1, 2).contiguous() # 调整 x 的维度顺序，从 (B, H, W, C') 变为 (B, C', H, W)。这是为了符合卷积操作所需的输入格式（[B,C,H,W]）。

        x = self.act(self.conv2d(x)) # 经过CNN，然后经过激活函数SiLU

        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4 # 将 y1, y2, y3, y4 逐元素相加，得到融合后的张量 y。
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        # torch.transpose(y, dim0=1, dim1=2)：交换 y 的第 1 和第 2 个维度，将形状从 (B, C_out, H_out, W_out) 变为 (B, H_out, W_out, C_out)。
        # view(B, H, W, -1)：将张量重新调整为 (B, H, W, C') 的形状，其中 C' 是调整后的通道数。

        y = self.out_norm(y) # 对 y 应用归一化操作
        y = y * F.silu(z)
        # F.silu(z)：对 z 应用 SiLU 激活函数（也称为 Swish 激活函数），即 x * sigmoid(x)。z 的形状是 (B, H, W, C')。
        # y * F.silu(z)：将 y 与 F.silu(z) 逐元素相乘，实现调制操作。

        out = self.out_proj(y) # 对 y 应用一个线性投影。
        if self.dropout is not None:
            out = self.dropout(out) # 应用 dropout 操作，防止过拟合。
        out = out.permute(0, 3, 1, 2).contiguous()  # (B,H,W,C) → (B,C,H,W)
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=True):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class AdaptiveFrequencyPromptBlock(nn.Module):
    def __init__(self, in_channels, num_bands=3, spatial_att=True):

        super().__init__()
        self.num_bands = num_bands
        self.spatial_att = spatial_att

        # 自适应频段半径学习参数
        self.band_radii = nn.Parameter(torch.tensor([0.08, 0.3, 0.6]))
        self.band_softness = nn.Parameter(torch.tensor(1.0))  # 频带过渡平滑度

        # 频段注意力分支
        self.band_attentions = nn.ModuleList([
            self._build_attention_branch(in_channels)
            for _ in range(num_bands)
        ])

        # 空域注意力分支
        if spatial_att:
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(in_channels // 4, 1, 3, padding=1),
                nn.Sigmoid()
            )

        # 自适应融合参数
        self.fusion_weights = nn.Parameter(torch.ones(num_bands + (1 if spatial_att else 0)))

        # 频段间过渡卷积
        self.transition_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)

    def _build_attention_branch(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1),
            nn.SiLU(),
            nn.Conv2d(in_channels // 2, in_channels, 1),
            nn.Sigmoid()
        )

    def _create_adaptive_band_masks(self, H, W, device):

        # 创建归一化频率坐标网格
        y_freq = torch.fft.fftfreq(H, device=device).abs().view(-1, 1)  # [H, 1]
        x_freq = torch.fft.rfftfreq(W, device=device).abs().view(1, -1)  # [1, W_rfft]

        # 计算归一化径向距离 (0-1范围)
        max_freq = torch.sqrt(y_freq[-1] ** 2 + x_freq[0, -1] ** 2)
        radial_dist = torch.sqrt(y_freq ** 2 + x_freq ** 2) / max_freq

        # 自适应频带划分
        masks = []
        softness = torch.sigmoid(self.band_softness) * 0.2 + 0.01

        prev_radius = 0.0
        for i in range(self.num_bands):
            radius = torch.sigmoid(self.band_radii[i])
            mask = torch.sigmoid((radial_dist - prev_radius) / softness) * \
                   torch.sigmoid((radius - radial_dist) / softness)
            masks.append(mask)
            prev_radius = radius

        return masks

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device
        W_rfft = W // 2 + 1  # 实数FFT输出宽度

        # 频域分解
        x_freq = torch.fft.rfft2(x, norm='ortho')  # [B, C, H, W_rfft]

        # 生成自适应频带掩码
        masks = self._create_adaptive_band_masks(H, W, device)  # 传入原始W

        # 各频段处理
        band_outputs = []
        for i in range(self.num_bands):
            # 确保使用正确的掩码尺寸 [H, W_rfft]
            mask = masks[i]  # 已经是[H, W_rfft]形状

            # 应用频带掩码
            freq_comp = x_freq * mask.unsqueeze(0).unsqueeze(0)  # 广播到[B, C, H, W_rfft]

            # 逆变换
            spatial_comp = torch.fft.irfft2(freq_comp, s=(H, W), norm='ortho')

            # 应用注意力
            att_weights = self.band_attentions[i](spatial_comp)
            band_outputs.append(x * att_weights)

        # --- 空域注意力处理 ---
        if self.spatial_att:
            spatial_w = self.spatial_attention(x)
            band_outputs.append(x * spatial_w)

        # --- 自适应融合 ---
        weights = torch.softmax(self.fusion_weights, dim=0)
        fused_output = sum(w * out for w, out in zip(weights, band_outputs))

        # 添加过渡区域处理
        transition_feat = self.transition_conv(x)
        fused_output = fused_output + transition_feat

        # 残差连接
        return fused_output + x


#
# def visualize_frequency_bands(block, x, save_path=None):
#     """
#     可视化FrequencyPromptBlock处理后的频带信息
#
#     Args:
#         block: FrequencyPromptBlock实例
#         x: 输入张量 [B, C, H, W]
#         save_path: 图像保存路径，如果为None则直接显示
#     """
#     # 确保模型在评估模式
#     block.eval()
#
#     B, C, H, W = x.shape
#     device = x.device
#     W_rfft = W // 2 + 1  # 实数FFT输出宽度
#
#     # 获取频带掩码
#     with torch.no_grad():
#         masks = block._create_adaptive_band_masks(H, W, device)
#
#         # 计算FFT
#         x_freq = torch.fft.rfft2(x, norm='ortho')
#         freq_magnitude = torch.abs(x_freq)
#
#         # 应用掩码并计算逆变换
#         band_spatial = []
#         band_freq = []
#
#         for i, mask in enumerate(masks):
#             # 应用频带掩码
#             freq_comp = x_freq * mask.unsqueeze(0).unsqueeze(0)
#
#             # 存储频域信息
#             band_freq.append(torch.abs(freq_comp))
#
#             # 逆变换
#             spatial_comp = torch.fft.irfft2(freq_comp, s=(H, W), norm='ortho')
#             band_spatial.append(spatial_comp)
#
#     # 创建可视化图像
#     fig = plt.figure(figsize=(18, 12))
#     gs = GridSpec(3, 4, figure=fig)
#
#     # 原始图像
#     ax_orig = fig.add_subplot(gs[0, 0])
#     orig_img = x[0, 0].cpu().numpy() if C > 1 else x[0].mean(dim=0).cpu().numpy()
#     ax_orig.imshow(orig_img, cmap='gray')
#     ax_orig.set_title('Original Image')
#     ax_orig.axis('off')
#
#     # 原始频域
#     ax_orig_freq = fig.add_subplot(gs[0, 1])
#     orig_freq = torch.fft.fftshift(freq_magnitude[0].mean(dim=0).cpu())
#     orig_freq = torch.log(orig_freq + 1e-9)  # 对数变换增强可视化
#     ax_orig_freq.imshow(orig_freq, cmap='viridis')
#     ax_orig_freq.set_title('Original Frequency')
#     ax_orig_freq.axis('off')
#
#     # 频带掩码可视化
#     for i, mask in enumerate(masks):
#         ax_mask = fig.add_subplot(gs[0, 2 + i])
#         # 将掩码扩展到完整频率范围以便可视化
#         full_mask = torch.zeros(H, W)
#         full_mask[:, :W_rfft] = mask.cpu()
#         full_mask = torch.fft.fftshift(full_mask)
#         ax_mask.imshow(full_mask, cmap='hot')
#         ax_mask.set_title(f'Band {i + 1} Mask')
#         ax_mask.axis('off')
#
#     # 各频带的空间域表示
#     band_names = ['Low Frequency', 'Medium Frequency', 'High Frequency']
#     for i, spatial in enumerate(band_spatial):
#         ax_spatial = fig.add_subplot(gs[1, i])
#         spatial_img = spatial[0, 0].cpu().numpy() if C > 1 else spatial[0].mean(dim=0).cpu().numpy()
#         ax_spatial.imshow(spatial_img, cmap='gray')
#         ax_spatial.set_title(f'{band_names[i]} Spatial')
#         ax_spatial.axis('off')
#
#     # 各频带的频域表示
#     for i, freq in enumerate(band_freq):
#         ax_freq = fig.add_subplot(gs[2, i])
#         freq_vis = torch.fft.fftshift(freq[0].mean(dim=0).cpu())
#         freq_vis = torch.log(freq_vis + 1e-9)  # 对数变换增强可视化
#         ax_freq.imshow(freq_vis, cmap='viridis')
#         ax_freq.set_title(f'{band_names[i]} Frequency')
#         ax_freq.axis('off')
#
#     # 注意力权重可视化（如果有）
#     if hasattr(block, 'spatial_attention') and block.spatial_att:
#         with torch.no_grad():
#             spatial_att = block.spatial_attention(x)
#
#         ax_att = fig.add_subplot(gs[1, 3])
#         att_img = spatial_att[0, 0].cpu().numpy()
#         ax_att.imshow(att_img, cmap='hot')
#         ax_att.set_title('Spatial Attention')
#         ax_att.axis('off')
#
#     plt.tight_layout()
#
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     else:
#         plt.show()
#
#     plt.close()



class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads,prompt=False,):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm2d(dim)
        self.attn = Attention(dim, num_heads, prompt)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = FeedForward(dim)
    def forward(self, x):
        # x = self.norm1(x)
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class SE(nn.Module):
    def __init__(self, c1, c2, ratio=16):
        super(SE, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 使用全局平均池化，将输入特征图的空间维度压缩为1x1，从而得到每个通道的全局信息。
        self.l1 = nn.Linear(c1, c1 // ratio, bias=False)  # 第一个全连接层，用于压缩特征维度，将通道数从c1减少到c1 // ratio。
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数，增加非线性。
        self.l2 = nn.Linear(c1 // ratio, c1, bias=False)  # 第二个全连接层，将维度从c1 // ratio恢复到c1，以匹配原始输入特征图的通道数。
        self.sig = nn.Sigmoid()  # Sigmoid激活函数，将输出映射到(0, 1)区间，用于生成每个通道的权重。

    def forward(self, x):
        b, c, _, _ = x.size()  # 获取输入特征图x的批次大小b、通道数c以及空间维度（这里不关心具体大小）。
        y = self.avgpool(x).view(b, c)  # 应用全局平均池化，并将结果重塑为二维张量，以便输入到全连接层。
        # 接着，通过两个全连接层和ReLU、Sigmoid激活函数，生成每个通道的权重y。
        y = self.l1(y)
        y = self.relu(y)
        y = self.l2(y)
        y = self.sig(y)
        y = y.view(b, c, 1, 1)  # 将权重y重塑为与输入特征图x相同的空间维度（即1x1），以便进行逐元素乘法。
        return x * y.expand_as(x)  # 使用expand_as(x)将权重y扩展到与输入特征图x相同的形状，然后逐元素相乘，以重新标定特征图的每个通道。


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class CAB(nn.Module):
    def __init__(self, c, compress_ratio=3,squeeze_factor=30):
        super(CAB, self).__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(c, c // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(c // compress_ratio, c, 3, 1, 1),
            CALayer(c, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)


class MTBlock(nn.Module):
    def __init__(self, c,
                 bias,
                 attn_drop_rate: float = 0,
                 d_state: int = 16,
                 expand: float = 2.,
                 DW_Expand=2,
                 expansion_ratio=4,
                 kernel_size=3,
                 **kwargs
                ):

        super().__init__()

        dw_channel = c * DW_Expand

        self.norm1 = LayerNorm2d(c)  # 4D

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,bias=True)

        self.relu = nn.LeakyReLU(0.1, inplace=True)

        self.conv3x3 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)

        self.reduce_chan = nn.Conv2d(dw_channel , c, kernel_size=1, bias=True)

        self.mamba = mamba(d_model=c, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)

        self.norm2 = LayerNorm2d(c)  # 4D

        # self.attention = HiLo_prompt(c, num_heads, prompt)

        self.attention = SE(int(c), int(c))

        self.norm3 = LayerNorm2d(c)

        # self.scblock = SCBlock(c, expansion_ratio, kernel_size)

        self.act1 = nn.LeakyReLU(0.1, inplace=True)

        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)

        self.dropout1 = nn.Dropout(attn_drop_rate) if attn_drop_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(attn_drop_rate) if attn_drop_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x):

        x_in = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv3x3(x)
        x = self.reduce_chan(x)
        x1 = self.attention(x)
        # x1 = self.norm3(x1)
        # x1 = self.scblock(x1)
        x1 = self.dropout1(x1)

        x2 = self.mamba(x)
        x2 = self.dropout2(x2)
        y = (x1 + x2) * self.beta
        y = self.act1(y)
        y = self.conv2(y)
        out = x_in + y * self.gamma

        return out


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 6, self.dim * 6 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 6 // reduction, self.dim * 2),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(B, self.dim * 2)
        std = torch.std(x, dim=(2, 3), keepdim=True).view(B, self.dim * 2)
        max = self.max_pool(x).view(B, self.dim * 2)
        y = torch.cat((avg, std, max), dim=1)  # B 6C
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)  # 2 B C 1 1
        return channel_weights


class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SpatialWeights, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // reduction, 2, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)  # B 2C H W
        spatial_weights = self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)  # 2 B 1 H W
        return spatial_weights

class FCM(nn.Module):
    def __init__(self, dim, reduction=1, eps=1e-8):
        super(FCM, self).__init__()
        # 自定义可训练权重参数
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)

        self.apply(self._init_weights)

    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)

        spatial_weights = self.spatial_weights(x1, x2)
        x1_1 = x1 + fuse_weights[0] * spatial_weights[1] * x2
        x2_1 = x2 + fuse_weights[0] * spatial_weights[0] * x1

        channel_weights = self.channel_weights(x1_1, x2_1)

        main_out = x1_1 + fuse_weights[1] * channel_weights[1] * x2_1
        aux_out = x2_1 + fuse_weights[1] * channel_weights[0] * x1_1
        return main_out+aux_out


class Attention(nn.Module):
    def __init__(self, dim, num_heads, is_prompt=False, bias=True):
        super(Attention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.is_prompt = is_prompt
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.prompt = nn.Parameter(torch.ones(num_heads, dim//num_heads, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def with_prompt(self, tensor, prompt):
        return tensor if prompt is None else tensor + prompt

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        if self.is_prompt:
            prompt = self.prompt
            q = self.with_prompt(q, prompt)
            k = self.with_prompt(k, prompt)
            v = self.with_prompt(v, prompt)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

@ARCH_REGISTRY.register()
class MTPNet(nn.Module):
    def __init__(self,
                 img_ch=3,
                 width=32,
                 # band_ratios=[0.08, 0.3, 0.6],
                 spatial_att=True,
                 num_bands=3,
                 prompt=True,
                 enc_nums=[1, 1, 1, 32],
                 dec_nums=[1, 1, 1, 1],
                 middle_num=1,
                 heads=[4, 8],
                 latent_head=16,
                 output_ch=3,
                 bias = False,
                 LayerNorm_type='WithBias'  ## Other option 'BiasFree'
                 ):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_ch, out_channels=width, kernel_size=3, padding=1, stride=1,groups=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=output_ch, kernel_size=3, padding=1, stride=1,groups=1,bias=True)

        chan = width

        self.encoder_level1 = nn.Sequential(
            *[MTBlock(c=chan, expansion_ratio=4, kernel_size=3, bias=bias) for _ in range(enc_nums[0])])

        self.down1_2 = nn.Conv2d(chan, chan * 2 ** 1, 2, 2)

        self.encoder_level2 = nn.Sequential(
            *[MTBlock(c=2 * chan, expansion_ratio=4, kernel_size=3, bias=bias) for _ in range(enc_nums[1])])

        self.down2_3 = nn.Conv2d(chan * 2 ** 1, chan * 2 ** 2, 2, 2)

        self.encoder_level3 = nn.Sequential(
            *[TransformerBlock(dim=int(chan * 2 ** 2), num_heads=heads[0]) for i in range(enc_nums[2])])
        self.down3_4 = nn.Conv2d(chan * 2 ** 2, chan * 2 ** 3, 2, 2)

        self.encoder_level4 = nn.Sequential(
            *[TransformerBlock(dim=int(chan * 2 ** 3), num_heads=heads[1]) for i in range(enc_nums[3])])
        self.down4_5 = nn.Conv2d(chan * 2 ** 3, chan * 2 ** 4, 2, 2)

        self.latent = nn.Sequential(
            *[TransformerBlock(dim=int(chan * 2 ** 4), num_heads=latent_head) for i in range(middle_num)])

        self.decoder_level4 = nn.Sequential(
            *[TransformerBlock(dim=int(chan * 2 ** 3), num_heads=heads[1], prompt=prompt) for i in range(dec_nums[0])])

        self.decoder_level3 = nn.Sequential(
            *[TransformerBlock(dim=int(chan * 2 ** 2), num_heads=heads[0], prompt=prompt) for i in range(dec_nums[1])])

        self.decoder_level2 = nn.Sequential(
            *[MTBlock(c=2 * chan,  expansion_ratio=4, kernel_size=3, bias=bias) for _ in range(dec_nums[2])])

        self.decoder_level1 = nn.Sequential(
            *[MTBlock(c=chan, expansion_ratio=4, kernel_size=3, bias=bias) for _ in range(dec_nums[3])])

        self.up5_4 = self._make_upsample(chan * 16, chan * 8)
        self.up4_3 = self._make_upsample(chan * 8, chan * 4)
        self.up3_2 = self._make_upsample(chan * 4, chan * 2)
        self.up2_1 = self._make_upsample(chan * 2, chan)

        self.FPBlock1 = AdaptiveFrequencyPromptBlock(chan, num_bands=3, spatial_att=True)

        self.FPBlock2 = AdaptiveFrequencyPromptBlock(2 * chan, num_bands=3, spatial_att=True)

        self.FPBlock3 = AdaptiveFrequencyPromptBlock(4 * chan, num_bands=3, spatial_att=True)

        self.FPBlock4 = AdaptiveFrequencyPromptBlock(8 * chan, num_bands=3, spatial_att=True)

        self.FPBlock5 = AdaptiveFrequencyPromptBlock(16 * chan,num_bands=3, spatial_att=True)

        self.fusion1 = FCM(8 * chan)

        self.fusion2 = FCM(4 * chan)

        self.fusion3 = FCM(2 * chan)

        self.fusion4 = FCM(chan)

    def _make_upsample(self, in_ch, out_ch):
        """更高效的上采样模块"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

    def forward(self, inp_img):
        x = self.intro(inp_img)

        # 编码器路径
        enc1 = self.encoder_level1(x)
        enc1_freq = self.FPBlock1(enc1)

        enc2 = self.encoder_level2(self.down1_2(enc1))
        enc2_freq = self.FPBlock2(enc2)

        enc3 = self.encoder_level3(self.down2_3(enc2))
        enc3_freq = self.FPBlock3(enc3)

        enc4 = self.encoder_level4(self.down3_4(enc3))
        enc4_freq = self.FPBlock4(enc4)

        latent = self.latent(self.down4_5(enc4))
        latent_freq = self.FPBlock5(latent)

        # 解码器路径
        dec4 = self.decoder_level4(self.fusion1(self.up5_4(latent_freq) , enc4_freq))
        dec3 = self.decoder_level3(self.fusion2(self.up4_3(dec4) , enc3_freq))
        dec2 = self.decoder_level2(self.fusion3(self.up3_2(dec3) , enc2_freq))
        dec1 = self.decoder_level1(self.fusion4(self.up2_1(dec2) , enc1_freq))
        # dec1 = self.decoder_level1(self.fusion4(self.up2_1(dec2) + enc1_freq + mid_feat))

        out = self.ending(dec1) + inp_img
        return out

    # 添加计算参数和FLOPs的方法
    # def calculate_params_flops(self, input_size=(1, 3, 256, 256), verbose=False):
    #     """
    #     计算模型的参数数量和FLOPs
    #     Args:
    #         input_size (tuple): 输入张量的尺寸 (batch, channels, height, width)
    #         verbose (bool): 是否打印详细统计信息
    #     Returns:
    #         params (float): 参数量 (百万)
    #         flops (float): FLOPs (十亿次操作)
    #     """
    #     # 创建随机输入
    #     device = next(self.parameters()).device
    #     input_tensor = torch.randn(input_size).to(device)
    #
    #     # 计算参数量
    #     total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    #     params_m = total_params / 1e6  # 转换为百万
    #
    #     # 计算FLOPs
    #     flops_counter = FlopCountAnalysis(self, input_tensor)
    #     if verbose:
    #         print(flop_count_table(flops_counter))
    #
    #     flops_g = flops_counter.total() / 1e9  # 转换为十亿次操作
    #
    #     return params_m, flops_g


if __name__ == "__main__":
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建模型实例并移动到设备
    model = MTPNet(
        img_ch=3,
        width=32,
        enc_nums=[1, 1, 1, 32],
        dec_nums=[1, 1, 1, 1],
        middle_num=1,
        heads=[4, 8],
        latent_head=16
    ).to(device)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_m = total_params / 1e6  # 转换为百万

    print(f"模型参数: {params_m:.2f} M")

    # 尝试计算FLOPs，如果失败则跳过
    try:
        # 创建随机输入并移动到设备
        input_tensor = torch.randn(1, 3, 256, 256).to(device)

        # 计算FLOPs
        flops = FlopCountAnalysis(model, input_tensor)
        flops.unsupported_ops_warnings(False)  # 关闭警告
        flops_g = flops.total() / 1e9  # 转换为 GFLOPs
        print(f"计算量: {flops_g:.2f} GFLOPs")
    except Exception as e:
        print(f"发生错误: {e}")

        # print(f"计算量: {flops_g:.2f} GFLOPs")
