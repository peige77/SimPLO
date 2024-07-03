import time
import math
import copy
import warnings
from functools import partial
from typing import Optional, Callable,Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from timm.models.registry import register_model
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

# import mamba_ssm.selective_scan_fn (in which causal_conv1d is needed)
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

try:
    "sscore acts the same as mamba_ssm"
    SSMODE = "sscore"
    import selective_scan_cuda_core
    print("Using \"selective_scan_cuda_core\"")
except Exception as e:
    warnings.warn(f"{e}\n\"selective_scan_cuda_core\" not found, use default \"selective_scan_cuda\" instead.")
    # print(e, flush=True)
    SSMODE = "mamba_ssm"
    import selective_scan_cuda

#计算selective scan的flops(浮点运算数),衡量计算复杂度
"""
B:批次大小(batch size)。
L:序列长度或图像的宽度和高度(sequence length or image width and height)。
D:特征维度(feature dimension)。
N:selective_scan 操作中的一个参数，可能与分组或输出的维度有关。
with_D:布尔值，指示是否考虑 D 参数相关的计算。
with_Z:布尔值，指示是否考虑 z 参数相关的计算。
with_Group:布尔值，指示是否考虑分组(grouping)计算。
with_complex:布尔值，指示是否处理复数运算，但代码中 assert not with_complex 表明当前版本不考虑复数。
"""
def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    import numpy as np
    
    # fvcore.nn.jit_handles  使用numpy的einsum_path函数计算flops
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop
    

    assert not with_complex

    flops = 0 # below code flops = 0
    #函数中的 u、delta、A、B、C、D 和 z 代表不同的输入张量或变量，它们在注释中有提及，但没有在函数参数中明确给出。这可能是为了模拟 selective_scan 操作的不同部分的计算。
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """
    
    in_for_flops = B * D * N   
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops 
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """
    
    return flops

#计算selective scan的flops(浮点运算数),衡量计算复杂度
def selective_scan_flop_jit(inputs, outputs):
    # xs, dts, As, Bs, Cs, Ds (skip), z (skip), dt_projs_bias (skip) 顺序
    #使用assert语句来验证传入的inputs是否符合预期的命名规范
    assert inputs[0].debugName().startswith("xs") # (B, D, L)
    assert inputs[2].debugName().startswith("As") # (D, N)
    assert inputs[3].debugName().startswith("Bs") # (D, N)
    with_Group = len(inputs[3].type().sizes()) == 4
    with_D = inputs[5].debugName().startswith("Ds")
    if not with_D:
        with_z = inputs[5].debugName().startswith("z")
    else:
        with_z = inputs[6].debugName().startswith("z")
    B, D, L = inputs[0].type().sizes()  #提取输入张量的形状信息
    N = inputs[2].type().sizes()[1]  #表示输入张量的第二个维度的大小
    #计算selective_scan的flops(浮点运算数),衡量计算复杂度
    flops = flops_selective_scan_ref(B=B, L=L, D=D, N=N, with_D=with_D, with_Z=with_z, with_Group=with_Group)
    return flops


#实现了一个图像到补丁（patch）嵌入的转换
class PatchEmbed2D(nn.Module):
    #每个补丁是一个4x4的小图像块，输入通道数是3，输出通道数是96，不使用归一化
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        #检查patch_size是否是整数，如果是整数，将其转换为元组，以支持正方形的patch
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        #使用卷积层实现图像到补丁（patch）嵌入的转换，用于将输入图像的每个补丁映射到embed_dim维度的空间
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None
    #前向传播，实现了一个图像到补丁（patch）嵌入的转换
    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)#进行线性转换，对输入张量的维度进行重新排列，将通道维度放到最后
        if self.norm is not None:
            x = self.norm(x)
        return x    #形状为B H/4 W/4 96

#实现了一种特征融合操作，以减少特征图的分辨率并增加通道数，进行特征图的降采样
class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)#线性变换，将输入张量的最后一个维度的大小从4*dim变为2*dim
        self.norm = norm_layer(4 * dim)#归一化层

    def forward(self, x):
        B, H, W, C = x.shape #获取输入特征x的批次大小B、高度H、宽度W和通道数C

        SHAPE_FIX = [-1, -1] #用于在必要时调整特征图的形状
        #如果特征图的高度和宽度是奇数，则打印警告信息并计算调整后的特征图的高度和宽度
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        #将输入特征图x按照通道数C进行切片，得到4个特征图x0、x1、x2、x3，对应原特征图4个象限
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        #如果特征图的高度和宽度是奇数，则调整特征图的形状
        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
        
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C 将4个特征图沿着通道维度拼接
        x = x.view(B, H//2, W//2, 4 * C)  # B H/2*W/2 4*C 将拼接后的特征图的形状调整为B H/2*W/2 4*C

        x = self.norm(x)    #对特征图进行归一化
        x = self.reduction(x)   #线性层将特征图的通道数减少到原来的一半

        return x    #最终特征图形状为B H/2*W/2 2*C
#实现了一个特征扩展操作，以增加特征图的分辨率并减少通道数，进行特征图的上采样
class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm): #dim_scale=2表示通道数扩展为原来的2倍
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(    #线性层将输入张量的最后一个维度的大小从dim扩展为2*dim
            dim, 2*dim, bias=False) if dim_scale == 2 else nn.Identity()    #如果dim_scale=2，使用线性层，否则使用恒等映射
        self.norm = norm_layer(dim // dim_scale)    #归一化层，输入通道数为dim//dim_scale

    #前向传播，实现了一个特征扩展操作，以增加特征图的分辨率并减少通道数，进行特征图的上采样
    def forward(self, x):
        x = self.expand(x)  #扩展成2倍的通道数
        B, H, W, C = x.shape    #获取输入特征x的批次大小B、高度H、宽度W和通道数C
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)   #将特征图的高度和宽度扩展为原来的2倍，通道数减少为原来的1/4
        x= self.norm(x) 

        return x    #最终特征图形状为B H*2 W*2 C/2

#用于在图像模型的最后阶段将特征图的空间分辨率扩大四倍，这种操作通常用于从模型的高维特征中恢复出与输入图像相同分辨率的输出。
class FinalPatchExpand_X4(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):  #dim_scale=4表示通道数扩展为原来的4倍
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)    #将输入特征的通道数扩展到原来的16倍
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim) #归一化层，输入通道数为output_dim，即dim

    def forward(self, x):

        x = self.expand(x)  #将输入特征的通道数扩展到原来的16倍
        B, H, W, C = x.shape    #获取输入特征x的批次大小B、高度H、宽度W和通道数C
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2)) #将特征图的高度和宽度扩展为原来的4倍，通道数减少为原来的1/16
        x= self.norm(x)

        return x    #最终特征图形状为B H*4 W*4 C/4

#实现了一个基于注意力机制的多头自注意力层
class SS2D(nn.Module):
    def __init__(
        self,
        d_model=96,
        d_state=16,   #状态维度
        ssm_ratio=2.0,
        # d_state="auto", # 20240109
        d_conv=3, 
        # dt init ==============  
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
        simple_init=False,
        directions=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype} 
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        d_expand = int(ssm_ratio * d_model)
        d_inner = d_expand
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.K = len(MultiScanVSSM.ALL_CHOICES) if directions is None else len(directions)
        self.K2 = self.K

        # in proj =======================================
        self.in_proj = nn.Linear(self.d_model, d_inner * 2, bias=bias, **factory_kwargs)
        self.act = nn.SiLU()    #激活函数为SiLU

        # conv =======================================
        self.conv2d = nn.Conv2d(    #二维卷积层，用于实现特征图的卷积操作
            in_channels=d_inner,
            out_channels=d_inner,
            groups=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        # rank ratio =====================================
        self.ssm_low_rank = False
        if d_inner < d_expand:
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)

        # x proj =======================================
        # self.x_proj = (
        #     nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        #     nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        #     nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        #     nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        # )
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        #将4个线性层的权重和偏置拼接在一起
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj   #删除x_proj

        #初始化4个特征扩展操作
        #dt projs =======================================
        self.dt_projs = [
            self.dt_init(self.dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
        del self.dt_projs
        
        #初始化A_log和D
        #A, D =======================================
        self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.K2, merge=True) # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=self.K2, merge=True) # (K * D)

        # Local Mamba
        self.multi_scan = MultiScanVSSM(d_expand, choices=directions)

        if simple_init:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.K2 * d_inner)))
            self.A_logs = nn.Parameter(torch.randn((self.K2 * d_inner, self.d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner))) 

        self.forward_core = self.forward_core
        #self.forward_core = self.forward_corev0
        # self.forward_core = self.forward_corev0_seq
        # self.forward_core = self.forward_corev1

        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None    #dropout层

    @staticmethod   #静态方法，用于初始化特征扩展操作
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs) #初始化一个特殊的线性层

        # Initialize special dt projection to preserve variance at initialization 初始化特殊的dt投影以在初始化时保留方差
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max 初始化dt偏置，使得F.softplus(dt_bias)在dt_min和dt_max之间
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759 softplus的反函数
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit 我们的初始化会将所有Linear.bias设置为零，需要将这个标记为_no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    #静态方法，用于初始化A_log，代表了一个对数空间中的参数
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization A_log = log(1, 2, ..., d_state) S4D真实初始化A_log = log(1, 2, ..., d_state)
        A = repeat( #将输入张量沿着指定的维度复制多次
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),   
            "n -> d n",
            d=d_inner,
        ).contiguous()  
        A_log = torch.log(A)  # Keep A_log in fp32  计算序列的自然对数，结果保持为32位浮点数（fp32）
        # Repeat and merge A_log if needed 如果需要，重复和合并A_log
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log) 
        A_log._no_weight_decay = True   #不参与权重衰减
        return A_log

    @staticmethod
    #静态方法，用于初始化D，它代表了一个“skip”连接的参数
    #方法的执行流程与A_log_init类似，但是它创建的是一个所有元素为1的参数，而不是对数参数。这个参数同样被设置为不进行权重衰减。
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, nrows=-1, channel_first=False):
        nrows = 1
        # if not channel_first:
        #     x = x.permute(0, 3, 1, 2).contiguous() # (b, h, w, c) -> (b, c, h, w)
        if self.ssm_low_rank:
            x = self.in_rank(x)
        # print("x_proj_weight shape:", self.x_proj_weight.shape) 
        # print("x shape:", x.shape) 
        # print("dt_projs_weight shape:", self.dt_projs_weight.shape)
        x = multi_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, self.out_norm,
            nrows=nrows, delta_softplus=True, multi_scan=self.multi_scan,
        )
        if self.ssm_low_rank:
            x = self.out_rank(x)
        return x

    # 首先对输入特征x进行一系列的变换，包括重新排列、翻转和分割，以准备进行选择性扫描。然后，它调用selective_scan_fn（未在代码中定义），一个执行选择性扫描操作的函数，并将结果通过一系列变换来恢复形状并进行归一化。
    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn #选择性扫描操作

        B, C, H, W = x.shape
        L = H * W   #特征图的长度
        K = 4   #4个特征扩展操作

        #通过改变视角（view）和拼接（torch.cat）操作，创建一个包含原始特征和它们转置后特征的组合
        #形成x_hwwh张量，其维度为[B, 2, -1, L]，其中L = H * W，并且2来自于原始特征和转置后特征的组合
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)   
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        #使用torch.einsum和self.x_proj_weight对特征进行线性投影，然后分割（torch.split）成dts、Bs和Cs三个部分
        #xs是通过view(B, K, -1, L)被调整为[B, K, D, L] k=4, d=96 
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y).to(x.dtype)

        return y

    
    #对输入特征进行相同的预处理，但随后循环地对每个子特征块执行选择性扫描，并将结果组合起来形成最终的输出
    def forward_corev0_seq(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        #不同于forward_corev0，这里对每个子特征块循环地执行选择性扫描，并将结果组合起来形成最终的输出
        out_y = []
        for i in range(4):
            yi = self.selective_scan(
                xs[:, i], dts[:, i], 
                As[i], Bs[:, i], Cs[:, i], Ds[i],
                delta_bias=dt_projs_bias[i],
                delta_softplus=True,
            ).view(B, -1, L)
            out_y.append(yi)
        out_y = torch.stack(out_y, dim=1)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y).to(x.dtype)

        return y
    #使用不同的函数selective_scan_fn_v1（同样未在代码中定义）。它在执行选择性扫描后，将结果转换为浮点数，并进行类似的后处理。
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.view(B, K, -1, L) # (b, k, d_state, l)
        
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        Ds = self.Ds.view(-1) # (k * d)
        dt_projs_bias = self.dt_projs_bias.view(-1) # (k * d)

        # print(self.Ds.dtype, self.A_logs.dtype, self.dt_projs_bias.dtype, flush=True) # fp16, fp16, fp16

        #不同之处在于，这里使用了selective_scan_fn_v1函数，它在执行选择性扫描后，将结果转换为浮点数，并进行类似的后处理
        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float16

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0].float() + inv_y[:, 0].float() + wh_y.float() + invwh_y.float()
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y).to(x.dtype)

        return y

    #是SS2D模块的主要前向传播入口点 无需修改
    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x) #处理输入特征x
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)   将处理后的特征x分割成两部分x和z

        x = x.permute(0, 3, 1, 2).contiguous()  # (b, d, h, w) 将特征x的维度重新排列
        x = self.act(self.conv2d(x)) # (b, d, h, w) 对特征图进行卷积操作
        y = self.forward_core(x) #这里使用forward_corev0函数
        y = y * F.silu(z)   #对特征图y进行激活函数操作
        out = self.out_proj(y)  #对特征图y进行线性变换，输出投影层
        if self.dropout is not None:
            out = self.dropout(out)
        return out

#代表一个变分自注意力模块，它包含了一个SS2D模块和一个可选的DropPath模块 这是方法中的vssblock(需要改)
class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,    #隐层维度
        drop_path: float = 0,   #DropPath概率
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),   #归一化层
        # =============================
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_simple_init=False,
        # =============================
        use_checkpoint: bool = False,
        directions=None,
        #attn_drop_rate: float = 0,  #注意力机制的Dropout概率
        #d_state: int = 16,  #状态维度
        **kwargs,
    ):
        super().__init__()
        self.norm = norm_layer(hidden_dim)
        self.use_checkpoint = use_checkpoint
        # print(f"type(directions): {type(directions)}, directions: {directions}")
        self.self_attention = SS2D(
            d_model=hidden_dim, 

            d_state=ssm_d_state, 
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            # ==========================
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            # ==========================
            dropout=ssm_drop_rate,
            # bias=False,
            # ==========================
            # dt_min=0.001,
            # dt_max=0.1,
            # dt_init="random",
            # dt_scale="random",
            # dt_init_floor=1e-4,
            simple_init=ssm_simple_init,
            # ==========================
            directions=directions,
            **kwargs
        )   #自注意力机制
        self.drop_path = DropPath(drop_path)

    #如何根据输入特征input计算输出特征，这里的计算过程包括了DropPath、自注意力机制和残差连接
    def _forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.norm(input)))
        return x
    
    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)

#用于构建类似于Swin Transformer的模型，但是使用了VSSBlock代替了Swin Transformer中的基本Transformer块
class VSSLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.  输入通道数
        depth (int): Number of blocks.  块的数量
        drop (float, optional): Dropout rate. Default: 0.0  Dropout概率
        attn_drop (float, optional): Attention dropout rate. Default: 0.0   注意力机制的Dropout概率
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0 随机深度率
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm    归一化层
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None    下采样层
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False. 是否使用检查点来节省内存
    """ 

    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        downsample=None, 
        use_checkpoint=False, 
        d_state=16,
        directions=None,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        #创建一个包含depth个VSSBlock的模块列表
        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,   #DropPath概率
                norm_layer=norm_layer,
                ssm_drop_rate=attn_drop,
                ssm_d_state=d_state,
                directions=directions[i] if directions is not None else None

            )
            for i in range(depth)]) #depth个VSSBlock
        
        #使用了Kaiming Uniform分布来初始化out_proj.weight参数，但这里有一个注释，表明这个初始化实际上可能不会应用，因为它在VSSM中被覆盖。
        if True: # is this really applied? Yes, but been overriden later in VSSM! #这个真的适用吗？是的，但后来在VSSM中被覆盖了！
            def _init_weights(module: nn.Module):   #初始化权重
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:  #如果downsample不为空
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)    
        else:
            self.downsample = None


    def forward(self, x):   
        for blk in self.blocks:   #对每个VSSBlock进行循环
            if self.use_checkpoint: #是否使用检查点省内存
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)  
        
        if self.downsample is not None:
            x = self.downsample(x)

        return x

#用于逐步恢复特征图的空间分辨率，在最后用的是up_sample而不是down_sample
class VSSLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        Upsample (nn.Module | None, optional): Upsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        upsample=None, 
        use_checkpoint=False, 
        d_state=16,
        directions=None,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        # print("Depth:", depth)
        # print("Directions Length:", len(directions))

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                ssm_drop_rate=attn_drop,
                ssm_d_state=d_state,
                directions=directions[i] if directions is not None else None
            )
            for i in range(depth)])
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if upsample is not None:
            self.upsample = PatchExpand(dim, dim_scale=2, norm_layer=nn.LayerNorm)
        else:
            self.upsample = None


    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        
        if self.upsample is not None:
            x = self.upsample(x)

        return x

#结合了编码器、瓶颈层和解码器，通常用于图像分割或其他需要特征层次化表示的任务
class VSSM(nn.Module):
    def __init__(
            self, 
            patch_size=4, 
            in_chans=1, 
            num_classes=4, 
            depths=[2, 2, 9, 2], 
            dims=[96, 192, 384, 768], 
            d_state=16, 
            drop_rate=0., 
            attn_drop_rate=0., 
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm, 
            patch_norm=True,
            use_checkpoint=False, 
            final_upsample="expand_first", 
            directions=None,
            **kwargs
        ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)   #层数 4层
        if isinstance(dims, int):   #如果dims是int类型
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)] #dims是一个列表，包含了每一层的维度
        self.embed_dim = dims[0]    #嵌入维度
        self.num_features = dims[-1]    #特征数
        self.num_features_up = int(dims[0] * 2)  #上采样后的特征数
        self.dims = dims
        self.final_upsample = final_upsample    #最终上采样

        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule  随机深度衰减规则

        # build encoder and bottleneck layers 编码器和瓶颈层
        self.layers = nn.ModuleList()   #存储子模块
        for i_layer in range(self.num_layers):      #对每一层进行循环
            layer = VSSLayer(   #为每个i_layer创建一个VSSLayer实例
                # dim=dims[i_layer], #int(embed_dim * 2 ** i_layer)
                dim = int(dims[0] * 2 ** i_layer),  #dims[0]：表示模型输入层的通道数，即嵌入维度（embed_dim）。2 ** i_layer：是一个指数运算，表示2的i_layer次方。
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state, # 20240109
                drop=drop_rate, 
                attn_drop=attn_drop_rate,   #注意力机制的Dropout概率
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], #随机深度衰减规则
                norm_layer=norm_layer,  
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None, #如果当前层不是最后一个编码器层，则使用PatchMerging2D作为下采样操作
                use_checkpoint=use_checkpoint,
                directions=None if directions is None else directions[sum(depths[:i_layer]):sum(depths[:i_layer + 1])]
            )
            self.layers.append(layer)

        # build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()  #用于存储线性层或恒等映射（nn.Identity），这些层用于调整特征维度。
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2*int(dims[0]*2**(self.num_layers-1-i_layer)),    #为每个i_layer创建一个解码器层和相应的线性层。
            int(dims[0]*2**(self.num_layers-1-i_layer))) if i_layer > 0 else nn.Identity()  #如果i_layer>0，则使用nn.Linear；否则使用nn.Identity。
            if i_layer ==0 :
                layer_up = PatchExpand(dim=int(self.embed_dim * 2 ** (self.num_layers-1-i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = VSSLayer_up(
                    dim= int(dims[0] * 2 ** (self.num_layers-1-i_layer)),
                    depth=depths[(self.num_layers-1-i_layer)],
                    d_state=math.ceil(dims[0] / 6) if d_state is None else d_state, # 20240109
                    drop=drop_rate, 
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:(self.num_layers-1-i_layer)]):sum(depths[:(self.num_layers-1-i_layer) + 1])],
                    norm_layer=norm_layer,
                    upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoint,
                    directions=None if directions is None else directions[sum(depths[:(self.num_layers-1-i_layer)]):sum(depths[:(self.num_layers-1-i_layer) + 1])] 
                )
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        if self.final_upsample == "expand_first":   #如果最终上采样是“expand_first”
            print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(dim_scale=4,dim=self.embed_dim)   #使用FinalPatchExpand_X4作为最终上采样
            self.output = nn.Conv2d(in_channels=self.embed_dim,out_channels=self.num_classes,kernel_size=1,bias=False)

        self.apply(self._init_weights)  #初始化权重



    #初始化权重 接受一个 nn.Module 类型的参数 m，这个参数代表模块的实例
    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless
        
        Conv2D is not intialized !!!
        """
        # print(m, getattr(getattr(m, "weight", nn.Identity()), "INIT", None), isinstance(m, nn.Linear), "======================")
        if isinstance(m, nn.Linear):    #如果m是nn.Linear类型
            trunc_normal_(m.weight, std=.02)    #对m.weight进行截断正态分布初始化，标准差为0.02
            if isinstance(m, nn.Linear) and m.bias is not None:  #如果m是nn.Linear类型且m.bias不为空
                nn.init.constant_(m.bias, 0)    #对m.bias进行常数初始化，值为0
        elif isinstance(m, nn.LayerNorm):   #如果m是nn.LayerNorm类型
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    #Encoder and Bottleneck 编码器和通过瓶颈层 瓶颈层就是最深的那一层
    def forward_features(self, x):  
        x = self.patch_embed(x) # B H W C 对输入特征x进行patch embedding操作

        x_downsample = []
        for layer in self.layers:   #对每个编码器层进行循环 
            x_downsample.append(x)  #将x添加到x_downsample中
            x = layer(x)    #将x设置为当前层的输出，即通过当前层处理后的新特征图
        x = self.norm(x)  # B H W C 所有编码器层处理完成后
        return x, x_downsample  #返回最终的特征图x和所有下采样特征图x_downsample，x_downsample：包含所有下采样特征图的列表，这些特征图将用于解码器部分的逐层恢复。

    # def forward_backbone(self, x):
    #     x = self.patch_embed(x)

    #     for layer in self.layers:
    #         x = layer(x)
    #     return x

    #Dencoder and Skip connection  x_downsample：一个包含编码器各层下采样特征的列表，这些特征将在解码器中用于与上采样的特征进行融合。
    def forward_up_features(self, x, x_downsample):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x) #当是第一个解码器层时，直接将输入x传递给该层layer_up进行处理。
            else:
                x = torch.cat([x,x_downsample[3-inx]],-1)   #对x和x_downsample[3-inx]进行拼接，然后将结果传递给layer_up进行处理。skip connection
                x = self.concat_back_dim[inx](x)    #调整拼接后特征的维度，以匹配当前解码器层的要求。
                x = layer_up(x) #将调整后的特征传递给layer_up进行处理。

        x = self.norm_up(x)  # B H W C  对最终的特征图x进行归一化处理
  
        return x    #返回最终的特征图x
    def up_x4(self, x):     #最终上采样
        if self.final_upsample=="expand_first":
            B,H,W,C = x.shape
            x = self.up(x)  #self.up模块对特征图x进行上采样
            x = x.view(B, 4*H, 4*W, -1)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W    将特征图的维度重新排列
            rep=x
            pred = self.output(rep)  #对特征图进行卷积操作 将上采样的特征图转换为最终的输出格式
            
        return pred,rep        #返回最终的输出
    #TODO: 修改这一部分，需要一个特征图的输出结果
    def forward(self, x):
        # print("x.shape",x.shape)  #torch.Size([8, 3, 224, 224])
        x,x_downsample = self.forward_features(x)   #对输入特征x进行编码器和瓶颈层处理
        x = self.forward_up_features(x,x_downsample)    #对编码器和瓶颈层处理后的特征x和x_downsample进行解码器和skip connection处理
        # x = self.up_x4(x)   #对解码器处理后的特征x进行最终上采样
        pred,rep = self.up_x4(x)
        outs = {"pred":pred, "rep":rep}
        return outs     #返回最终的输出

    #无需修改
    def flops(self, shape=(3, 224, 224)):   #计算模型的FLOPs shape是输入特征的形状
        # shape = self.__input_shape__[1:]
        supported_ops={
            "aten::silu": None, # as relu is in _IGNORED_OPS
            "aten::neg": None, # as relu is in _IGNORED_OPS
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            "prim::PythonOp.SelectiveScanFn": selective_scan_flop_jit, # latter
        }

        model = copy.deepcopy(self) #深拷贝模型
        model.cuda().eval() #将模型移动到cuda设备并设置为评估模式

        input = torch.randn((1, *shape), device=next(model.parameters()).device)    #生成一个随机初始化的输入张量，其形状由shape参数指定，并且与模型的参数位于同一个设备上。
        params = parameter_count(model)[""]  #计算模型的参数量
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

        del model, input    #删除模型和输入张量
        return sum(Gflops.values()) * 1e9   #返回模型的FLOPs
        return f"params {params} GFLOPs {sum(Gflops.values())}"   #返回模型的参数量和FLOPs


# APIs with VMamba2Dp ================= 未用到
def check_vssm_equals_vmambadp():
    from bak.vmamba_bak1 import VMamba2Dp

    # test 1 True =================================
    torch.manual_seed(time.time()); torch.cuda.manual_seed(time.time())
    oldvss = VMamba2Dp(depths=[2,2,6,2]).half().cuda()
    newvss = VSSM(depths=[2,2,6,2]).half().cuda()
    newvss.load_state_dict(oldvss.state_dict())
    input = torch.randn((12, 3, 224, 224)).half().cuda()
    torch.cuda.manual_seed(0)
    with torch.cuda.amp.autocast():
        y1 = oldvss.forward_backbone(input)
    torch.cuda.manual_seed(0)
    with torch.cuda.amp.autocast():
        y2 = newvss.forward_backbone(input)
    print((y1 -y2).abs().sum()) # tensor(0., device='cuda:0', grad_fn=<SumBackward0>)
    
    # test 2 True ==========================================
    torch.manual_seed(0); torch.cuda.manual_seed(0)
    oldvss = VMamba2Dp(depths=[2,2,6,2]).cuda()
    torch.manual_seed(0); torch.cuda.manual_seed(0)
    newvss = VSSM(depths=[2,2,6,2]).cuda()

    miss_align = 0
    for k, v in oldvss.state_dict().items(): 
        same = (oldvss.state_dict()[k] == newvss.state_dict()[k]).all()
        if not same:
            print(k, same)
            miss_align += 1
    print("init miss align", miss_align) # init miss align 0

import lib.mamba
class SelectiveScan(torch.autograd.Function):
    
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32) #这是自定义的前向传播函数
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
        assert nrows in [1, 2, 3, 4], f"{nrows}" # 8+ is too slow to compile
        assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
        ctx.delta_softplus = delta_softplus
        ctx.nrows = nrows
        # all in float
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if B.dim() == 3:
            B = B.unsqueeze(dim=1)
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = C.unsqueeze(dim=1)
            ctx.squeeze_C = True
        
        if SSMODE == "mamba_ssm":
            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        else:
            out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)
        
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)  # save the intermediate results
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd #这是自定义的反向传播函数
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        
        if SSMODE == "mamba_ssm":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
                False  # option to recompute out_z, not used here
            )
        else:
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
                # u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, ctx.nrows,
            )
        
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None)
"""
Local Mamba
"""
from lib.mamba.multi_mamba import MultiScan
#实现一种特殊的扫描操作
class MultiScanVSSM(MultiScan):

    ALL_CHOICES = MultiScan.ALL_CHOICES #一共有八种扫描方向

    def __init__(self, dim, choices=None):
        super().__init__(dim, choices=choices, token_size=None)
        self.attn = BiAttn(dim)

    def merge(self, xs):
        # xs: [B, K, D, L]
        # return: [B, D, L]

        # remove the padded tokens
        xs = [xs[:, i, :, :l] for i, l in enumerate(self.scan_lengths)]
        xs = super().multi_reverse(xs)
        xs = [self.attn(x.transpose(-2, -1)) for x in xs]
        x = super().forward(xs)
        return x

    
    def multi_scan(self, x):
        # x: [B, C, H, W]
        # return: [B, K, C, H * W]
        B, C, H, W = x.shape
        self.token_size = (H, W)

        xs = super().multi_scan(x)  # [[B, C, H, W], ...]   

        self.scan_lengths = [x.shape[2] for x in xs]
        max_length = max(self.scan_lengths)

        # pad the tokens into the same length as VMamba compute all directions together 将令牌填充到与VMamba一起计算所有方向的相同长度
        new_xs = []
        for x in xs:
            if x.shape[2] < max_length:
                x = F.pad(x, (0, max_length - x.shape[2]))
            new_xs.append(x)
        return torch.stack(new_xs, 1)   #它将填充后的张量堆叠起来，形成输出张量

    def __repr__(self):
        scans = ', '.join(self.choices) #将choices中的元素用逗号连接起来
        return super().__repr__().replace('MultiScanVSSM', f'MultiScanVSSM[{scans}]')   #将MultiScanVSSM替换为MultiScanVSSM[scans]

#实现一种双向注意力机制 这里只实现了通道注意力(空间注意力部分被注释掉了)
class BiAttn(nn.Module):
    def __init__(self, in_channels, act_ratio=0.125, act_fn=nn.GELU, gate_fn=nn.Sigmoid):
        super().__init__()
        reduce_channels = int(in_channels * act_ratio)
        self.norm = nn.LayerNorm(in_channels)
        self.global_reduce = nn.Linear(in_channels, reduce_channels)
        # self.local_reduce = nn.Linear(in_channels, reduce_channels)
        self.act_fn = act_fn()
        self.channel_select = nn.Linear(reduce_channels, in_channels)
        # self.spatial_select = nn.Linear(reduce_channels * 2, 1)
        self.gate_fn = gate_fn()

    def forward(self, x):
        ori_x = x
        x = self.norm(x)
        x_global = x.mean(1, keepdim=True)
        x_global = self.act_fn(self.global_reduce(x_global))
        # x_local = self.act_fn(self.local_reduce(x))

        c_attn = self.channel_select(x_global)
        c_attn = self.gate_fn(c_attn)  # [B, 1, C]
        # s_attn = self.spatial_select(torch.cat([x_local, x_global.expand(-1, x.shape[1], -1)], dim=-1))
        # s_attn = self.gate_fn(s_attn)  # [B, N, 1]

        attn = c_attn #* s_attn  # [B, N, C]
        out = ori_x * attn
        return out  

#输入数据 x 进行一系列的处理，包括投影、分割、扫描和归一化，最终输出处理后的数据 y
def multi_selective_scan(
    x: torch.Tensor=None,                  #[24, 56, 192, 56]
    x_proj_weight: torch.Tensor=None,   #[8, 38, 192]
    x_proj_bias: torch.Tensor=None,     #none
    dt_projs_weight: torch.Tensor=None,     #[8, 192, 6]
    dt_projs_bias: torch.Tensor=None,
    A_logs: torch.Tensor=None,
    Ds: torch.Tensor=None,
    out_norm: torch.nn.Module=None,
    nrows = -1,
    delta_softplus = True,
    to_dtype=True,
    multi_scan=None,        #self.multi_scan = MultiScanVSSM(d_expand, choices=directions)
):
    B, D, H, W = x.shape #[24, 56, 192, 56]
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape  #[8,192,6]
    L = H * W

    if nrows < 1:
        if D % 4 == 0:
            nrows = 4
        elif D % 3 == 0:
            nrows = 3
        elif D % 2 == 0:
            nrows = 2
        else:
            nrows = 1

    xs = multi_scan.multi_scan(x)   #使用 multi_scan 实例对 x 执行多扫描操作，得到 xs 的维度是 [B, K, D, max(H, W)]

    L = xs.shape[-1]   #L是xs的最后一个维度的大小为max(H, W)
    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight) # l fixed

    #使用 x_proj_weight 对 xs 进行投影，然后分割成 dts, Bs, Cs
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    
    xs = xs.view(B, -1, L).to(torch.float)
    dts = dts.contiguous().view(B, -1, L).to(torch.float)
    As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
    Bs = Bs.contiguous().to(torch.float)
    Cs = Cs.contiguous().to(torch.float)
    Ds = Ds.to(torch.float) # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)
    
    #定义一个内部函数 selective_scan，它使用 SelectiveScan.apply 方法应用选择性扫描操作
    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)
    
    #使用 selective_scan 函数对 xs, dts, As, Bs, Cs, Ds, delta_bias 进行选择性扫描操作
    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
    ).view(B, K, -1, L) 
    
    y = multi_scan.merge(ys)    #使用 multi_scan 实例对 ys 进行合并操作，得到 y y: [B, D, L]
    
    y = out_norm(y).view(B, H, W, -1)   #对 y 进行归一化操作

    return (y.to(x.dtype) if to_dtype else y)   #返回处理后的数据 y

@register_model
def local_vssm_tiny_search(*args, drop_path_rate=0.1, **kwargs):
    return VSSM(dims=[32, 64, 128, 256], depths=[2, 2, 9, 2], d_state=16, drop_path_rate=drop_path_rate)
    
@register_model
def local_vssm_tiny(*args, drop_path_rate=0.2, **kwargs):
    #是一个list
    directions = [
        ['h', 'h_flip', 'w7', 'w7_flip'],
        ['h_flip', 'v_flip', 'w2', 'w2_flip'],
        ['h_flip', 'v_flip', 'w2_flip', 'w7'],
        ['h_flip', 'v', 'v_flip', 'w2'],
        ['h', 'h_flip', 'v_flip', 'w2_flip'],
        ['h_flip', 'v_flip', 'w2', 'w2_flip'],
        ['h', 'w2_flip', 'w7', 'w7_flip'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'v_flip', 'w7', 'w7_flip'],
        ['h_flip', 'v', 'w2', 'w7_flip'],
        ['v', 'v_flip', 'w2', 'w7_flip'],
        ['h', 'h_flip', 'v_flip', 'w2_flip'],
        ['v_flip', 'w2_flip', 'w7', 'w7_flip'],
        ['h_flip', 'v_flip', 'w2_flip', 'w7_flip'],
        ['h_flip', 'v', 'w7', 'w7_flip'],
    ]
    return VSSM(dims=[96, 192, 384, 768], depths=[2, 2, 9, 2], in_chans=3,d_state=16, drop_path_rate=drop_path_rate, directions=directions) #解决使用报错Given groups=1, weight of size [96, 1, 4, 4], expected input[8, 3, 224, 224] to have 1 channels, but got 3 channels instead  增加in_chans=3

@register_model
def local_vssm_small_search(*args, drop_path_rate=0.1, **kwargs):
    return VSSM(dims=[32, 64, 128, 256], depths=[2, 2, 27, 2], d_state=16, drop_path_rate=drop_path_rate)


@register_model
def local_vssm_small(*args, drop_path_rate=0.2, **kwargs):
    directions = [
        ['h', 'v', 'v_flip', 'w7_flip'],
        ['h', 'h_flip', 'v', 'w7'],
        ['h_flip', 'v', 'v_flip', 'w7'],
        ['h_flip', 'v', 'w2_flip', 'w7'],
        ['h_flip', 'v', 'w2_flip', 'w7'],
        ['h_flip', 'v_flip', 'w2', 'w7_flip'],
        ['h_flip', 'v', 'v_flip', 'w7'],
        ['h', 'v', 'v_flip', 'w7'],
        ['h', 'v', 'v_flip', 'w7'],
        ['h', 'v', 'v_flip', 'w7_flip'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'v', 'v_flip', 'w2'],
        ['v', 'v_flip', 'w2_flip', 'w7'],
        ['h', 'h_flip', 'v', 'w2'],
        ['h_flip', 'v', 'v_flip', 'w7_flip'],
        ['h', 'h_flip', 'v', 'v_flip'],
        ['h', 'v', 'v_flip', 'w7_flip'],
        ['h', 'v', 'v_flip', 'w7_flip'],
        ['h', 'h_flip', 'v_flip', 'w7'],
        ['h_flip', 'v_flip', 'w2_flip', 'w7'],
        ['h_flip', 'v', 'v_flip', 'w7_flip'],
        ['v', 'v_flip', 'w7', 'w7_flip'],
        ['h', 'v', 'v_flip', 'w7_flip'],
        ['h_flip', 'v', 'v_flip', 'w2'],
        ['h', 'v', 'v_flip', 'w2_flip'],
        ['h', 'h_flip', 'v', 'w7'],
        ['h', 'h_flip', 'w7', 'w7_flip'],
        ['h', 'v_flip', 'w2', 'w2_flip'],
        ['h', 'v_flip', 'w2', 'w7'],
        ['h', 'v', 'v_flip', 'w7_flip'],
        ['h_flip', 'v', 'w2_flip', 'w7'],
        ['h_flip', 'v_flip', 'w7', 'w7_flip'],
        ['h', 'v', 'w7', 'w7_flip']
    ]
    return VSSM(dims=[96, 192, 384, 768], depths=[2, 2, 27, 2], d_state=16, drop_path_rate=drop_path_rate, directions=directions,in_chans=3)



if __name__ == "__main__":
    # check_vssm_equals_vmambadp()
    model = VSSM().to('cuda')
    int = torch.randn(16,1,224,224).cuda()
    out = model(int)
    print(out.shape)
