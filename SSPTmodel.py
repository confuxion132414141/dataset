import math
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mmdet.utils import get_root_logger
from mmcv.runner import load_checkpoint
from mmcv.ops import DeformConv2dPack

# ---------------------- 动态卷积相关模块（保持不变） ----------------------
class DynamicConvAttention(nn.Module):
    def __init__(self, in_planes, ratio, K, temprature=30, init_weight=True):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.temprature = temprature
        assert in_planes > ratio or ratio == 1, f"in_planes={in_planes} 必须大于 ratio={ratio}（除非 ratio=1）"
        hidden_planes = max(in_planes // ratio, 1)
        self.net = nn.Sequential(
            nn.Conv2d(in_planes, hidden_planes, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_planes, K, kernel_size=1, bias=False)
        )

        if init_weight:
            self._initialize_weights()

    def update_temprature(self):
        if self.temprature > 1:
            self.temprature -= 1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        att = self.avgpool(x)  # bs,dim,1,1
        att = self.net(att).view(x.shape[0], -1)  # bs,K
        return F.softmax(att / self.temprature, -1)


class DynamicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, dilation=1, grounps=1, bias=True, K=4,
                 temprature=30, ratio=4, init_weight=True):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        # 拆分 kernel_size 为整数（高、宽）
        self.kh, self.kw = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        # 保存 stride（下采样步长）和 padding（与 kernel_size 匹配）
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = grounps
        self.bias = bias
        self.K = K
        self.init_weight = init_weight
        self.attention = DynamicConvAttention(in_planes=in_planes, ratio=ratio, K=K, temprature=temprature,
                                              init_weight=init_weight)

        # 权重初始化：K × out_planes × (in_planes//groups) × kh × kw
        self.weight = nn.Parameter(torch.randn(
            K, out_planes, in_planes // grounps, self.kh, self.kw
        ), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.randn(K, out_planes), requires_grad=True)
        else:
            self.bias = None

        if self.init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def forward(self, x):
        bs, in_planels, h, w = x.shape  # 输入尺寸：[bs, in_planes, h, w]
        softmax_att = self.attention(x)  # [bs, K]：每个样本的K个卷积核权重

        # 1. 调整输入形状：适配 group conv（将bs维度合并到channels）
        x = x.view(1, -1, h, w)  # [1, bs×in_planes, h, w]

        # 2. 聚合K个卷积核：按注意力权重加权求和
        weight = self.weight.view(self.K, -1)  # [K, out_planes×(in_planes//groups)×kh×kw]
        aggregate_weight = torch.mm(softmax_att, weight)  # [bs, out_planes×(in_planes//groups)×kh×kw]
        # 调整聚合权重形状：适配 F.conv2d
        aggregate_weight = aggregate_weight.view(
            bs * self.out_planes, self.in_planes // self.groups, self.kh, self.kw
        )  # [bs×out_planes, in_planes//groups, kh, kw]

        # 3. 卷积操作：关键是传递 stride、padding、dilation 参数，确保尺寸正确
        if self.bias is not None:
            # 聚合偏置：[K, out_planes] → [bs, out_planes]
            aggregate_bias = torch.mm(softmax_att, self.bias)  # [bs, out_planes]
            aggregate_bias = aggregate_bias.view(-1)  # [bs×out_planes]
            # 卷积：传递所有参数，确保尺寸按 stride 缩小
            output = F.conv2d(
                x, weight=aggregate_weight, bias=aggregate_bias,
                stride=self.stride, padding=self.padding, dilation=self.dilation,
                groups=self.groups * bs  # group数=bs×groups，实现逐样本独立卷积
            )
        else:
            output = F.conv2d(
                x, weight=aggregate_weight, bias=None,
                stride=self.stride, padding=self.padding, dilation=self.dilation,
                groups=self.groups * bs
            )

        # 4. 计算输出尺寸（验证是否符合预期）
        out_h = (h + 2 * self.padding[0] - self.dilation[0] * (self.kh - 1) - 1) // self.stride[0] + 1
        out_w = (w + 2 * self.padding[1] - self.dilation[1] * (self.kw - 1) - 1) // self.stride[1] + 1

        # 5. 调整输出形状：恢复 bs 维度
        output = output.view(bs, self.out_planes, out_h, out_w)  # [bs, out_planes, out_h, out_w]
        return output

# ---------------------- Transformer注意力模块（保持不变） ----------------------
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
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

    def forward(self, x, Hx, Wx, z=None, Hz=None, Wz=None):
        if z is not None:
            B, Nx, C = x.shape
            B, Nz, C = z.shape

            qx = self.q(x).reshape(B, Nx, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            qz = self.q(z).reshape(B, Nz, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

            if not self.linear:
                if self.sr_ratio > 1:
                    x_ = x.permute(0, 2, 1).reshape(B, C, Hx, Wx)
                    x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                    x_ = self.norm(x_)
                    kvx = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

                    z_ = z.permute(0, 2, 1).reshape(B, C, Hz, Wz)
                    z_ = self.sr(z_).reshape(B, C, -1).permute(0, 2, 1)
                    z_ = self.norm(z_)
                    kvz = self.kv(z_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                else:
                    kvx = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                    kvz = self.kv(z).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                x_ = x.permute(0, 2, 1).reshape(B, C, Hx, Wx)
                x_= self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                x_ = self.act(x_)
                kvx = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

                z_ = z.permute(0, 2, 1).reshape(B, C, Hz, Wz)
                z_ = self.sr(self.pool(z_)).reshape(B, C, -1).permute(0, 2, 1)
                z_ = self.norm(z_)
                z_ = self.act(z_)
                kvz = self.kv(z_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            kx, vx = kvx[0], kvx[1]
            kz, vz = kvz[0], kvz[1]

            attnx = (qx @ kz.transpose(-2, -1)) * self.scale
            attnx = attnx.softmax(dim=-1)
            attnx = self.attn_drop(attnx)

            attnz = (qz @ kx.transpose(-2, -1)) * self.scale
            attnz = attnz.softmax(dim=-1)
            attnz = self.attn_drop(attnz)

            x = (attnx @ vz).transpose(1, 2).reshape(B, Nx, C)
            x = self.proj(x)
            x = self.proj_drop(x)

            z = (attnz @ vx).transpose(1, 2).reshape(B, Nz, C)
            z = self.proj(z)
            z = self.proj_drop(z)
            return x, z
        else:
            B, N, C = x.shape
            q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            if not self.linear:
                if self.sr_ratio > 1:
                    x_ = x.permute(0, 2, 1).reshape(B, C, Hx, Wx)
                    x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                    x_ = self.norm(x_)
                    kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                else:
                    kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                x_ = x.permute(0, 2, 1).reshape(B, C, Hx, Wx)
                x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                x_ = self.act(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

# ---------------------- 其他原有模块（Mlp、Block、DWConv）保持不变 ----------------------
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
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

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

        self.apply(self._init_weights)

    def _init_weights(self, m):
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

    def forward(self, x, Hx, Wx, z=None, Hz=None, Wz=None, cross_flag = False):
        if cross_flag:
            x_reconst_from_z, z_reconst_from_x = self.attn(self.norm1(x), Hx, Wx, self.norm1(z), Hz, Wz)

            x_fuse = x + self.drop_path(x_reconst_from_z)
            z_fuse = z + self.drop_path(z_reconst_from_x)

            x = x_fuse + self.drop_path(self.mlp(self.norm2(x_fuse), Hx, Wx))
            z = z_fuse + self.drop_path(self.mlp(self.norm2(z_fuse), Hz, Wz))

        else:
            if z is not None:
                x = x + self.drop_path(self.attn(self.norm1(x), Hx, Wx))
                x = x + self.drop_path(self.mlp(self.norm2(x), Hx, Wx))

                z = z + self.drop_path(self.attn(self.norm1(z), Hz, Wz))
                z = z + self.drop_path(self.mlp(self.norm2(z), Hz, Wz))
            else:
                x = x + self.drop_path(self.attn(self.norm1(x), Hx, Wx))
                x = x + self.drop_path(self.mlp(self.norm2(x), Hx, Wx))

        return x, z

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

# ---------------------- 关键修改：OverlapPatchEmbed 按阶段选择卷积类型 ----------------------
class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding：第一阶段用普通卷积，第二/三阶段用动态卷积
    """
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768,
                 stage_idx=0,  # 新增：标识当前是第几个阶段（0=第一阶段，1=第二，2=第三）
                 K=4, ratio=4, temprature=30):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.stage_idx = stage_idx  # 保存阶段索引

        # 核心逻辑：按阶段选择卷积类型
        if stage_idx == 0:  # 第一阶段：用原始普通卷积（nn.Conv2d）
            self.proj = nn.Conv2d(
                in_chans, embed_dim,
                kernel_size=patch_size,
                stride=stride,
                padding=(patch_size[0] // 2, patch_size[1] // 2),
                bias=False  # 与原有配置一致
            )
        else:  # 第二/三阶段：用动态卷积（DynamicConv）
            self.proj = DynamicConv(
                in_planes=in_chans,
                out_planes=embed_dim,
                kernel_size=patch_size,
                stride=stride,
                padding=(patch_size[0] // 2, patch_size[1] // 2),
                bias=False,
                K=K,
                ratio=ratio,
                temprature=temprature,
                init_weight=True
            )
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        # 普通卷积的初始化（动态卷积已自带初始化）
        elif isinstance(m, nn.Conv2d) and self.stage_idx == 0:
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

# ---------------------- SSPT 类：传递阶段索引给 OverlapPatchEmbed ----------------------
class SSPT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256],
                 num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[3, 4, 6],
                 sr_ratios=[8, 4, 2], num_stages=3, linear=False, pretrained=None, cross_dict= None, cross_num=5, 
                 down_sample=[4, 2, 2],
                 dynconv_K=4, dynconv_ratios=[4, 4],  # 仅第二/三阶段用（索引1、2）
                 dynconv_temprature=30):
        super().__init__()
        self.depths = depths
        self.num_stages = num_stages
        self.linear = linear
        self.embed_dim = embed_dims[num_stages-1]
        self.dynconv_K = dynconv_K
        self.dynconv_ratios = dynconv_ratios  # 长度=2：对应第二（index0）、三（index1）阶段
        self.dynconv_temprature = dynconv_temprature

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(num_stages):
            # 传递阶段索引 i（0=第一阶段，1=第二，2=第三）
            patch_embed = OverlapPatchEmbed(
                img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                patch_size=7 if i == 0 else 3,
                stride=down_sample[i],
                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i],
                stage_idx=i,  # 关键：传递阶段索引
                K=dynconv_K if i >=1 else 0,  # 第一阶段不用K
                ratio=self.dynconv_ratios[i-1] if i >=1 else 0,  # 第二阶段用ratios[0]，第三用ratios[1]
                temprature=dynconv_temprature if i >=1 else 0  # 第一阶段不用temprature
            )

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        self.cross_dict = cross_dict
        for i in range(len(self.cross_dict)):
            if len(self.cross_dict[i]) > 0:
               self.start_stage = i
               self.start_blk = self.cross_dict[i][0] -1
               break

        self.apply(self._init_weights)
        self.init_weights(pretrained)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) and not hasattr(m, 'stage_idx'):  # 排除OverlapPatchEmbed中的普通卷积（已单独初始化）
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        return None

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x, _ = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs

    def forward_single(self, x):
        x = self.forward_features(x)
        return x

    def forward_cross(self, x, z ):
        B = x.shape[0]
        seq_out = {"x":[],"z":[]}
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, Hx, Wx = patch_embed(x)
            z, Hz, Wz = patch_embed(z)

            cnt=0
            for blk in block:
                cnt = cnt + 1
                cross_flag = cnt in self.cross_dict[i]
                x, z = blk(x, Hx, Wx, z, Hz, Wz, cross_flag=cross_flag)

            x = x.reshape(B, Hx, Wx, -1).permute(0, 3, 1, 2).contiguous()
            z = z.reshape(B, Hz, Wz, -1).permute(0, 3, 1, 2).contiguous()

            seq_out["x"].append(x)
            seq_out["z"].append(z)

        return seq_out

    def forward(self, x, z = None):
        if z is not None:
            x_fused = self.forward_cross(z, x)
        else:
            x_fused = self.forward_single(x)
        return x_fused

# ---------------------- SSPT_base 类：配置动态卷积参数（仅第二/三阶段） ----------------------
class SSPT_base(SSPT):
    def __init__(self, **kwargs):
        cross_num = kwargs.get('cross_num', 5)
        cross_dict = kwargs.get('cross_dict', [
            [3],
            [3,4],
            [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
        ])
        super(SSPT_base, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320], num_heads=[1, 2, 5], mlp_ratios=[8, 8, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 10],
            sr_ratios=[8, 4, 2],
            drop_rate=0.0, drop_path_rate=0.1,
            down_sample=[4, 2, 2],
            num_stages=3,
            cross_dict=cross_dict,
            cross_num=cross_num,
            dynconv_K=4,
            dynconv_ratios=[2, 4],  # 第二阶段ratio=4，第三阶段ratio=4（输入通道64、128均>4）
            dynconv_temprature=30
        )

    def load_param(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def load_param_self_backbone(self, pretrained=None):
        if isinstance(pretrained, str):
            state_dict = torch.load(pretrained, map_location='cpu')  # 加载预训练权重
            current_state_dict = self.state_dict()  # 当前模型权重

            # 遍历预训练权重，处理形状匹配
            new_state_dict = {}
            for key, pretrain_param in state_dict.items():
                # 跳过不存在于当前模型的权重（如分类头）
                if key not in current_state_dict:
                    print(f"跳过不存在的权重：{key}")
                    continue

                current_param = current_state_dict[key]
                # 处理动态卷积的权重（patch_embed2.proj.weight 和 patch_embed3.proj.weight）
                if 'patch_embed2.proj.weight' in key or 'patch_embed3.proj.weight' in key:
                    # 预训练权重形状：[out, in, h, w] → 当前权重形状：[K, out, in, h, w]
                    # 将预训练权重复制 K 次，初始化动态卷积的 K 个核
                    pretrain_shape = pretrain_param.shape  # [out, in, h, w]
                    current_shape = current_param.shape    # [K, out, in, h, w]
                    # 验证核心维度是否匹配（out/in/h/w 必须一致）
                    if pretrain_shape == current_shape[1:]:
                        # 复制 K 次：[out,in,h,w] → [K,out,in,h,w]
                        adapted_param = pretrain_param.unsqueeze(0).repeat(self.dynconv_K, 1, 1, 1, 1)
                        new_state_dict[key] = adapted_param
                        print(f"适配动态卷积权重：{key} → 形状 {pretrain_shape} → {current_shape}")
                    else:
                        # 核心维度不匹配，使用当前模型的初始化权重（不加载预训练）
                        new_state_dict[key] = current_param
                        print(f"动态卷积权重维度不匹配，跳过加载：{key}（预训练 {pretrain_shape} vs 当前 {current_shape}）")
                else:
                    # 其他权重：形状一致直接加载，不一致跳过
                    if pretrain_param.shape == current_param.shape:
                        new_state_dict[key] = pretrain_param
                    else:
                        new_state_dict[key] = current_param
                        print(f"权重形状不匹配，跳过加载：{key}（预训练 {pretrain_param.shape} vs 当前 {current_param.shape}）")

            # 加载适配后的权重（strict=False 确保兼容）
            self.load_state_dict(new_state_dict, strict=False)
            print("预训练权重加载完成（已适配动态卷积）！")

# ---------------------- 构建函数（保持不变） ----------------------
def build_sspt(cross_dict=None):
    if cross_dict is None:
        cross_dict = [
            [3],
            [3,4],
            [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
        ]
    model = SSPT_base(cross_dict=cross_dict, cross_num=5)
    return model

# ---------------------- 测试代码 ----------------------
if __name__ == '__main__':
    img_size = 384
    x = torch.rand(1, 3, 256, 256)
    z = torch.rand(1, 3, 96, 96)
    model = build_sspt()
    
    # 验证各阶段卷积类型
    print("第一阶段卷积类型：", type(model.patch_embed1.proj))  # 输出：<class 'torch.nn.modules.conv.Conv2d'>
    print("第二阶段卷积类型：", type(model.patch_embed2.proj))  # 输出：<class '__main__.DynamicConv'>
    print("第三阶段卷积类型：", type(model.patch_embed3.proj))  # 输出：<class '__main__.DynamicConv'>
    
    # 前向传播测试
    output = model(z, x)
    print("输出结构：", {k: [v.shape for v in lst] for k, lst in output.items()})