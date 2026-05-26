'''by lyuwenyu

改进说明 (v2):
1. WindowAttentionBlock 增强:
   - 新增可学习的 per-head 相对位置偏置 (relative position bias)
   - 新增轻量 MLP (expansion=2) 增强特征变换能力
   - 可学习 gamma 参数 (初始化为 0), 让模型渐进式利用窗口注意力
2. 窗口注意力作用于 FPN+PAN 融合之后 (而非 input_proj 之后):
   - P3/P4 先通过 FPN+PAN 获取多尺度语义上下文
   - 再在富含语义信息的特征上做窗口局部自注意力, 效果显著优于在低级特征上做
3. 多尺度窗口注意力: 同时作用于 P3 和 P4 (use_window_attn_idx 可配置)
   - P3 (stride=8): 最高分辨率, 覆盖极小目标
   - P4 (stride=16): 中等分辨率, 覆盖中小目标
   - P5 (stride=32): 已有全局 Transformer encoder 处理, 不再重复
4. FPN 上采样优化: nearest -> bilinear + 抗混叠 depthwise conv
'''

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_activation

from src.core import register


__all__ = ['HybridEncoder']



class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=(kernel_size-1)//2 if padding is None else padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


# ============================================================================
# 窗口自注意力块 (增强版)
# 相比 v1 的改进:
#   1. 新增 per-head 可学习相对位置偏置, 让注意力感知空间位置关系
#   2. 新增轻量 MLP 层, 增强窗口内的特征变换能力
#   3. 可学习 gamma (初始化为 0), 实现从 identity 到窗口注意力的平滑过渡
#
# 维度变换逻辑:
#   Input:  [B, C, H, W]
#   Step 1: Pad H, W 至 window_size 的整数倍 → [B, C, Hp, Wp]
#   Step 2: 划分为窗口 → [B, C, num_win_h, ws, num_win_w, ws]
#   Step 3: permute + reshape → [B * num_windows, ws*ws, C]
#   Step 4: W-MSA (含相对位置偏置) + MLP, 各带残差 + LayerNorm
#   Step 5: reshape + permute 逆操作 → [B, C, Hp, Wp]
#   Step 6: 裁掉 padding → [B, C, H, W]
# ============================================================================
class WindowAttentionBlock(nn.Module):
    def __init__(self, dim, window_size=7, nhead=8, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.nhead = nhead
        self.head_dim = dim // nhead

        assert dim % nhead == 0, f"dim ({dim}) must be divisible by nhead ({nhead})"

        # ---- QKV 投影权重 (手动管理以支持相对位置偏置注入) ----
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)

        # ---- 轻量 MLP (expansion=2, 平衡效果与参数量) ----
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # ---- Per-head 可学习相对位置偏置表 ----
        # 相对位置范围: [-(ws-1), (ws-1)], 共 (2*ws-1) 个取值
        # 形状: [nhead, 2*ws-1, 2*ws-1]
        self.rel_pos_bias_table = nn.Parameter(
            torch.zeros(nhead, 2 * window_size - 1, 2 * window_size - 1)
        )
        # 预计算相对位置索引: 窗口内任意两点 (i,j) 映射到偏置表的下标
        self._init_rel_pos_index(window_size)

        # ---- 可学习残差缩放因子, 初始化为 0 实现平滑启动 ----
        self.gamma = nn.Parameter(torch.zeros(1))

    def _init_rel_pos_index(self, ws: int):
        """预计算窗口内所有 token 对的相对位置索引, 存为 buffer 避免重复计算."""
        coords_h = torch.arange(ws)
        coords_w = torch.arange(ws)
        # meshgrid: [ws, ws] grid, 输出每个格点的 (h, w) 坐标
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # [2, ws, ws]
        coords_flat = coords.reshape(2, ws * ws)                                  # [2, N]

        # rel_coords[d, i, j] = coords_flat[d, i] - coords_flat[d, j]
        rel_coords = coords_flat[:, :, None] - coords_flat[:, None, :]            # [2, N, N]
        # 将范围从 [-(ws-1), ws-1] 平移到 [0, 2*ws-2]
        rel_coords = rel_coords + (ws - 1)
        # 分别存储行/列索引, 形状均为 [N, N]
        self.register_buffer('rel_pos_h', rel_coords[0].contiguous())
        self.register_buffer('rel_pos_w', rel_coords[1].contiguous())

    def _get_rel_pos_bias(self, num_windows: int) -> torch.Tensor:
        """从偏置表中按索引取出 per-head 相对位置偏置, 并扩展到所有窗口.

        Args:
            num_windows: B * num_win_h * num_win_w

        Returns:
            [num_windows * nhead, N, N] 的 additive mask, 可直接传入 MHA
        """
        # [nhead, N, N]: 每个 head 取出自己的2D偏置矩阵
        bias = self.rel_pos_bias_table[:, self.rel_pos_h, self.rel_pos_w]
        # 扩展到所有窗口: [num_windows * nhead, N, N]
        bias = bias.unsqueeze(0).expand(num_windows, -1, -1, -1)
        bias = bias.reshape(num_windows * self.nhead, self.rel_pos_h.shape[0], self.rel_pos_h.shape[1])
        return bias

    def _window_attn(self, x: torch.Tensor, num_windows: int) -> torch.Tensor:
        """窗口内多头自注意力 (含相对位置偏置).

        Args:
            x: [B*num_windows, N, C]  where N = ws*ws
            num_windows: 窗口总数 (B * num_win_h * num_win_w)

        Returns:
            [B*num_windows, N, C]
        """
        B_N, N, C = x.shape

        # 线性投影 → [B_N, N, nhead, head_dim]
        q = self.q_proj(x).reshape(B_N, N, self.nhead, self.head_dim)
        k = self.k_proj(x).reshape(B_N, N, self.nhead, self.head_dim)
        v = self.v_proj(x).reshape(B_N, N, self.nhead, self.head_dim)

        # 转置为 [B_N, nhead, N, head_dim]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # 缩放点积注意力
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale            # [B_N, nhead, N, N]

        # ---- 注入相对位置偏置 ----
        rel_bias = self._get_rel_pos_bias(num_windows)      # [B_N * nhead, N, N]
        rel_bias = rel_bias.reshape(B_N, self.nhead, N, N)
        attn = attn + rel_bias

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # 加权求和
        out = attn @ v                                       # [B_N, nhead, N, head_dim]
        out = out.permute(0, 2, 1, 3).reshape(B_N, N, C)    # [B_N, N, C]
        out = self.out_proj(out)

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ws = self.window_size

        # 保存原始 4D 特征用于最终的残差连接
        identity_4d = x  # [B, C, H, W]

        # ---- Pad 到 window_size 的整数倍 ----
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        Hp, Wp = H + pad_h, W + pad_w
        num_win_h, num_win_w = Hp // ws, Wp // ws
        num_windows = B * num_win_h * num_win_w

        # ---- [B, C, Hp, Wp] → [B * num_windows, ws*ws, C] ----
        x = x.reshape(B, C, num_win_h, ws, num_win_w, ws)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(num_windows, ws * ws, C)

        # ---- W-MSA with relative position bias + residual ----
        x = x + self._window_attn(self.norm1(x), num_windows)

        # ---- MLP + residual ----
        x = x + self.mlp(self.norm2(x))

        # ---- 还原 → [B, C, Hp, Wp] ----
        x = x.reshape(B, num_win_h, num_win_w, ws, ws, C)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(B, C, Hp, Wp)

        # ---- 裁掉 padding ----
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]

        # gamma(初始为0) 控制窗口注意力贡献, 实现从 identity 到增强特征的平滑过渡
        return identity_4d + self.gamma * x


# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


@register
class HybridEncoder(nn.Module):
    __share__ = ['eval_spatial_size']

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward = 1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None,
                 # ---- 新增参数 ----
                 window_size=7,               # 局部窗口大小
                 use_window_attn_idx=[0, 1],  # 窗口注意力作用的特征层级 (默认 P3=0, P4=1)
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.use_window_attn_idx = use_window_attn_idx

        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides

        # ---- channel projection (1x1 Conv + BN) ----
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim)
                )
            )

        # ---- 全局 Transformer encoder (通常仅作用于 P5) ----
        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act)

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers)
            for _ in range(len(use_encoder_idx))
        ])

        # ---- Top-down FPN ----
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        # 抗混叠深度可分离卷积: 消除 bilinear 上采样的混叠伪影
        self.fpn_aa_convs = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )
            self.fpn_aa_convs.append(
                nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1,
                              groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                )
            )

        # ---- Bottom-up PAN ----
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act)
            )
            self.pan_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        # ---- 多尺度窗口自注意力 (FPN+PAN 融合之后作用于 outs) ----
        # 仅对 use_window_attn_idx 中指定的层级创建 WindowAttentionBlock
        self.window_attn_blocks = nn.ModuleList()
        for idx in range(len(in_channels)):
            if idx in use_window_attn_idx:
                self.window_attn_blocks.append(
                    WindowAttentionBlock(
                        dim=hidden_dim,
                        window_size=window_size,
                        nhead=nhead,
                        dropout=dropout,
                    )
                )
            else:
                self.window_attn_blocks.append(nn.Identity())

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        """
        feats: list of [B, C_i, H_i, W_i], 长度 = len(in_channels)

        前向流程:
          1. input_proj:        通道降维
          2. Transformer encoder: P5 全局自注意力 (含 sincos position embedding)
          3. FPN top-down:      bilinear 上采样 + 抗混叠 depthwise conv + CSPRepLayer
          4. PAN bottom-up:     stride-2 卷积下采样 + CSPRepLayer
          5. 窗口自注意力:      对 P3/P4 的输出特征做局部窗口自注意力 (新增)
        """
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # ---- Step 1: 全局 Transformer encoder (P5) ----
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(
                    -1, self.hidden_dim, h, w).contiguous()

        # ---- Step 2: Top-down FPN (bilinear + anti-aliasing) ----
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            fpn_idx = len(self.in_channels) - 1 - idx

            feat_high = self.lateral_convs[fpn_idx](feat_high)
            inner_outs[0] = feat_high

            # bilinear 替换 nearest, 避免棋盘格伪影
            upsample_feat = F.interpolate(feat_high, scale_factor=2.,
                                           mode='bilinear', align_corners=False)
            # depthwise conv 平滑双线性插值的混叠效应
            upsample_feat = self.fpn_aa_convs[fpn_idx](upsample_feat)

            inner_out = self.fpn_blocks[fpn_idx](
                torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        # ---- Step 3: Bottom-up PAN ----
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_high], dim=1))
            outs.append(out)

        # ---- Step 4: 多尺度窗口自注意力 (FPN+PAN 融合之后) ----
        # 此时 outs 已包含多尺度语义上下文, 窗口注意力能更有效地捕获局部空间关系
        # P3 (outs[0]): 覆盖极小目标; P4 (outs[1]): 覆盖中小目标
        outs = [block(o) for block, o in zip(self.window_attn_blocks, outs)]

        return outs
