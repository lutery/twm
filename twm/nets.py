import copy
import math
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

import utils


def get_activation(nonlinearity, param=None):
    if nonlinearity is None or nonlinearity == 'none' or nonlinearity == 'linear':
        return nn.Identity()
    elif nonlinearity == 'relu':
        return nn.ReLU()
    elif nonlinearity == 'leaky_relu':
        if param is None:
            param = 1e-2
        return nn.LeakyReLU(negative_slope=param)
    elif nonlinearity == 'elu':
        if param is None:
            param = 1.0
        return nn.ELU(alpha=param)
    elif nonlinearity == 'silu':
        return nn.SiLU()
    else:
        raise ValueError(f'Unsupported nonlinearity: {nonlinearity}')


def get_norm_1d(norm, k):
    if norm is None or norm == 'none':
        return nn.Identity()
    elif norm == 'batch_norm':
        return nn.BatchNorm1d(k)
    elif norm == 'layer_norm':
        return nn.LayerNorm(k)
    else:
        raise ValueError(f'Unsupported norm: {norm}')


def get_norm_2d(norm, c, h=None, w=None):
    if norm == 'none':
        return nn.Identity()
    elif norm == 'batch_norm':
        return nn.BatchNorm2d(c)
    elif norm == 'layer_norm':
        assert h is not None and w is not None
        return nn.LayerNorm([c, h, w])
    else:
        raise ValueError(f'Unsupported norm: {norm}')


def _calculate_gain(nonlinearity, param=None):
    if nonlinearity == 'elu':
        nonlinearity = 'selu'
        param = 1
    elif nonlinearity == 'silu':
        nonlinearity = 'relu'
        param = None
    return torch.nn.init.calculate_gain(nonlinearity, param)


def _kaiming_uniform_(tensor, gain):
    # same as torch.nn.init.kaiming_uniform_, but uses gain
    fan = torch.nn.init._calculate_correct_fan(tensor, mode='fan_in')
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    torch.nn.init._no_grad_uniform_(tensor, -bound, bound)


def _get_initializer(name, nonlinearity=None, param=None):
    if nonlinearity is None:
        assert param is None
    if name == 'kaiming_uniform':
        if nonlinearity is None:
            # defaults from PyTorch
            nonlinearity = 'leaky_relu'
            param = math.sqrt(5)
        return lambda x: _kaiming_uniform_(x, gain=_calculate_gain(nonlinearity, param))
    elif name == 'xavier_uniform':
        if nonlinearity is None:
            nonlinearity = 'relu'
        return lambda x: torch.nn.init.xavier_uniform_(x, gain=_calculate_gain(nonlinearity, param))
    elif name == 'orthogonal':
        if nonlinearity is None:
            nonlinearity = 'relu'
        return lambda x: torch.nn.init.orthogonal_(x, gain=_calculate_gain(nonlinearity, param))
    elif name == 'zeros':
        return lambda x: torch.nn.init.zeros_(x)
    else:
        raise ValueError(f'Unsupported initializer: {name}')


def init_(mod, weight_initializer=None, bias_initializer=None, nonlinearity=None, param=None):
    weight_initializer = _get_initializer(weight_initializer, nonlinearity, param) \
        if weight_initializer is not None else lambda x: x
    bias_initializer = _get_initializer(bias_initializer, nonlinearity='linear', param=None) \
        if bias_initializer is not None else lambda x: x

    def fn(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            weight_initializer(m.weight)
            if m.bias is not None:
                bias_initializer(m.bias)

    return mod.apply(fn)


class _MultilayerModule(nn.Module):

    def __init__(self, layer_prefix, ndim, in_dim, num_layers, nonlinearity, param,
                 norm, dropout_p, pre_activation, post_activation,
                 weight_initializer, bias_initializer, final_bias_init):
        super().__init__()
        self.layer_prefix = layer_prefix
        self.ndim = ndim
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.param = param
        self.pre_activation = pre_activation
        self.post_activation = post_activation
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.final_bias_init = final_bias_init

        self.has_norm = norm is not None and norm != 'none'
        self.has_dropout = dropout_p != 0
        self.unsqueeze = in_dim == 0

        self.act = get_activation(nonlinearity, param)

    def reset_parameters(self):
        init_(self, self.weight_initializer, self.bias_initializer, self.nonlinearity, self.param)
        final_layer = getattr(self, f'{self.layer_prefix}{self.num_layers}')
        if not self.post_activation:
            init_(final_layer, self.weight_initializer, self.bias_initializer, nonlinearity='linear', param=None)
        if self.final_bias_init is not None:
            def final_init(m):
                if isinstance(m, (nn.Linear, nn.Conv2d)) and m.bias is not None:
                    with torch.no_grad():
                        m.bias.data.fill_(self.final_bias_init)
            final_layer.apply(final_init)

    def forward(self, x):
        if self.unsqueeze:
            x = x.unsqueeze(-self.ndim)

        if x.ndim > self.ndim + 1:
            batch_shape = x.shape[:-self.ndim]
            x = x.reshape(-1, *x.shape[-self.ndim:])
        else:
            batch_shape = None

        if self.pre_activation:
            if self.has_norm:
                x = getattr(self, 'norm0')(x)
            x = self.act(x)

        for i in range(self.num_layers - 1):
            x = getattr(self, f'{self.layer_prefix}{i + 1}')(x)
            if self.has_norm:
                x = getattr(self, f'norm{i + 1}')(x)
            x = self.act(x)
            if self.has_dropout:
                x = self.dropout(x)
        x = getattr(self, f'{self.layer_prefix}{self.num_layers}')(x)

        if self.post_activation:
            if self.has_norm:
                x = getattr(self, f'norm{self.num_layers}')(x)
            x = self.act(x)

        if batch_shape is not None:
            x = x.unflatten(0, batch_shape)
        return x


class MLP(_MultilayerModule):

    def __init__(self, in_dim, hidden_dims, out_dim, nonlinearity, param=None, norm=None, dropout_p=0, bias=True,
                 pre_activation=False, post_activation=False,
                 weight_initializer='kaiming_uniform', bias_initializer='zeros', final_bias_init=None):
        dims = (in_dim,) + tuple(hidden_dims) + (out_dim,)
        super().__init__('linear', 1, in_dim, len(dims) - 1, nonlinearity, param, norm, dropout_p,
                         pre_activation, post_activation, weight_initializer, bias_initializer, final_bias_init)
        if self.unsqueeze:
            dims = (1,) + dims[1:]

        if pre_activation and self.has_norm:
            norm_layer = get_norm_1d(norm, in_dim)
            self.add_module(f'norm0', norm_layer)

        for i in range(self.num_layers - 1):
            linear_layer = nn.Linear(dims[i], dims[i + 1], bias=bias)
            self.add_module(f'linear{i + 1}', linear_layer)
            if self.has_norm:
                norm_layer = get_norm_1d(norm, dims[i + 1])
                self.add_module(f'norm{i + 1}', norm_layer)

        linear_layer = nn.Linear(dims[-2], dims[-1], bias=bias)
        self.add_module(f'linear{self.num_layers}', linear_layer)

        if post_activation and self.has_norm:
            norm_layer = get_norm_1d(norm, dims[-1])
            self.add_module(f'norm{self.num_layers}', norm_layer)

        if self.has_dropout:
            self.dropout = nn.Dropout(dropout_p)

        self.reset_parameters()


class CNN(_MultilayerModule):
    '''
    这里只是覆盖了构造网络的结构
    总体调用方式在_MultilayerModule中
    '''

    def __init__(self, in_dim, hidden_dims, out_dim, kernel_sizes, strides, paddings, nonlinearity,
                 param=None, norm=None, dropout_p=0, bias=True, padding_mode='zeros', in_shape=None,
                 pre_activation=False, post_activation=False,
                 weight_initializer='kaiming_uniform', bias_initializer='zeros', final_bias_init=None):
        '''
        in_dim: num_channels，输入的通道数，等于帧堆叠数和图像的通道数乘积
        hidden_dims: [h, h * 2, h * 4]
        out_dim: h * 8
        kernel_sizes: [4, 4, 4, 4]
        strides: [2, 2, 2, 2]
        paddings: [0, 0, 0, 0]
        nonlinearity: activation
        norm=norm
        post_activation=True
        '''
        assert len(kernel_sizes) == len(hidden_dims) + 1
        assert len(strides) == len(kernel_sizes) and len(paddings) == len(kernel_sizes)
        # 拼接输入的通道数、隐藏层的通道数和输出的通道数
        dims = (in_dim,) + tuple(hidden_dims) + (out_dim,)
        super().__init__('conv', 3, in_dim, len(dims) - 1, nonlinearity, param, norm, dropout_p,
                         pre_activation, post_activation, weight_initializer, bias_initializer, final_bias_init)
        if self.unsqueeze: # 如果输入的通道数是0，那么就要讲第一个维度进行扩展，默认为1个维度
            dims = (1,) + dims[1:]

        def to_pair(x):
            if isinstance(x, int):
                return x, x
            assert isinstance(x, tuple) and len(x) == 2
            return x

        def calc_out_shape(shape, kernel_size, stride, padding):
            kernel_size, padding, stride = [to_pair(x) for x in (kernel_size, stride, padding)]
            return tuple((shape[j] + 2 * padding[j] - kernel_size[j]) / stride[j] + 1 for j in [0, 1])

        # 这里应该是获取归一化层添加到模型中
        if pre_activation and self.has_norm:
            norm_layer = get_norm_2d(norm, in_dim, in_shape[0], in_shape[1])
            self.add_module('norm0', norm_layer)

        shape = in_shape
        # 增加卷积层
        for i in range(self.num_layers - 1):
            conv_layer = nn.Conv2d(dims[i], dims[i + 1], kernel_sizes[i], strides[i], paddings[i],
                                   bias=bias, padding_mode=padding_mode)
            self.add_module(f'conv{i + 1}', conv_layer)
            if self.has_norm:
                if shape is not None:
                    # 计算卷积层输出的形状
                    shape = calc_out_shape(shape, kernel_sizes[i], strides[i], paddings[i])
                # 获取归一化层,输入的shape时针对LayerNorm使用
                norm_layer = get_norm_2d(norm, dims[i + 1], shape[0], shape[1])
                self.add_module(f'norm{i + 1}', norm_layer)

        # 增加最后一层卷积层
        conv_layer = nn.Conv2d(dims[-2], dims[-1], kernel_sizes[-1], strides[-1], paddings[-1],
                               bias=bias, padding_mode=padding_mode)
        self.add_module(f'conv{self.num_layers}', conv_layer)

        # 增加最后一层归一化层
        if post_activation and self.has_norm:
            shape = calc_out_shape(shape, kernel_sizes[-1], strides[-1], paddings[-1])
            norm_layer = get_norm_2d(norm, dims[-1], shape[0], shape[1])
            self.add_module(f'norm{self.num_layers}', norm_layer)

        # 如果有归一化层则添加dropout层，但是理论上不应该要和norm分开使用吗？
        if self.has_dropout:
            self.dropout = nn.Dropout2d(dropout_p)

        # 重置参数
        self.reset_parameters()


class TransposeCNN(_MultilayerModule):

    def __init__(self, in_dim, hidden_dims, out_dim, kernel_sizes, strides, paddings, nonlinearity,
                 param=None, norm=None, dropout_p=0, bias=True, padding_mode='zeros', in_shape=None,
                 pre_activation=False, post_activation=False,
                 weight_initializer='kaiming_uniform', bias_initializer='zeros', final_bias_init=None):
        assert len(kernel_sizes) == len(hidden_dims) + 1
        assert len(strides) == len(kernel_sizes) and len(paddings) == len(kernel_sizes)
        dims = (in_dim,) + tuple(hidden_dims) + (out_dim,)
        super().__init__('conv_transpose', 3, in_dim, len(dims) - 1, nonlinearity, param, norm, dropout_p,
                         pre_activation, post_activation, weight_initializer, bias_initializer, final_bias_init)
        if self.unsqueeze:
            dims = (1,) + dims[1:]

        def to_pair(x):
            if isinstance(x, int):
                return x, x
            assert isinstance(x, tuple) and len(x) == 2
            return x

        def calc_out_shape(shape, kernel_size, stride, padding):
            kernel_size, padding, stride = [to_pair(x) for x in (kernel_size, stride, padding)]
            return tuple((shape[j] - 1) * stride[j] - 2 * padding[j] + kernel_size[j] for j in [0, 1])

        if pre_activation and self.has_norm:
            norm_layer = get_norm_2d(norm, in_dim, in_shape[0], in_shape[1])
            self.add_module('norm0', norm_layer)

        shape = in_shape
        for i in range(self.num_layers - 1):
            conv_transpose_layer = nn.ConvTranspose2d(dims[i], dims[i + 1], kernel_sizes[i], strides[i], paddings[i],
                                                      bias=bias, padding_mode=padding_mode)
            self.add_module(f'conv_transpose{i + 1}', conv_transpose_layer)
            if self.has_norm:
                if shape is not None:
                    shape = calc_out_shape(shape, kernel_sizes[i], strides[i], paddings[i])
                norm_layer = get_norm_2d(norm, dims[i + 1], shape[0], shape[1])
                self.add_module(f'norm{i + 1}', norm_layer)

        conv_transpose_layer = nn.ConvTranspose2d(dims[-2], dims[-1], kernel_sizes[-1], strides[-1], paddings[-1],
                                                  bias=bias, padding_mode=padding_mode)
        self.add_module(f'conv_transpose{self.num_layers}', conv_transpose_layer)

        if post_activation and self.has_norm:
            shape = calc_out_shape(shape, kernel_sizes[-1], strides[-1], paddings[-1])
            norm_layer = get_norm_2d(norm, dims[-1], shape[0], shape[1])
            self.add_module(f'norm{self.num_layers}', norm_layer)

        if self.has_dropout:
            self.dropout = nn.Dropout2d(dropout_p)

        self.reset_parameters()


# adopted from
# https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py
# and https://github.com/sooftware/attentions/blob/master/attentions.py
class TransformerXLDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, max_length, mem_length, batch_first=False):
        super().__init__()
        # 构建多个transformer解码层
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.mem_length = mem_length
        self.batch_first = batch_first

        self.pos_enc = PositionalEncoding(decoder_layer.dim, max_length, dropout_p=decoder_layer.dropout_p)
        # todo 这两个参数的作用
        self.u_bias = nn.Parameter(torch.Tensor(decoder_layer.num_heads, decoder_layer.head_dim))
        self.v_bias = nn.Parameter(torch.Tensor(decoder_layer.num_heads, decoder_layer.head_dim))
        nn.init.xavier_uniform_(self.u_bias)
        nn.init.xavier_uniform_(self.v_bias)

    def init_mems(self):
        if self.mem_length > 0:
            param = next(self.parameters())
            dtype, device = param.dtype, param.device
            mems = []
            for i in range(self.num_layers + 1):
                mems.append(torch.empty(0, dtype=dtype, device=device))
            return mems
        else:
            return None

    def forward(self, x, positions, attn_mask, mems=None, tgt_length=None, return_attention=False):
        if self.batch_first:
            x = x.transpose(0, 1)

        if mems is None:
            mems = self.init_mems()

        if tgt_length is None:
            tgt_length = x.shape[0]
        assert tgt_length > 0

        pos_enc = self.pos_enc(positions)
        hiddens = [x]
        attentions = []
        out = x
        for i, layer in enumerate(self.layers):
            out, attention = layer(out, pos_enc, self.u_bias, self.v_bias, attn_mask=attn_mask, mems=mems[i])
            hiddens.append(out)
            attentions.append(attention)

        out = out[-tgt_length:]

        if self.batch_first:
            out = out.transpose(0, 1)

        assert len(hiddens) == len(mems)
        with torch.no_grad():
            new_mems = []
            for i in range(len(hiddens)):
                cat = torch.cat([mems[i], hiddens[i]], dim=0)
                new_mems.append(cat[-self.mem_length:].detach())
        if return_attention:
            attention = torch.stack(attentions, dim=-2)
            return out, new_mems, attention
        return out, new_mems


class TransformerXLDecoderLayer(nn.Module):

    def __init__(self, dim, feedforward_dim, head_dim, num_heads, activation, dropout_p, layer_norm_eps=1e-5):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.self_attn = RelativeMultiheadSelfAttention(dim, head_dim, num_heads, dropout_p)
        self.linear1 = nn.Linear(dim, feedforward_dim)
        self.linear2 = nn.Linear(feedforward_dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.act = get_activation(activation)
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

    def _ff(self, x):
        x = self.linear2(self.dropout(self.act(self.linear1(x))))
        return self.dropout(x)

    def forward(self, x, pos_encodings, u_bias, v_bias, attn_mask=None, mems=None):
        out, attention = self.self_attn(x, pos_encodings, u_bias, v_bias, attn_mask, mems)
        out = self.dropout(out)
        out = self.norm1(x + out)
        out = self.norm2(out + self._ff(out))
        return out, attention


class RelativeMultiheadSelfAttention(nn.Module):

    def __init__(self, dim, head_dim, num_heads, dropout_p):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.scale = 1 / (dim ** 0.5)

        self.qkv_proj = nn.Linear(dim, 3 * num_heads * head_dim, bias=False)
        self.pos_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

    def _rel_shift(self, x):
        zero_pad = torch.zeros((x.shape[0], 1, *x.shape[2:]), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)
        x_padded = x_padded.view(x.shape[1] + 1, x.shape[0], *x.shape[2:])
        x = x_padded[1:].view_as(x)
        return x

    def forward(self, x, pos_encodings, u_bias, v_bias, attn_mask=None, mems=None):
        tgt_length, batch_size = x.shape[:2]
        pos_len = pos_encodings.shape[0]

        if mems is not None:
            cat = torch.cat([mems, x], dim=0)
            qkv = self.qkv_proj(cat)
            q, k, v = torch.chunk(qkv, 3, dim=-1)
            q = q[-tgt_length:]
        else:
            qkv = self.qkv_proj(x)
            q, k, v = torch.chunk(qkv, 3, dim=-1)

        pos_encodings = self.pos_proj(pos_encodings)

        src_length = k.shape[0]
        num_heads = self.num_heads
        head_dim = self.head_dim

        q = q.view(tgt_length, batch_size, num_heads, head_dim)
        k = k.view(src_length, batch_size, num_heads, head_dim)
        v = v.view(src_length, batch_size, num_heads, head_dim)
        pos_encodings = pos_encodings.view(pos_len, num_heads, head_dim)

        content_score = torch.einsum('ibnd,jbnd->ijbn', (q + u_bias, k))
        pos_score = torch.einsum('ibnd,jnd->ijbn', (q + v_bias, pos_encodings))
        pos_score = self._rel_shift(pos_score)

        # [tgt_length x src_length x batch_size x num_heads]
        attn_score = content_score + pos_score
        attn_score.mul_(self.scale)

        if attn_mask is not None:
            if attn_mask.ndim == 2:
                attn_score = attn_score.masked_fill(attn_mask[:, :, None, None], -float('inf'))
            elif attn_mask.ndim == 3:
                attn_score = attn_score.masked_fill(attn_mask[:, :, :, None], -float('inf'))

        # [tgt_length x src_length x batch_size x num_heads]
        attn = F.softmax(attn_score, dim=1)
        return_attn = attn
        attn = self.dropout(attn)

        context = torch.einsum('ijbn,jbnd->ibnd', (attn, v))
        context = context.reshape(context.shape[0], context.shape[1], num_heads * head_dim)
        return self.out_proj(context), return_attn


class PositionalEncoding(nn.Module):

    def __init__(self, dim, max_length, dropout_p=0, batch_first=False):
        super().__init__()
        self.dim = dim
        self.max_length = max_length
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

        encodings = torch.zeros(max_length, dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        encodings[:, 0::2] = torch.sin(position * div_term)
        encodings[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('encodings', encodings)

    def forward(self, positions):
        out = self.encodings[positions]
        out = self.dropout(out)
        return out.unsqueeze(0) if self.batch_first else out.unsqueeze(1)


class PredictionNet(nn.Module):

    def __init__(self, modality_order, num_current, embeds, out_heads, embed_dim, activation, norm, dropout_p,
                 feedforward_dim, head_dim, num_heads, num_layers, memory_length, max_length):
        '''
        modality_order: ['z', 'a', 'r'(存在config: dyn_input_rewards则有), 'g'(存在dyn_input_discounts则有)]
        num_current: 2
        embeds: {'z': {'in_dim': z_dim, 'categorical': False}, 'a': {'in_dim': num_actions, 'categorical': True}, 'r'(config: dyn_input_rewards): {'in_dim': 0, 'categorical': False}, 'g'(dyn_input_discounts): {'in_dim': 0, 'categorical': False}}
        out_heads: {'z': {'hidden_dims': config['dyn_z_dims'], 'out_dim': z_dim}, 'r': {'hidden_dims': config['dyn_reward_dims'], 'out_dim': 1, 'final_bias_init': 0.0}, 'g': {'hidden_dims': config['dyn_discount_dims'], 'out_dim': 1, 'final_bias_init': config['env_discount_factor']}}
        embed_dim: config['dyn_embed_dim']
        activation: config['dyn_act']
        norm: config['dyn_norm']
        dropout_p: config['dyn_dropout']
        feedforward_dim: config['dyn_feedforward_dim']
        head_dim: config['dyn_head_dim']
        num_heads: config['dyn_num_heads']
        num_layers: config['dyn_num_layers']
        memory_length: config['wm_memory_length']
        max_length: 1 + config['wm_sequence_length']  # 1 for context
        '''
        super().__init__()
        self.embed_dim = embed_dim
        self.memory_length = memory_length
        self.modality_order = tuple(modality_order) # 将list转换为元组
        self.num_current = num_current

        # embed['in_dim']：观察的特征提取后的维度
        # embed_dim： 这个潜入维度的作用是啥？todo
        # 根据embed的类型来选择是使用nn.Embedding还是MLP，可知除了动作之外，其余的都是MLP
        # todo 那为什么动作需要使用nn.Embedding呢？因为动作是离散的，所以需要使用nn.Embedding来进行嵌入
        # semf.embeds = {
        #   'z': nn.Embedding(z_dim, embed_dim),
        #   'a': nn.Embedding(num_actions, embed_dim),
        #   'r': MLP(0, [], embed_dim, activation, norm=norm, dropout_p=dropout_p, post_activation=True), 虽然传入的是0，但是根据代码，会自动将0转换为1个维度
        #   'g': MLP(0, [], embed_dim, activation, norm=norm, dropout_p=dropout_p, post_activation=True)，虽然传入的是0，但是根据代码，会自动将0转换为1个维度
        #}
        self.embeds = nn.ModuleDict({
            name: nn.Embedding(embed['in_dim'], embed_dim) if embed.get('categorical', False) else
            MLP(embed['in_dim'], [], embed_dim, activation, norm=norm, dropout_p=dropout_p, post_activation=True)
            for name, embed in embeds.items()
        })

        # TransformerXl 特征解码层
        # todo 对比标准的Transformer
        decoder_layer = TransformerXLDecoderLayer(
            embed_dim, feedforward_dim, head_dim, num_heads, activation, dropout_p)

        num_modalities = len(modality_order)
        max_length = max_length * num_modalities + self.num_current
        mem_length = memory_length * num_modalities + self.num_current
        self.transformer = TransformerXLDecoder(decoder_layer, num_layers, max_length, mem_length, batch_first=True)

        # todo 查看如何调用
        self.out_heads = nn.ModuleDict({
            name: MLP(embed_dim, head['hidden_dims'], head['out_dim'], activation, norm=norm, dropout_p=dropout_p,
                      pre_activation=True, final_bias_init=head.get('final_bias_init', None))
            for name, head in out_heads.items()
        })

    @lru_cache(maxsize=20)
    def _get_base_mask(self, src_length, tgt_length, device):
        '''
        src_length: 历史序列长度 * 模态数量 + 当前模态数量=cat z\a\r\g的总长度 (sequence_length + extra - 1 - 1) * 模态数量(4) + 当前模态数量(2)
        tgt_length/src_length: 历史序列长度 * 模态数量 + 当前模态数量=cat z\a\r\g的总长度 (sequence_length + extra - 1 - 1) * 模态数量(4) + 当前模态数量(2)
        input.device: 输入的设备
        '''
        src_mask = torch.ones(tgt_length, src_length, dtype=torch.bool, device=device)
        num_modalities = len(self.modality_order)
        for tgt_index in range(tgt_length):
            # the last indices are always 'current'
            start_index = src_length - self.num_current
            src_index = src_length - tgt_length + tgt_index
            modality_index = (src_index - start_index) % num_modalities
            if modality_index < self.num_current:
                start = max(src_index - (self.memory_length + 1) * num_modalities, 0)
            else:
                start = max(src_index - modality_index - self.memory_length * num_modalities, 0)
            src_mask[tgt_index, start:src_index + 1] = False
        return src_mask

    def _get_mask(self, src_length, tgt_length, device, stop_mask):
        '''
        src_length: 历史序列长度 * 模态数量 + 当前模态数量=cat z\a\r\g的总长度 (sequence_length + extra - 1 - 1) * 模态数量(4) + 当前模态数量(2)
        tgt_length/src_length: 历史序列长度 * 模态数量 + 当前模态数量=cat z\a\r\g的总长度 (sequence_length + extra - 1 - 1) * 模态数量(4) + 当前模态数量(2)
        input.device: 输入的设备
        stop_mask: 停止掩码，shape is (1, sequence_length + extra - 3)
        '''
        # prevent attention over episode ends using stop_mask
        num_modalities = len(self.modality_order) # 模态数量
        assert stop_mask.shape[1] * num_modalities + self.num_current == src_length # 确认输入的src_length和stop_mask的长度是相同的

        src_mask = self._get_base_mask(src_length, tgt_length, device)

        batch_size, seq_length = stop_mask.shape
        stop_mask = stop_mask.t()
        stop_mask_shift_right = torch.cat([stop_mask.new_zeros(1, batch_size), stop_mask], dim=0)
        stop_mask_shift_left = torch.cat([stop_mask, stop_mask.new_zeros(1, batch_size)], dim=0)

        tril = stop_mask.new_ones(seq_length + 1, seq_length + 1).tril()
        src = torch.logical_and(stop_mask_shift_left.unsqueeze(0), tril.unsqueeze(-1))
        src = torch.cummax(src.flip(1), dim=1).values.flip(1)

        shifted_tril = stop_mask.new_ones(seq_length + 1, seq_length + 1).tril(diagonal=-1)
        tgt = torch.logical_and(stop_mask_shift_right.unsqueeze(1), shifted_tril.unsqueeze(-1))
        tgt = torch.cummax(tgt, dim=0).values

        idx = torch.logical_and(src, tgt)

        i, j, k = idx.shape
        idx = idx.reshape(i, 1, j, 1, k).expand(i, num_modalities, j, num_modalities, k) \
            .reshape(i * num_modalities, j * num_modalities, k)

        offset = num_modalities - self.num_current
        if offset > 0:
            idx = idx[:-offset, :-offset]
        idx = idx[-tgt_length:]

        src_mask = src_mask.unsqueeze(-1).tile(1, 1, batch_size)
        src_mask[idx] = True
        return src_mask

    def forward(self, inputs, tgt_length, stop_mask, heads=None, mems=None, return_attention=False):
                '''
        1: inputs: {
            'z': z, shape is (1, sequence_length + extra - 1, z_categoricals * z_categories)
            'a': a, shape is (1, sequence_length + extra - 2)
            'r': r, shape is (1, sequence_length + extra - 3)
            'g': g shape is (1, sequence_length + extra - 3)
        }
        2: tgt_length: 目标长度，sequence_length + extra - 2 或者 sequence_length + extra - 1
        stop_mask: d shape is (1, sequence_length + extra - 3)
        heads: 预测的头部，默认为 ('z', 'r', 'g')
        mems: 记忆，默认为 None
        return_attention: 是否返回注意力，默认为 False
        '''
        modality_order = self.modality_order 
        num_modalities = len(modality_order)
        num_current = self.num_current # 这current代表说输入的inputs确认一定存在的数量有多少个，其余的都是可选存在

        # 这里的是为了确保输入的batch shape是相同的
        assert utils.same_batch_shape([inputs[name] for name in modality_order[:num_current]]) 
        if num_modalities > num_current:
            assert utils.same_batch_shape([inputs[name] for name in modality_order[num_current:]])

        # 根据不同的输入类型来获取不同的特征（z：环境采样特征，a：动作特征，r：奖励特征，g：折扣因子特征）
        # embeds shape is {
        #  'z': (1, sequence_length + extra - 1, embed_dim),
        #  'a': (1, sequence_length + extra - 2, embed_dim),
        #  'r': (1, sequence_length + extra - 3, embed_dim),
        #  'g': (1, sequence_length + extra - 3, embed_dim)
        #}
        embeds = {name: mod(inputs[name]) for name, mod in self.embeds.items()}

        def cat_modalities(xs):
            '''
            传入的xs时一个四维变量，比如传入的是一个z\a\r\g的列表，所以这里就是将z\a\r\g的特征在序列维度上进行拼接
            '''
            batch_size, seq_len, dim = xs[0].shape
            # torch.cat(xs, dim=2)是将所有的xs在seq序列维度上进行拼接
            return torch.cat(xs, dim=2).reshape(batch_size, seq_len * len(xs), dim)

        if mems is None: 
            # # modality_order[0] 表示 z 
            # # 获取 历史序列长度。这里应该是将序列中除了最后一个，其余的当作历史序列
            history_length = embeds[modality_order[0]].shape[1] - 1
            if num_modalities == num_current:
                inputs = cat_modalities([embeds[name] for name in modality_order])
            else:
                # 将所有序列中的历史信息和当前信息进行拼接
                history = cat_modalities([embeds[name][:, :history_length] for name in modality_order])
                current = cat_modalities([embeds[name][:, history_length:] for name in modality_order[:num_current]])
                # 再将cat后的历史信息和当前信息进行拼接
                inputs = torch.cat([history, current], dim=1)
            # tgt_length 是单条序列的长度 - 2 todo 为什么tgt_length是这么长
            tgt_length = (tgt_length - 1) * num_modalities + num_current
            # history_length 历史序列长度
            src_length = history_length * num_modalities + num_current
            # 而以上* num_modalities + num_current就是为了计算拼接后的长度
            assert inputs.shape[1] == src_length
            src_mask = self._get_mask(src_length, src_length, inputs.device, stop_mask)
        else:
            # todo 后续补充
            # modality_order[0] 表示 z
            # 获取 序列长度
            sequence_length = embeds[modality_order[0]].shape[1]
            # switch order so that 'currents' are last
            '''
            modality_order[num_current:] 得到 ('r', 'g')
            modality_order[:num_current] 得到 ('z', 'a')
            合并后变成 ('r', 'g', 'z', 'a')

            在 Transformer 架构中，这种重排序是为了：

            调整注意力机制：

            将历史信息（如奖励、折扣）放在序列的前部
            将当前需要预测的信息（如状态、动作）放在序列的后部
            序列处理顺序：

            在调用 cat_modalities 时，这种排序确保了编码器先处理历史相关模态，再处理当前模态
            这种处理顺序对于预测任务尤为重要
            内存处理：

            在使用记忆机制时，这种排序与内存的存储和检索模式一致
            这是典型的 Transformer XL 或类似架构中的一种优化技术，有助于模型更有效地处理时序依赖关系。
            '''
            # 将embeds中的模态按照modality_order的顺序进行拼接
            inputs = cat_modalities(
                [embeds[name] for name in (modality_order[num_current:] + modality_order[:num_current])])
            tgt_length = tgt_length * num_modalities # todo 这里是将目标长度乘以模态数量
            mem_length = mems[0].shape[0] 
            src_length = mem_length + sequence_length * num_modalities
            src_mask = self._get_mask(src_length, tgt_length, inputs.device, stop_mask)

        positions = torch.arange(src_length - 1, -1, -1, device=inputs.device)
        outputs = self.transformer(
            inputs, positions, attn_mask=src_mask, mems=mems, tgt_length=tgt_length, return_attention=return_attention)
        hiddens, mems, attention = outputs if return_attention else (outputs + (None,))

        # take outputs at last current
        assert hiddens.shape[1] == tgt_length
        out_idx = torch.arange(tgt_length - 1, -1, -num_modalities, device=inputs.device).flip([0])
        hiddens = hiddens[:, out_idx]
        if return_attention:
            attention = attention[out_idx]

        if heads is None:
            heads = self.out_heads.keys()

        out = {name: self.out_heads[name](hiddens) for name in heads}

        return (out, hiddens, mems, attention) if return_attention else (out, hiddens, mems)
