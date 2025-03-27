import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn import init


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None,
                 BN_momentum=0.1):
        super(ConvLayer, self).__init__()

        bias = False if norm == 'BN' else True
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if activation is not None:
            self.activation = getattr(torch, activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.conv2d(x.float())

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class TransposedConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None):
        super(TransposedConvLayer, self).__init__()

        bias = False if norm == 'BN' else True
        self.transposed_conv2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=2, padding=padding, output_padding=1, bias=bias)

        if activation is not None:
            self.activation = getattr(torch, activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.transposed_conv2d(x)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None):
        super(UpsampleConvLayer, self).__init__()

        bias = False if norm == 'BN' else True
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        if activation is not None:
            self.activation = getattr(torch, activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        x_upsampled = f.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.conv2d(x_upsampled)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class RecurrentConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 recurrent_block_type='convlstm', activation='relu', norm=None, BN_momentum=0.1):
        super(RecurrentConvLayer, self).__init__()

        assert(recurrent_block_type in ['convlstm', 'convgru'])
        self.recurrent_block_type = recurrent_block_type
        if self.recurrent_block_type == 'convlstm':
            RecurrentBlock = ConvLSTM
        else:
            RecurrentBlock = ConvGRU
        self.conv = ConvLayer(in_channels, out_channels, kernel_size, stride, padding, activation, norm,
                              BN_momentum=BN_momentum)
        self.recurrent_block = RecurrentBlock(input_size=out_channels, hidden_size=out_channels, kernel_size=3)

    def forward(self, x, prev_state):
        x = self.conv(x)
        state = self.recurrent_block(x, prev_state)
        x = state[0] if self.recurrent_block_type == 'convlstm' else state
        return x, state


class DownsampleRecurrentConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, recurrent_block_type='convlstm', padding=0, activation='relu'):
        super(DownsampleRecurrentConvLayer, self).__init__()

        self.activation = getattr(torch, activation)

        assert(recurrent_block_type in ['convlstm', 'convgru'])
        self.recurrent_block_type = recurrent_block_type
        if self.recurrent_block_type == 'convlstm':
            RecurrentBlock = ConvLSTM
        else:
            RecurrentBlock = ConvGRU
        self.recurrent_block = RecurrentBlock(input_size=in_channels, hidden_size=out_channels, kernel_size=kernel_size)

    def forward(self, x, prev_state):
        state = self.recurrent_block(x, prev_state)
        x = state[0] if self.recurrent_block_type == 'convlstm' else state
        x = f.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        return self.activation(x), state


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm=None,
                 BN_momentum=0.1):
        super(ResidualBlock, self).__init__()
        bias = False if norm == 'BN' else True
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.norm = norm
        if norm == 'BN':
            self.bn1 = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
            self.bn2 = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
        elif norm == 'IN':
            self.bn1 = nn.InstanceNorm2d(out_channels)
            self.bn2 = nn.InstanceNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.norm in ['BN', 'IN']:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.norm in ['BN', 'IN']:
            out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ConvLSTM(nn.Module):
    """Adapted from: https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py """

    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        pad = kernel_size // 2

        # cache a tensor filled with zeros to avoid reallocating memory at each inference step if --no-recurrent is enabled
        self.zero_tensors = {}

        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=pad)


    def forward(self, input_, prev_state=None):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:

            # create the zero tensor if it has not been created already
            state_size = tuple([batch_size, self.hidden_size] + list(spatial_size))
            if state_size not in self.zero_tensors:
                # allocate a tensor with size `spatial_size`, filled with zero (if it has not been allocated already)
                self.zero_tensors[state_size] = (
                    torch.zeros(state_size, dtype=input_.dtype).to(input_.device),
                    torch.zeros(state_size, dtype=input_.dtype).to(input_.device)
                )

            prev_state = self.zero_tensors[tuple(state_size)]

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell


class ConvGRU(nn.Module):
    """
    Generate a convolutional GRU cell
    Adapted from: https://github.com/jacobkimmel/pytorch_convgru/blob/master/convgru.py
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

        init.orthogonal_(self.reset_gate.weight)
        init.orthogonal_(self.update_gate.weight)
        init.orthogonal_(self.out_gate.weight)
        init.constant_(self.reset_gate.bias, 0.)
        init.constant_(self.update_gate.bias, 0.)
        init.constant_(self.out_gate.bias, 0.)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = torch.zeros(state_size, dtype=input_.dtype).to(input_.device)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state


class RecurrentResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels,
                 recurrent_block_type='convlstm', norm=None, BN_momentum=0.1):
        super(RecurrentResidualLayer, self).__init__()

        assert(recurrent_block_type in ['convlstm', 'convgru'])
        self.recurrent_block_type = recurrent_block_type
        if self.recurrent_block_type == 'convlstm':
            RecurrentBlock = ConvLSTM
        else:
            RecurrentBlock = ConvGRU
        self.conv = ResidualBlock(in_channels=in_channels,
                                  out_channels=out_channels,
                                  norm=norm,
                                  BN_momentum=BN_momentum)
        self.recurrent_block = RecurrentBlock(input_size=out_channels,
                                              hidden_size=out_channels,
                                              kernel_size=3)

    def forward(self, x, prev_state):
        x = self.conv(x)
        state = self.recurrent_block(x, prev_state)
        x = state[0] if self.recurrent_block_type == 'convlstm' else state
        return x, state


class NAM(nn.Module):
    """Adapted from: https://github.com/chengtan9907/OpenSTL/blob/master/openstl/modules/predrnn_modules.py """

    def __init__(self, input_size, hidden_size, kernel_size):
        super(NAM, self).__init__()

        # convert e2vid style to predrnn style
        in_channel = input_size
        num_hidden = hidden_size
        filter_size = kernel_size
        stride = 1

        # cache a tensor filled with zeros to avoid reallocating memory at each inference step if --no-recurrent is enabled
        self.zero_tensors = {}

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0

        self.conv_x = nn.Sequential(nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), )
        self.conv_h = nn.Sequential(nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), )
        self.conv_m = nn.Sequential(nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), )
        self.conv_o = nn.Sequential(nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), )
        # conv_last: channel=channel/2 with same size
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)

        # self.conv = nn.Conv2d(input_size, input_size, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t, m_t):

        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new


class ContextBlock2d(nn.Module):
    """Adapted from: https://github.com/Charlie0215/AWNet-Attentive-Wavelet-Network-for-Image-ISP/blob/master/models/modules_3channel.py"""

    def __init__(self, inplanes=9, planes=32, pool='att', fusions=['channel_add'], ratio=4):
        super(ContextBlock2d, self).__init__()
        assert pool in ['avg', 'att']
        assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
        assert len(fusions) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.planes = planes
        self.pool = pool
        self.fusions = fusions
        if 'att' in pool:
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)  # context Modeling
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusions:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes // ratio, kernel_size=1),
                nn.LayerNorm([self.planes // ratio, 1, 1]),
                nn.PReLU(),
                nn.Conv2d(self.planes // ratio, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes // ratio, kernel_size=1),
                nn.LayerNorm([self.planes // ratio, 1, 1]),
                nn.PReLU(),
                nn.Conv2d(self.planes // ratio, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pool == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(3)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = x * channel_mul_term
        else:
            out = x
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out


class NAM_Complete_add(nn.Module):
    """Adapted from: https://github.com/chengtan9907/OpenSTL/blob/master/openstl/modules/predrnn_modules.py """

    def __init__(self, input_size, hidden_size, kernel_size):
        super(NAM_Complete_add, self).__init__()

        # convert e2vid style to predrnn style
        in_channel = input_size
        num_hidden = hidden_size
        filter_size = kernel_size
        stride = 1

        # cache a tensor filled with zeros to avoid reallocating memory at each inference step if --no-recurrent is enabled
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0

        self.conv_x = nn.Sequential(nn.Conv2d(in_channel, num_hidden * 5, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), )
        self.conv_h = nn.Sequential(nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), )
        self.conv_m = nn.Sequential(nn.Conv2d(num_hidden, num_hidden * 1, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), )
        self.conv_o = nn.Sequential(nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), )
        # conv_last: channel=channel/2 with same size
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)

        # Local Adaptation Gate(LAG)
        self.LAG_conv = nn.Conv2d(input_size, input_size, kernel_size=1, stride=1, padding=0, bias=False)

        # Global Context Block(GCB)
        self.conv_1x1 = nn.Conv2d(in_channel * 2, in_channel * 2, kernel_size=1, padding=0)
        self.GCB = ContextBlock2d(inplanes=in_channel * 2, planes=in_channel * 2)

    def forward(self, x_t, h_t, c_t, m_t):

        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        # i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_x, f_x, g_x, x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        # i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)
        m_prime = m_concat

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        # ------------Local Adaptation Gate(LAG)------------
        alpha = torch.exp(torch.sigmoid(self.LAG_conv(x_t)))
        f_t = torch.sigmoid(f_t - alpha * i_t)
        # --------------------------------------------------
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        # ------------Global Context Block(GCB)-------------
        gcb_in = torch.cat((x_prime, m_prime), 1)
        gcb = self.conv_1x1(gcb_in)
        gcb = self.GCB(gcb)
        gcb = gcb + gcb_in
        gcb_out = self.conv_last(gcb)
        # --------------------------------------------------

        i_t_prime = torch.sigmoid(gcb_out)
        f_t_prime = torch.sigmoid(gcb_out)
        g_t_prime = torch.tanh(gcb_out)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new


class NAM_Complete_concat(nn.Module):
    """Adapted from: https://github.com/chengtan9907/OpenSTL/blob/master/openstl/modules/predrnn_modules.py """

    def __init__(self, input_size, hidden_size, kernel_size):
        super(NAM_Complete_concat, self).__init__()

        # convert e2vid style to predrnn style
        in_channel = input_size
        num_hidden = hidden_size
        filter_size = kernel_size
        stride = 1

        # cache a tensor filled with zeros to avoid reallocating memory at each inference step if --no-recurrent is enabled
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0

        self.Gates = nn.Conv2d(in_channel + num_hidden, 3 * num_hidden, kernel_size, padding=self.padding)
        self.conv_o = nn.Sequential(nn.Conv2d(num_hidden * 4, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), )
        # conv_last: channel=channel/2 with same size
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)

        # Local Adaptation Gate(LAG)
        self.LAG_conv = nn.Conv2d(input_size, input_size, kernel_size=1, stride=1, padding=0, bias=False)

        # Global Context Block(GCB)
        self.conv_1x1 = nn.Conv2d(in_channel * 2, in_channel * 2, kernel_size=1, padding=0)
        self.GCB = ContextBlock2d(inplanes=in_channel * 2, planes=in_channel * 2)

    def forward(self, x_t, h_t, c_t, m_t):

        stacked_inputs = torch.cat((x_t, h_t), 1)
        gates = self.Gates(stacked_inputs)
        i_t, f_t, g_t = gates.chunk(3, 1)

        i_t = torch.sigmoid(i_t)
        f_t = torch.sigmoid(f_t)
        # ------------Local Adaptation Gate(LAG)------------
        alpha = torch.exp(torch.sigmoid(self.LAG_conv(x_t)))
        f_t = torch.sigmoid(f_t - alpha * i_t)
        # --------------------------------------------------
        g_t = torch.tanh(g_t)

        c_new = f_t * c_t + i_t * g_t

        # ------------Global Context Block(GCB)-------------
        gcb_in = torch.cat((x_t, m_t), 1)
        gcb = self.conv_1x1(gcb_in)
        gcb = self.GCB(gcb)
        gcb = gcb + gcb_in
        gcb_out = self.conv_last(gcb)
        # --------------------------------------------------

        i_t_prime = torch.sigmoid(gcb_out)
        f_t_prime = torch.sigmoid(gcb_out)
        g_t_prime = torch.tanh(gcb_out)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.cat((x_t, h_t, c_new, m_new), 1)
        o_t = torch.sigmoid(self.conv_o(o_t))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new


class NAM_withoutGCB(nn.Module):
    """Adapted from: https://github.com/chengtan9907/OpenSTL/blob/master/openstl/modules/predrnn_modules.py """

    def __init__(self, input_size, hidden_size, kernel_size):
        super(NAM_withoutGCB, self).__init__()

        # convert e2vid style to predrnn style
        in_channel = input_size
        num_hidden = hidden_size
        filter_size = kernel_size
        stride = 1

        # cache a tensor filled with zeros to avoid reallocating memory at each inference step if --no-recurrent is enabled
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0

        self.conv_x = nn.Sequential(nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), )
        self.conv_h = nn.Sequential(nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), )
        self.conv_m = nn.Sequential(nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), )
        self.conv_o = nn.Sequential(nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), )
        # conv_last: channel=channel/2 with same size
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)

        # Local Adaptation Gate(LAG)
        self.LAG_conv = nn.Conv2d(input_size, input_size, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t, m_t):

        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)

        # ------------Local Adaptation Gate(LAG)------------
        alpha = torch.exp(torch.sigmoid(self.LAG_conv(x_t)))
        f_t = torch.sigmoid(f_t - alpha * i_t)
        # --------------------------------------------------
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new


class NAM_GCB(nn.Module):
    """Adapted from: https://github.com/chengtan9907/OpenSTL/blob/master/openstl/modules/predrnn_modules.py """

    def __init__(self, input_size, hidden_size, kernel_size):
        super(NAM_GCB, self).__init__()

        # convert e2vid style to predrnn style
        in_channel = input_size
        num_hidden = hidden_size
        filter_size = kernel_size
        stride = 1

        # cache a tensor filled with zeros to avoid reallocating memory at each inference step if --no-recurrent is enabled
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0

        self.conv_x = nn.Sequential(nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), )
        self.conv_h = nn.Sequential(nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), )
        self.conv_m = nn.Sequential(nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), )
        self.conv_o = nn.Sequential(nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), )
        # conv_last: channel=channel/2 with same size
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)

        # Local Adaptation Gate(LAG)
        self.LAG_conv = nn.Conv2d(input_size, input_size, kernel_size=1, stride=1, padding=0, bias=False)

        # Global Context Block(GCB)
        self.conv_1x1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, padding=0)
        self.GCB = ContextBlock2d(inplanes=in_channel, planes=in_channel)

    def forward(self, x_t, h_t, c_t, m_t):

        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        # ------------Local Adaptation Gate(LAG)------------
        alpha = torch.exp(torch.sigmoid(self.LAG_conv(x_t)))
        f_t = torch.sigmoid(f_t - alpha * i_t)
        # --------------------------------------------------
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        # # ------------Global Context Block(GCB)-------------
        gcb = self.conv_1x1(h_new)
        gcb = self.GCB(gcb)
        h_gcb_out = gcb + h_new
        # # --------------------------------------------------

        return h_gcb_out, c_new, m_new


class RecurrentConvLayer_NAM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 recurrent_block_type='NAM_Complete_add', activation='relu', norm=None, BN_momentum=0.1):
        super(RecurrentConvLayer_NAM, self).__init__()

        # assert(recurrent_block_type in ['NAM', 'NAM_Complete_add', 'NAM_Complete_concat', 'NAM_GCB'])
        self.recurrent_block_type = recurrent_block_type
        if self.recurrent_block_type == 'NAM':
            RecurrentBlock = NAM
        elif self.recurrent_block_type == 'NAM_Complete_add':
            RecurrentBlock = NAM_Complete_add
        elif self.recurrent_block_type == 'NAM_Complete_concat':
            RecurrentBlock = NAM_Complete_concat
        elif self.recurrent_block_type == 'NAM_withoutGCB':
            RecurrentBlock = NAM_withoutGCB
        elif self.recurrent_block_type == 'CausalLSTMCell_denoise_with_flow':
            RecurrentBlock = CausalLSTMCell_denoise_with_flow
        else:
            RecurrentBlock = NAM_GCB

        self.conv = ConvLayer(in_channels, out_channels, kernel_size, stride, padding, activation, norm,
                              BN_momentum=BN_momentum)
        self.conv_mem = ConvLayer(in_channels, out_channels, kernel_size, stride, padding, activation, norm,
                              BN_momentum=BN_momentum)
        self.recurrent_block = RecurrentBlock(input_size=out_channels, hidden_size=out_channels, kernel_size=3)

    def forward(self, x, x_warp, h_t, c_t, m_t):
        x = self.conv(x)
        m_t = self.conv_mem(m_t)
        h_new, c_new, m_new = self.recurrent_block(x, x_warp, h_t, c_t, m_t)
        return h_new, c_new, m_new


class RecurrentConvLayer_NAM_GCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 recurrent_block_type='NAM_Complete_add', activation='relu', norm=None, BN_momentum=0.1):
        super(RecurrentConvLayer_NAM_GCB, self).__init__()

        RecurrentBlock = NAM_withoutGCB

        # in_channels, out_channels : [32, 64], [64, 128], [128, 256]
        self.conv = ConvLayer(in_channels, out_channels, kernel_size, stride, padding, activation, norm,
                              BN_momentum=BN_momentum)
        self.conv_mem = ConvLayer(in_channels, out_channels, kernel_size, stride, padding, activation, norm,
                                  BN_momentum=BN_momentum)
        self.recurrent_block = RecurrentBlock(input_size=out_channels, hidden_size=out_channels, kernel_size=3)

        # Global Context Block(GCB)
        self.conv_1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.GCB = ContextBlock2d(inplanes=in_channels, planes=in_channels)

    def forward(self, x, h_t, c_t, m_t):

        # # ------------Global Context Block(GCB)-------------
        gcb = self.conv_1x1(x)
        gcb = self.GCB(gcb)
        x_gcb = gcb + x
        # # --------------------------------------------------

        # x = self.conv(x)
        x = self.conv(x_gcb)
        m_t = self.conv_mem(m_t)
        h_new, c_new, m_new = self.recurrent_block(x, h_t, c_t, m_t)

        return h_new, c_new, m_new


    # v1: v53
class CausalLSTMCell_denoise_with_flow(nn.Module):
    """Adapted from: https://github.com/chengtan9907/OpenSTL/blob/OpenSTL-Lightning/openstl/modules/predrnnpp_modules.py """

    def __init__(self, input_size, hidden_size, kernel_size):
        super(CausalLSTMCell_denoise_with_flow, self).__init__()
        # convert e2vid style to predrnn style
        in_channel = input_size
        num_hidden = hidden_size
        filter_size = kernel_size
        stride = 1

        # cache a tensor filled with zeros to avoid reallocating memory at each inference step if --no-recurrent is enabled
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0

        self.conv_x = nn.Sequential(nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), )
        self.conv_h = nn.Sequential(nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), )
        self.h_denoise = nn.Sequential(nn.Conv2d(num_hidden * 3, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), )
        self.conv_c = nn.Sequential(nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), )
        self.conv_m = nn.Sequential(nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), )
        self.conv_o = nn.Sequential(nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False), )
        self.conv_c2m = nn.Sequential(nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),)
        self.conv_om = nn.Sequential(nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),)

        # Global Context Block(GCB)
        self.conv_1x1 = nn.Conv2d(in_channel*3, in_channel*3, kernel_size=1, padding=0)
        self.GCB = ContextBlock2d(inplanes=in_channel*3, planes=in_channel*3)

        # conv_last: channel=channel/2 with same size
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)

        # Local Adaptation Gate(LAG)
        self.LAG_conv = nn.Conv2d(input_size, input_size, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t0_warp, h_t, c_t, m_t):
        # print('--------------------')
        diff = x_t - h_t
        warp = x_t * h_t0_warp
        # diff.zero_()  # ablation study
        # h_t.zero_()  # ablation study
        input_concat = torch.cat((diff, warp, h_t), 1)

        # # ------------Global Context Block(GCB)-------------
        gcb = self.conv_1x1(input_concat)
        gcb = self.GCB(gcb)
        h_gcb_out = gcb + input_concat
        h_denoise = self.h_denoise(h_gcb_out)
        # # --------------------------------------------------
        # h_denoise.zero_()  # ablation study

        x_concat = self.conv_x(x_t)
        # h_concat = self.conv_h(h_t)
        h_concat = self.conv_h(h_denoise)
        c_concat = self.conv_c(c_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)

        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, m_m = torch.split(m_concat, self.num_hidden, dim=1)
        i_c, f_c, g_c = torch.split(c_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h + i_c)
        f_t = torch.sigmoid(f_x + f_h + f_c + self._forget_bias)
        # ------------Local Adaptation Gate(LAG)------------
        alpha = torch.exp(torch.sigmoid(self.LAG_conv(x_t)))
        f_t = torch.sigmoid(f_t - alpha * i_t)
        # --------------------------------------------------
        g_t = torch.tanh(g_x + g_h + g_c)

        c_new = f_t * c_t + i_t * g_t

        c2m = self.conv_c2m(c_new)
        i_c, g_c, f_c, o_c = torch.split(c2m, self.num_hidden, dim=1)

        i_t_prime = torch.sigmoid(i_x_prime + i_m + i_c)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + f_c + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_c)

        m_new = f_t_prime * torch.tanh(m_m) + i_t_prime * g_t_prime
        o_m = self.conv_om(m_new)

        o_t = torch.tanh(o_x + o_h + o_c + o_m)
        mem = torch.cat((c_new, m_new), 1)
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new
