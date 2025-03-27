import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn import init
from .submodules import \
    ConvLayer, UpsampleConvLayer, TransposedConvLayer, \
    RecurrentConvLayer, ResidualBlock, ConvLSTM, \
    ConvGRU, RecurrentResidualLayer, RecurrentConvLayer_NAM, RecurrentConvLayer_NAM_GCB

from .model_util import *


class BaseUNet(nn.Module):
    def __init__(self, base_num_channels, num_encoders, num_residual_blocks,
                 num_output_channels, skip_type, norm, use_upsample_conv,
                 num_bins, recurrent_block_type=None, kernel_size=5,
                 channel_multiplier=2, crop_size=112, mlp_layers=None, use_cnn_representation=False, normalize=False, combine_voxel=False,
                 RepCNN_num_layers=3, RepCNN_kernel_size=3, RepCNN_padding=1, RepCNN_channel=64, recurrent_network='E2VID', use_dynamic_decoder=False):
        super(BaseUNet, self).__init__()
        self.base_num_channels = base_num_channels
        self.num_encoders = num_encoders
        self.num_residual_blocks = num_residual_blocks
        self.num_output_channels = num_output_channels
        self.kernel_size = kernel_size
        self.skip_type = skip_type
        self.norm = norm
        self.num_bins = num_bins
        self.recurrent_block_type = recurrent_block_type
        self.use_dynamic_decoder = use_dynamic_decoder

        self.encoder_input_sizes = [int(self.base_num_channels * pow(channel_multiplier, i)) for i in range(self.num_encoders)]
        self.encoder_output_sizes = [int(self.base_num_channels * pow(channel_multiplier, i + 1)) for i in range(self.num_encoders)]
        self.max_num_channels = self.encoder_output_sizes[-1]
        self.skip_ftn = eval('skip_' + skip_type)
        print('Using skip: {}'.format(self.skip_ftn))
        if use_upsample_conv:
            print('Using UpsampleConvLayer (slow, but no checkerboard artefacts)')
            self.UpsampleLayer = UpsampleConvLayer
        else:
            print('Using TransposedConvLayer (fast, with checkerboard artefacts)')
            self.UpsampleLayer = TransposedConvLayer
        assert(self.num_output_channels > 0)
        print(f'Kernel size {self.kernel_size}')
        print(f'Skip type {self.skip_type}')
        print(f'norm {self.norm}')

    def build_resblocks(self):
        self.resblocks = nn.ModuleList()
        for i in range(self.num_residual_blocks):
            self.resblocks.append(ResidualBlock(self.max_num_channels, self.max_num_channels, norm=self.norm))

    def build_decoders(self):
        decoder_input_sizes = reversed(self.encoder_output_sizes)
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        decoders = nn.ModuleList()
        for input_size, output_size in zip(decoder_input_sizes, decoder_output_sizes):
            decoders.append(self.UpsampleLayer(
                input_size if self.skip_type == 'sum' else 2 * input_size,
                output_size, kernel_size=self.kernel_size,
                padding=self.kernel_size // 2, norm=self.norm))
        return decoders

    def build_m_upsample_layer(self):
        decoder_input_sizes = reversed(self.encoder_output_sizes)
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        decoders = nn.ModuleList()
        for input_size, output_size in zip(decoder_input_sizes, decoder_output_sizes):
            decoders.append(self.UpsampleLayer(
                input_size if self.skip_type == 'sum' else 2 * input_size,
                output_size, kernel_size=self.kernel_size,
                padding=self.kernel_size // 2, norm=self.norm))
        return decoders

    def build_prediction_layer(self, num_output_channels, norm=None):
        return ConvLayer(self.base_num_channels if self.skip_type == 'sum' else 2 * self.base_num_channels,
                         num_output_channels, 1, activation=None, norm=norm)


class UNetRecurrent(BaseUNet):
    """
    Compatible with E2VID_lightweight
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block,
    such as a ConvLSTM or a ConvGRU.
    Symmetric, skip connections on every encoding layer.
    """
    def __init__(self, unet_kwargs):
        final_activation = unet_kwargs.pop('final_activation', 'none')
        self.final_activation = getattr(torch, final_activation, None)
        print(f'Using {self.final_activation} final activation')
        unet_kwargs['num_output_channels'] = 1
        super().__init__(**unet_kwargs)
        num_bins = self.num_bins
        self.head = ConvLayer(num_bins, self.base_num_channels,
                              kernel_size=self.kernel_size, stride=1,
                              padding=self.kernel_size // 2)  # N x C x H x W -> N x 32 x H x W

        self.encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.encoders.append(RecurrentConvLayer(input_size, output_size, kernel_size=self.kernel_size, stride=2,
                padding=self.kernel_size // 2,
                recurrent_block_type=self.recurrent_block_type, norm=self.norm))

        self.build_resblocks()
        self.decoders = self.build_decoders()
        self.pred = self.build_prediction_layer(self.num_output_channels, self.norm)
        self.states = [None] * self.num_encoders

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: N x num_output_channels x H x W
        """

        # head
        x = self.head(x)
        head = x

        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            x, state = encoder(x, self.states[i])
            blocks.append(x)
            self.states[i] = state

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder
        for i, decoder in enumerate(self.decoders):
            x = decoder(self.skip_ftn(x, blocks[self.num_encoders - i - 1]))

        # tail
        img = self.pred(self.skip_ftn(x, head))
        if self.final_activation is not None:
            img = self.final_activation(img)
        return {'image': img}


class UNetNIAM_STcell_GCB(BaseUNet):
    """
    Non-stationary Aware
    Compatible with E2VID_lightweight
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block,
    such as a ConvLSTM or a ConvGRU.
    Symmetric, skip connections on every encoding layer.
    """
    def __init__(self, unet_kwargs):
        final_activation = unet_kwargs.pop('final_activation', 'none')
        self.final_activation = getattr(torch, final_activation, None)
        print(f'Using {self.final_activation} final activation')
        unet_kwargs['num_output_channels'] = 1
        super().__init__(**unet_kwargs)
        if 'mlp_layers' in unet_kwargs:
            num_bins = self.num_bins * 2
        else:
            num_bins = self.num_bins
        self.head = ConvLayer(num_bins, self.base_num_channels,
                              kernel_size=self.kernel_size, stride=1,
                              padding=self.kernel_size // 2)  # N x C x H x W -> N x 32 x H x W

        self.encoders = nn.ModuleList()

        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            # input_size, output_size: (32, 64), (64, 128), (128, 256)
            self.encoders.append(RecurrentConvLayer_NAM_GCB(
                input_size, output_size, kernel_size=self.kernel_size, stride=2,
                padding=self.kernel_size // 2,
                recurrent_block_type=self.recurrent_block_type, norm=self.norm))
            # layer1: 32 64 5 2 2 convlstm relu
            # layer2: 64 128 5 2 2 convlstm relu
            # layer3: 128 256 5 2 2 convlstm relu

        self.build_resblocks()
        self.decoders = self.build_decoders()
        self.m_t_UpsampleLayer = self.build_m_upsample_layer()
        self.pred = self.build_prediction_layer(self.num_output_channels, self.norm)
        self.states = None

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: N x num_output_channels x H x W
        """

        # head
        x = self.head(x)
        head = x

        # generate empty prev_state, if None is provided
        # ----------------------initialize----------------------
        batch = x.data.size()[0]
        height, width = x.data.size()[2:]
        if self.states is None:
            # reset state
            self.h_t = []
            self.c_t = []
            for i in range(self.num_encoders):
                zeros_h_c = torch.zeros([batch, int(self.encoder_input_sizes[i] * 2), int(height / (2 ** (i + 1))), int(width / (2 ** (i + 1)))]).to(x.device)
                self.h_t.append(zeros_h_c)
                self.c_t.append(zeros_h_c)
            # reset m_t
            self.m_t = torch.zeros([batch, int(self.encoder_input_sizes[0]), height, width]).to(x.device)
            self.states = 1

        blocks_h, blocks_m = [], []
        self.h_t[0], self.c_t[0], self.m_t = self.encoders[0](x, self.h_t[0], self.c_t[0], self.m_t)
        blocks_h.append(self.h_t[0])
        blocks_m.append(self.m_t)

        # ----------------------encoder----------------------
        for i, encoder in enumerate(self.encoders):
            # i = 0, 1, 2
            if i == 0:
                continue
            self.h_t[i], self.c_t[i], self.m_t = encoder(self.h_t[i - 1], self.h_t[i], self.c_t[i], self.m_t)
            x = self.h_t[i]
            blocks_h.append(x)
            blocks_m.append(self.m_t)

        # ---------------------- m_t upsampler ----------------------
        m_t = blocks_m[-1]
        for i, UpsampleLayer in enumerate(self.m_t_UpsampleLayer):
            m_t = UpsampleLayer(self.skip_ftn(m_t, blocks_m[self.num_encoders - i - 1]))
        self.m_t = m_t

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder
        for i, decoder in enumerate(self.decoders):
            x = decoder(self.skip_ftn(x, blocks_h[self.num_encoders - i - 1]))

        # tail
        img = self.pred(self.skip_ftn(x, head))
        if self.final_activation is not None:
            img = self.final_activation(img)
        return {'image': img}


def warp_images_with_flow(image0, flow01):
    """
        :param image0: [N x C x H x W] input image 0
        :param flow01: [N x 2 x H x W] displacement map from image0 to image1
    """
    t_width, t_height = image0.shape[3], image0.shape[2]
    xx, yy = torch.meshgrid(torch.arange(t_width), torch.arange(t_height))  # xx, yy -> WxH
    xx, yy = xx.to(image0.device), yy.to(image0.device)
    xx.transpose_(0, 1)
    yy.transpose_(0, 1)
    xx, yy = xx.float(), yy.float()

    flow01_x = flow01[:, 0, :, :]  # N x H x W
    flow01_y = flow01[:, 1, :, :]  # N x H x W

    warping_grid_x = xx + flow01_x  # N x H x W
    warping_grid_y = yy + flow01_y  # N x H x W

    # normalize warping grid to [-1,1]
    warping_grid_x = (2 * warping_grid_x / (t_width - 1)) - 1
    warping_grid_y = (2 * warping_grid_y / (t_height - 1)) - 1

    warping_grid = torch.stack([warping_grid_x, warping_grid_y], dim=3)  # 1 x H x W x 2

    image0_warped_to1 = f.grid_sample(image0, warping_grid)

    return image0_warped_to1


class UNet_STcell_Denoise_with_flow(BaseUNet):
    """
    Non-stationary Aware
    Compatible with E2VID_lightweight
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block,
    such as a ConvLSTM or a ConvGRU.
    Symmetric, skip connections on every encoding layer.
    """

    def __init__(self, unet_kwargs):
        final_activation = unet_kwargs.pop('final_activation', 'none')
        self.final_activation = getattr(torch, final_activation, None)
        print(f'Using {self.final_activation} final activation')
        unet_kwargs['num_output_channels'] = 1
        super().__init__(**unet_kwargs)
        if 'mlp_layers' in unet_kwargs:
            num_bins = self.num_bins * 2
        else:
            num_bins = self.num_bins
        self.head = ConvLayer(num_bins, self.base_num_channels,
                              kernel_size=self.kernel_size, stride=1,
                              padding=self.kernel_size // 2)  # N x C x H x W -> N x 32 x H x W

        self.encoders = nn.ModuleList()

        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            # input_size, output_size: (32, 64), (64, 128), (128, 256)
            self.encoders.append(RecurrentConvLayer_NAM(
                input_size, output_size, kernel_size=self.kernel_size, stride=2,
                padding=self.kernel_size // 2,
                recurrent_block_type=self.recurrent_block_type, norm=self.norm))
            # layer1: 32 64 5 2 2 convlstm relu
            # layer2: 64 128 5 2 2 convlstm relu
            # layer3: 128 256 5 2 2 convlstm relu

        self.build_resblocks()
        self.decoders = self.build_decoders()
        self.m_t_UpsampleLayer = self.build_m_upsample_layer()
        self.pred = self.build_prediction_layer(self.num_output_channels, self.norm)
        self.states = None
        self.iter_n = 0

    def forward(self, evs_flow):
        """
        :param evs_flow: (voxel, flow): (N x B x H x W, N x 2 x H x W)
        :return: N x num_output_channels x H x W
        """
        x, flow = evs_flow

        # head
        x = self.head(x)
        head = x

        # generate empty prev_state, if None is provided
        # ----------------------initialize----------------------
        batch = x.data.size()[0]
        height, width = x.data.size()[2:]

        if self.states is None:
            # reset state
            self.h_t = []
            self.c_t = []
            self.h_t0_warp = []
            for i in range(self.num_encoders):
                zeros_h_c = torch.zeros([batch, int(self.encoder_input_sizes[i] * 2), int(height / (2 ** (i + 1))), int(width / (2 ** (i + 1)))]).to(x.device)
                # [b, 64, h/2, w/2]; [b, 128, h/4, w/4]; [b, 256, h/8, w/8]
                self.h_t.append(zeros_h_c)
                self.c_t.append(zeros_h_c)
                self.h_t0_warp.append(zeros_h_c)
            # reset m_t
            self.m_t = torch.zeros([batch, int(self.encoder_input_sizes[0]), height, width]).to(x.device)
            self.states = 1
        else:
            # warp image with optical flow
            for i in range(self.num_encoders):
                flow_warp = f.interpolate(flow, scale_factor=0.5 ** (i+1), mode='bilinear', align_corners=False)
                self.h_t0_warp[i] = warp_images_with_flow(self.h_t[i], -flow_warp)

        blocks_h, blocks_m = [], []
        self.h_t[0], self.c_t[0], self.m_t = self.encoders[0](x, self.h_t0_warp[0], self.h_t[0], self.c_t[0], self.m_t)
        blocks_h.append(self.h_t[0])
        blocks_m.append(self.m_t)

        # ----------------------encoder----------------------
        for i, encoder in enumerate(self.encoders):
            if i == 0:
                continue
            self.h_t[i], self.c_t[i], self.m_t = encoder(self.h_t[i-1], self.h_t0_warp[i], self.h_t[i], self.c_t[i], self.m_t)
            x = self.h_t[i]
            blocks_h.append(x)
            blocks_m.append(self.m_t)

        # ---------------------- m_t upsampler ----------------------
        m_t = blocks_m[-1]
        for i, UpsampleLayer in enumerate(self.m_t_UpsampleLayer):
            m_t = UpsampleLayer(self.skip_ftn(m_t, blocks_m[self.num_encoders - i - 1]))
        self.m_t = m_t

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder
        for i, decoder in enumerate(self.decoders):
            x = decoder(self.skip_ftn(x, blocks_h[self.num_encoders - i - 1]))

        # tail
        img = self.pred(self.skip_ftn(x, head))
        if self.final_activation is not None:
            img = self.final_activation(img)

        # iter number
        self.iter_n += 1

        return {'image': img}
