import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
# local modules

from .model_util import CropParameters, recursive_clone
from .base.base_model import BaseModel

from .unet import UNetRecurrent
from .submodules import ResidualBlock, ConvGRU, ConvLayer
from model.representation_modules import Voxelization
from model.unet import UNetNIAM_STcell_GCB, UNet_STcell_Denoise_with_flow
from model.eraft.eraft import ERAFT


def copy_states(states):
    """
    LSTM states: [(torch.tensor, torch.tensor), ...]
    GRU states: [torch.tensor, ...]
    """
    if states[0] is None:
        return copy.deepcopy(states)
    return recursive_clone(states)


class RepresentationRecurrent(BaseModel):
    """
    Compatible with E2VID_lightweight and Representation network
    Recurrent, UNet-like architecture where each encoder is followed by a ConvLSTM or ConvGRU.
    """
    def __init__(self, sensor_resolution, unet_kwargs):
        super().__init__()
        self.num_bins = unet_kwargs['num_bins']  # legacy
        self.num_encoders = unet_kwargs['num_encoders']  # legacy
        self.crop_size = unet_kwargs['crop_size']
        self.mlp_layers = unet_kwargs['mlp_layers']
        self.normalize = unet_kwargs['normalize']
        # self.normalize = False
        if sensor_resolution:
            height, width = sensor_resolution[0], sensor_resolution[1]
        else:
            height, width = self.crop_size, self.crop_size
        self.representation = Voxelization(unet_kwargs, unet_kwargs['use_cnn_representation'], voxel_dimension=(self.num_bins, height, width),
                                           mlp_layers=self.mlp_layers, activation=nn.LeakyReLU(negative_slope=0.1), pretrained=True, normalize=self.normalize, combine_voxel=unet_kwargs['combine_voxel'])

        self.network = unet_kwargs['recurrent_network']
        if self.network == 'NIAM_STcell_GCB':
            self.unetrecurrent = UNetNIAM_STcell_GCB(unet_kwargs)
        else:
            self.unetrecurrent = UNetRecurrent(unet_kwargs)

        self.crop = CropParameters(width, height, self.num_encoders)  # num of encoder

    @property
    def states(self):
        return copy_states(self.unetrecurrent.states)

    @states.setter
    def states(self, states):
        self.unetrecurrent.states = states

    def reset_states(self):
        if 'NIAM' in self.network or 'NAS' in self.network:
            self.unetrecurrent.states = None
        else:
            self.unetrecurrent.states = [None] * self.unetrecurrent.num_encoders

    def forward(self, x):
        """
        :param x: events[x, y, t, p]
        :return: output dict with image taking values in [0,1], and
                 displacement within event_tensor.
        """
        event_tensor = self.representation.forward(x)
        event_tensor = self.crop.pad(event_tensor)
        output_dict = self.unetrecurrent.forward(event_tensor)
        return output_dict, event_tensor


class RepresentationRecurrent_with_flow(BaseModel):
    """
    Compatible with E2VID_lightweight and Representation network
    Recurrent, UNet-like architecture where each encoder is followed by a ConvLSTM or ConvGRU.
    """
    def __init__(self, sensor_resolution, unet_kwargs):
        super().__init__()
        self.num_bins = unet_kwargs['num_bins']  # legacy
        self.num_encoders = unet_kwargs['num_encoders']  # legacy
        self.crop_size = unet_kwargs['crop_size']
        self.mlp_layers = unet_kwargs['mlp_layers']
        self.normalize = unet_kwargs['normalize']
        # self.normalize = False
        height, width = sensor_resolution[0], sensor_resolution[1]

        if height == 660:
            height = 656
        self.representation = Voxelization(unet_kwargs, unet_kwargs['use_cnn_representation'], voxel_dimension=(self.num_bins, height, width),
                                           mlp_layers=self.mlp_layers, activation=nn.LeakyReLU(negative_slope=0.1), pretrained=True, normalize=self.normalize, combine_voxel=unet_kwargs['combine_voxel'])
        self.network = unet_kwargs['recurrent_network']
        if self.network == 'NIAM_STcell_GCB':
            self.unetrecurrent = UNetNIAM_STcell_GCB(unet_kwargs)
        elif self.network == 'UNet_STcell_Denoise_with_flow':
            self.unetrecurrent = UNet_STcell_Denoise_with_flow(unet_kwargs)
        else:
            self.unetrecurrent = UNetRecurrent(unet_kwargs)
        if unet_kwargs['combine_voxel']:
            self.flow_net = ERAFT(n_first_channels=unet_kwargs['num_bins'])
        else:
            self.flow_net = ERAFT(n_first_channels=unet_kwargs['num_bins']*2)

    @property
    def states(self):
        return copy_states(self.unetrecurrent.states)

    @states.setter
    def states(self, states):
        self.unetrecurrent.states = states

    def reset_states(self):
        if 'NIAM' in self.network or 'STcell' in self.network:
            self.unetrecurrent.states = None
        else:
            self.unetrecurrent.states = [None] * self.unetrecurrent.num_encoders
        self.unetrecurrent.iter_n = 0

    def forward(self, x_0, x):
        """
        :param x_0: events[x, y, t, p], t0~t1
        :param x: events[x, y, t, p], t1~t2
        :return: output dict with image taking values in [0,1], and
                 displacement within event_tensor.
        """
        event_tensor_0 = self.representation.forward(x_0)
        event_tensor = self.representation.forward(x)
        _, flow_predictions = self.flow_net(event_tensor_0, event_tensor)
        output_dict = self.unetrecurrent.forward((event_tensor, flow_predictions[-1]))
        return output_dict, event_tensor, flow_predictions[-1]
