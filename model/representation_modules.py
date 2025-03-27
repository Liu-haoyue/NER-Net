"""
@Time ： 2023/9/19 23:41
@Auth ： Haoyue Liu
@File ：representation_modules.py
"""
import torch.nn as nn
from os.path import join, dirname, isfile
import torch
import numpy as np
import tqdm


class ValueLayer(nn.Module):
    def __init__(self, mlp_layers, activation=nn.ReLU(), num_channels=9):
        # mlp_layers=[1, 100, 100, 1],
        # activation=nn.LeakyReLU(negative_slope=0.1)
        # num_channels=9
        assert mlp_layers[-1] == 1, "Last layer of the mlp must have 1 input channel."
        assert mlp_layers[0] == 1, "First layer of the mlp must have 1 output channel"

        nn.Module.__init__(self)
        self.mlp = nn.ModuleList()
        self.activation = activation

        # create mlp
        in_channels = 1
        for out_channels in mlp_layers[1:]:
            self.mlp.append(nn.Linear(in_channels, out_channels))  # [1, 30], [30, 30], [30, 1]
            in_channels = out_channels

        # init with trilinear kernel (train)
        # path = join(dirname(__file__), "quantization_layer_init", "trilinear_init.pth")

        # if isfile(path):
        #     state_dict = torch.load(path)
        #     try:
        #         self.load_state_dict(state_dict)
        #     except:
        #         self.init_kernel(num_channels)
        # else:
        #     self.init_kernel(num_channels)

    def forward(self, x):
        # create sample of batchsize 1 and input channels 1
        x = x[None, ..., None]  # torch.Size([n]) ---> torch.Size([1, n, 1])

        # apply mlp convolution
        # len(self.mlp[:-1]) = 2
        for i in range(len(self.mlp[:-1])):
            x = self.activation(self.mlp[i](x))  # x.shape: torch.Size([1, len(events), 30])

        x = self.mlp[-1](x)  # x.shape: torch.Size([1, len(events), 30])
        x = x.squeeze()  # x.shape: torch.Size([len(events)])
        # print(x)
        return x

    def init_kernel(self, num_channels):
        ts = torch.zeros((1, 200000))
        optim = torch.optim.Adam(self.parameters(), lr=1e-2)

        torch.manual_seed(1)

        for _ in tqdm.tqdm(range(1000)):  # converges in a reasonable time
            optim.zero_grad()

            ts.uniform_(-1, 1)

            # gt
            gt_values = self.trilinear_kernel(ts, num_channels)

            # pred
            values = self.forward(ts)

            # optimize
            loss = (values - gt_values).pow(2).sum()

            loss.backward()
            optim.step()

    def trilinear_kernel(self, ts, num_channels):
        gt_values = torch.zeros_like(ts)

        gt_values[ts > 0] = (1 - (num_channels - 1) * ts)[ts > 0]
        gt_values[ts < 0] = ((num_channels - 1) * ts + 1)[ts < 0]

        gt_values[ts < -1.0 / (num_channels - 1)] = 0
        gt_values[ts > 1.0 / (num_channels - 1)] = 0

        return gt_values


class QuantizationLayer_trail_combined(nn.Module):
    def __init__(self, dim,
                 mlp_layers=[1, 100, 100, 1],
                 activation=nn.LeakyReLU(negative_slope=0.1),
                 normalize=False):
        nn.Module.__init__(self)
        self.value_layer = ValueLayer(mlp_layers,
                                      activation=activation,
                                      num_channels=dim[0])
        self.dim = dim  # (Bins, H, W)
        self.normalize = normalize

    def forward(self, events):
        # print('-------events representation----------')
        # points is a list, since events can have any size
        # events: shape: torch.Size([n, 5]), [x, y, t, p, b]
        B = int((1 + events[-1, -1]).item())  # Batch size
        num_voxels = int(np.prod(self.dim) * B)  # 3110400 = int(2 * Bins * H * W * Batchsize)，即分为正负极性2 * 离散化后的Bins数量9 * H * W * Batchsize
        vox = events[0].new_full([num_voxels, ], fill_value=0)  # tensor([0., 0., 0.,  ..., 0., 0., 0.], device='cuda:0'), shape: torch.Size([num_voxels])
        C, H, W = self.dim  # Bins H W  C: Channel, number of bins

        # get values for each channel
        x, y, t, p, b = events[:, 0],  events[:, 1], events[:, 2], events[:, 3], events[:, 4]  # x,y,t,p,b: shape:torch.Size([n])

        # normalizing timestamps
        # 按照一个batch中的每个样本序列进行时间戳的归一化至[0, 1]之间， B为Batch
        for bi in range(B):
            if self.normalize:
                # -----归一化方法2-----
                t[events[:, -1] == bi] = t[events[:, -1] == bi] - t[events[:, -1] == bi][0]
                if t[events[:, -1] == bi].max() == 0:
                    continue
                t[events[:, -1] == bi] /= t[events[:, -1] == bi].max()
                # print('*****')
                # print(t[events[:, -1] == bi])
            else:
                # -----标准化方法1-----
                dt = t[events[:, -1] == bi][-1] - t[events[:, -1] == bi][0]
                if dt == 0:
                    continue
                t[events[:, -1] == bi] = (t[events[:, -1] == bi] - t[events[:, -1] == bi][0]) / dt * (C - 1)

        # p = (p + 1) / 2  # maps polarity -1, 1 to 0, 1
        t_ = p * t

        idx_before_bins = x \
                          + W * y \
                          + 0 \
                          + W * H * C * b

        # 这里是将W * H * C(时间维度离散后的数量) * P(极性，分为两个通道) * B(Batch size) 拉成一维后每个单位的序号，共W * H * C * P* B个元素
        # idx_before_bins跳过了离散时间维度(第三行为0)，在后面循环的时候再相加，根据每个事件的(x, y, i_bin(t), p, b)得到每个事件对应的最终索引

        for i_bin in range(C):
            # 这里的时间t已被归一化，t-i_bin/(C-1)为t-0, t-1/8，..., t-1，t在循环中不变，该操作即为循环计算每个时间戳分别在以第C个bins左边界为中心的相对位置
            # 将包含相对位置信息的时间戳序列输入MLP进行优化学习
            # t的长度为事件的个数n，MLP的输入为[1, n, 1]的一维数组，通道数为1，可接受任意长度的t，经过[1, 30], [30, 30], [30, 1]的MLP后输出与t长度一致的张量
            # MLP的输出为每个事件对时间轴上i_bin处体素的权重，再*t得到最终填入每个体素的值(根据索引填，索引记录了每个事件的位置)；这样输入的所有事件都会对不同位置的体素产生一定影响
            # print(i_bin)
            if self.normalize:
                # -----归一化方法2-----
                # print(torch.min(t - i_bin / (C - 1)))
                # print(torch.max(t - i_bin / (C - 1)))
                t_weights = self.value_layer.forward(t_ - i_bin / (C - 1))
            else:
                # -----标准化方法1-----
                # print(t - i_bin)
                t_weights = self.value_layer.forward(t_ - i_bin)

            values = t_ * t_weights

            # draw in voxel grid
            idx = idx_before_bins + W * H * i_bin

            # batch size>1时会出现idx异常越界的情况，原因未知，这里强制限定idx的取值范围(可能是显存OOM)
            idx = torch.clamp(idx.long(), max=vox.shape[0] - 1)

            vox.put_(idx.long(), values, accumulate=True)  # 将网络输出的t值填入的voxel,当同一索引处(即同一像素点、且在同一个Bin触发的事件)触发多个事件时，.put_操作会将同一索引下的所有值相加

        vox = vox.view(-1, C, H, W)  # 如：torch.Size([4(Batch), Bin, H, W])

        return vox


class QuantizationLayer_trail(nn.Module):
    def __init__(self, dim,
                 mlp_layers=[1, 100, 100, 1],
                 activation=nn.LeakyReLU(negative_slope=0.1),
                 normalize=False):
        nn.Module.__init__(self)
        self.value_layer = ValueLayer(mlp_layers,
                                      activation=activation,
                                      num_channels=dim[0])
        self.dim = dim  # (Bins, H, W)
        self.normalize = normalize

    def forward(self, events):
        # print('---------events representation---------')
        # points is a list, since events can have any size
        # events: shape: torch.Size([n, 5]), [x, y, t, p, b]
        B = int((1 + events[-1, -1]).item())  # Batch size
        num_voxels = int(2 * np.prod(self.dim) * B)  # 3110400 = int(2 * Bins * H * W * Batchsize)，即分为正负极性2 * 离散化后的Bins数量9 * H * W * Batchsize
        vox = events[0].new_full([num_voxels, ], fill_value=0)  # tensor([0., 0., 0.,  ..., 0., 0., 0.], device='cuda:0'), shape: torch.Size([num_voxels])
        C, H, W = self.dim  # Bins H W  C: Channel, number of bins

        # get values for each channel
        x, y, t, p, b = events[:, 0],  events[:, 1], events[:, 2], events[:, 3], events[:, 4]  # x,y,t,p,b: shape:torch.Size([n])

        # normalizing timestamps
        # 按照一个batch中的每个样本序列进行时间戳的归一化至[0, 1]之间， B为Batch
        for bi in range(B):
            if self.normalize:
                # -----归一化方法2-----
                t[events[:, -1] == bi] = t[events[:, -1] == bi] - t[events[:, -1] == bi][0]
                if t[events[:, -1] == bi].max() == 0:
                    continue
                t[events[:, -1] == bi] /= t[events[:, -1] == bi].max()
            else:
                # -----标准化方法1-----
                dt = t[events[:, -1] == bi][-1] - t[events[:, -1] == bi][0]
                if dt == 0:
                    continue
                t[events[:, -1] == bi] = (t[events[:, -1] == bi] - t[events[:, -1] == bi][0]) / dt * (C - 1)

        p = (p + 1) / 2  # maps polarity -1, 1 to 0, 1

        idx_before_bins = x \
                          + W * y \
                          + 0 \
                          + W * H * C * p \
                          + W * H * C * 2 * b

        # 这里是将W * H * C(时间维度离散后的数量) * P(极性，分为两个通道) * B(Batch size) 拉成一维后每个单位的序号，共W * H * C * P* B个元素
        # idx_before_bins跳过了离散时间维度(第三行为0)，在后面循环的时候再相加，根据每个事件的(x, y, i_bin(t), p, b)得到每个事件对应的最终索引

        for i_bin in range(C):
            # 这里的时间t已被归一化，t-i_bin/(C-1)为t-0, t-1/8，..., t-1，t在循环中不变，该操作即为循环计算每个时间戳分别在以第C个bins左边界为中心的相对位置
            # 将包含相对位置信息的时间戳序列输入MLP进行优化学习
            # t的长度为事件的个数n，MLP的输入为[1, n, 1]的一维数组，通道数为1，可接受任意长度的t，经过[1, 30], [30, 30], [30, 1]的MLP后输出与t长度一致的张量
            # MLP的输出为每个事件对时间轴上i_bin处体素的权重，再*t得到最终填入每个体素的值(根据索引填，索引记录了每个事件的位置)；这样输入的所有事件都会对不同位置的体素产生一定影响
            # print(i_bin)
            if self.normalize:
                # -----归一化方法2-----
                t_weights = self.value_layer.forward(t - i_bin / (C - 1))
            else:
                # -----标准化方法1-----
                t_weights = self.value_layer.forward(t - i_bin)

            values = t * t_weights

            # draw in voxel grid
            idx = idx_before_bins + W * H * i_bin

            # batch size>1时会出现idx异常越界的情况，原因未知，这里强制限定idx的取值范围(可能是显存OOM)
            idx = torch.clamp(idx.long(), max=vox.shape[0] - 1)

            vox.put_(idx.long(), values, accumulate=True)  # 将网络输出的t值填入的voxel,当同一索引处(即同一像素点、且在同一个Bin触发的事件)触发多个事件时，.put_操作会将同一索引下的所有值相加
            # 输入事件，最终输出已赋值的体素

        vox = vox.view(-1, 2, C, H, W)  # 如：torch.Size([4(Batch), 2, Bin, H, W])
        # vox = torch.cat([vox[:, 0, ...], vox[:, 1, ...]], 1)  # 如：torch.Size([4(Batch), 2*Bin, H, W])
        vox = torch.cat([vox[:, 1, ...], vox[:, 0, ...]], 1)  # 如：torch.Size([4(Batch), 2*Bin, H, W])
        # vox = vox[:, 1, ...] - vox[:, 0, ...]
        # print('-----------------')
        # print(vox.shape)
        # print(torch.max(vox))
        # print(torch.min(vox))
        # vox[vox < -50] = -1000
        # vox[vox > -50] = 100000
        return vox


class RepresentationCNN(nn.Module):
    def __init__(self, channels, net_kwargs):
        super(RepresentationCNN, self).__init__()
        kernel_size = net_kwargs['RepCNN_kernel_size']
        padding = net_kwargs['RepCNN_padding']
        features = net_kwargs['RepCNN_channel']
        num_of_layers = net_kwargs['RepCNN_num_layers']
        # kernel_size = 3
        # padding = 1
        # features = 64
        layers = [nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False), nn.ReLU(inplace=True)]
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.cnn(x)
        return out


class Voxelization(nn.Module):
    def __init__(self,
                 net_kwargs,
                 use_cnn_representation,
                 voxel_dimension=(5, 720, 1280),  # dimension of voxel will be C x H x W
                 mlp_layers=[1, 30, 30, 1],
                 activation=nn.LeakyReLU(negative_slope=0.1),
                 pretrained=True, normalize=False, combine_voxel=False):
        nn.Module.__init__(self)
        if combine_voxel:
            self.quantization_layer = QuantizationLayer_trail_combined(voxel_dimension, mlp_layers, activation, normalize)
            channels = voxel_dimension[0]
        else:
            self.quantization_layer = QuantizationLayer_trail(voxel_dimension, mlp_layers, activation, normalize)
            channels = voxel_dimension[0] * 2
        self.use_cnn_representation = use_cnn_representation
        if use_cnn_representation:
            self.ConvLayer = RepresentationCNN(channels, net_kwargs)

    def forward(self, x):
        # x: events
        vox = self.quantization_layer.forward(x)
        if self.use_cnn_representation:
            out = self.ConvLayer(vox)
        else:
            out = vox
        return out
