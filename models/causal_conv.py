import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model_utils import dilation, get_receptive_field_size


class CausalConvNet(nn.Module):
    def __init__(self,
                 n_channels,
                 kernel_size,
                 n_blocks,
                 out_channels,
                 device):

        super(CausalConvNet, self).__init__()

        self.device = device
        self.n_wn_blocks = n_blocks
        self.receptive_field_size = get_receptive_field_size(kernel_size, n_blocks)

        left_padding = kernel_size - 1
        self.padding = (left_padding, 0)

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=n_channels,
                               kernel_size=kernel_size)
        self.wn_blocks = nn.ModuleList([CausalConvNetBlock(n_channels,
                                                           kernel_size,
                                                           dilation(i),
                                                           self.receptive_field_size)
                                        for i in range(self.n_wn_blocks)])
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=out_channels, kernel_size=1)

    def forward(self, x):

        residual = F.pad(x, self.padding)
        residual = self.conv1(residual)
        skip_connections = torch.zeros(self.n_wn_blocks,
                                       x.shape[0],
                                       128,
                                       x.shape[2]-self.receptive_field_size+1,
                                       dtype=x.dtype,
                                       device=self.device)

        for i, block in enumerate(self.wn_blocks):
            residual, skip_out = block(residual)
            skip_connections[i] = skip_out

        z = F.relu(torch.sum(skip_connections, dim=0))
        z = F.relu(self.conv2(z))  # (n, out_channels, len_y)

        return z


class CausalConvNetBlock(nn.Module):
    def __init__(self, n_channels, kernel_size, dilation, receptive_field_size):

        super(CausalConvNetBlock, self).__init__()

        self.receptive_field_size = receptive_field_size
        left_padding = dilation * (kernel_size - 1)
        self.padding = (left_padding, 0)

        self.conv_tanh = nn.Conv1d(in_channels=n_channels,
                                   out_channels=n_channels,
                                   kernel_size=kernel_size,
                                   dilation=dilation)
        self.conv_sigmoid = nn.Conv1d(in_channels=n_channels,
                                      out_channels=n_channels,
                                      kernel_size=kernel_size,
                                      dilation=dilation)

        self.conv_residual = nn.Conv1d(in_channels=n_channels,
                                       out_channels=n_channels,
                                       kernel_size=1)
        self.conv_skip_out = nn.Conv1d(in_channels=n_channels,
                                       out_channels=128,
                                       kernel_size=1)

    def forward(self, x):

        x_ = F.pad(x, self.padding)
        tanh_out = F.tanh(self.conv_tanh(x_))
        sigmoid_out = F.sigmoid(self.conv_sigmoid(x_))

        x_ = torch.mul(tanh_out, sigmoid_out)

        residual = self.conv_residual(x_)
        residual = torch.add(residual, x)

        skip_out = self.conv_skip_out(x_)
        skip_out = skip_out[:, :, self.receptive_field_size-1:]

        return residual, skip_out
    