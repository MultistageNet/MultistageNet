import torch
import torch.nn as nn

class DepthwiseConvBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation, activation):
        super(DepthwiseConvBlock, self).__init__()
        receptive_field = (kernel_size-1)*dilation + 1
        padding = receptive_field // 2
        self.activation = activation

        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, dilation=dilation, padding=padding, groups=in_channels)
        self.act1 = getattr(nn, activation)()
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, dilation=dilation, padding=padding, groups=in_channels)
        self.act2 = getattr(nn, activation)()

        self.remove = 1 if receptive_field % 2 == 0 else 0

    def forward(self, x):
        resid = x

        x = self.conv1(x)
        if self.remove > 0: x = x[:, :, : -self.remove]
        x = self.act1(x)

        x = self.conv2(x)
        
        if self.remove > 0: x = x[:, :, : -self.remove]
        x = self.act2(x)
            
        x = x + resid
        return x

class DepthwiseDilatedConvolution(nn.Module):
    def __init__(self, in_channels, kernel_size, num_layers, activation):
        super(DepthwiseDilatedConvolution, self).__init__()
        self.in_channels = in_channels # input dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.activation = activation

        self.layers = nn.Sequential(*[
                DepthwiseConvBlock(
                    in_channels,
                    kernel_size=kernel_size,
                    dilation=2**i,
                    activation = activation
                )
                for i in range(num_layers)    
            ])

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.layers(x)
        x = x.permute(0,2,1)

        return x
    
class RegressorDepthwiseDilatedConvolution(nn.Module):
    def __init__(self, in_channels, kernel_size, num_layers, activation, d_ff, pred_len):
        super(RegressorDepthwiseDilatedConvolution, self).__init__()
        self.in_channels = in_channels # input dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.activation = activation

        self.layers = nn.Sequential(*[
                DepthwiseConvBlock(
                    in_channels,
                    kernel_size=kernel_size,
                    dilation=2**i,
                    activation = activation
                )
                for i in range(num_layers)    
            ])

        self.positionwise_ConvBlock_1 = nn.Conv1d(in_channels, d_ff, kernel_size=1)
        self.act = getattr(nn, activation)()
        self.positionwise_ConvBlock_2 = nn.Conv1d(d_ff, in_channels, kernel_size=1)

        self.regressor = nn.Sequential(
            nn.Linear(in_channels, in_channels//2),
            getattr(nn, activation)(),
            nn.Linear(in_channels//2, in_channels//2),
            getattr(nn, activation)(),
            nn.Linear(in_channels//2, pred_len))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.layers(x)
        x = self.positionwise_ConvBlock_1(x)
        x = self.act(x)
        x = self.positionwise_ConvBlock_2(x)
        x = x.permute(0,2,1)

        x = self.regressor(x)
        x = x[:,-1,:]

        return x