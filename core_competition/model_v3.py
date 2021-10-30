import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa


class UNet(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(UNet, self).__init__()
        self.down_path = nn.ModuleList()
        self.down_path.append(
            UNetConvBlock(
                in_channels=args.in_channels,
                out_channels=args.hidden_channels[0],
                kernel_size=args.kernel_size,
                padding=args.padding,
                num_groups=args.num_groups
            )
        )
        for i in range(1, len(args.hidden_channels)):
            self.down_path.append(
                UNetConvBlock(
                    in_channels=args.hidden_channels[i - 1],
                    out_channels=args.hidden_channels[i],
                    kernel_size=args.kernel_size,
                    padding=args.padding,
                    num_groups=args.num_groups
                )
            )
        self.up_path = nn.ModuleList()
        for i in reversed(range(len(args.hidden_channels) - 1)):
            self.up_path.append(
                UNetUpBlock(
                    in_channels=args.hidden_channels[i + 1],
                    out_channels=args.hidden_channels[i],
                    kernel_size=args.kernel_size,
                    padding=args.padding,
                    num_groups=args.num_groups
                )
            )
        self.out = nn.Conv2d(args.hidden_channels[0], args.out_channels, kernel_size=(1, 1))

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, kernel_size=(2, 2))
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-(i + 1)])
        return self.out(x)


class UNetConvLayer(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            padding: int,
            num_groups: int
    ):
        super(UNetConvLayer, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(kernel_size, kernel_size),
                padding=(padding, padding)
            ),
            nn.GroupNorm(
                num_groups=num_groups,
                num_channels=out_channels
            ) if num_groups > 0 else nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor):
        return self.block(x)


class UNetConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            padding: int,
            num_groups: int
    ):
        super(UNetConvBlock, self).__init__()
        self.conv1 = UNetConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            num_groups=num_groups
        )
        self.conv2 = UNetConvLayer(
            in_channels=in_channels + out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            num_groups=num_groups
        )

    def forward(self, x: torch.Tensor):
        return self.conv2(torch.cat([x, self.conv1(x)], dim=1))


class UNetUpBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            padding: int,
            num_groups: int
    ):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=(2, 2))
        self.conv_block = UNetConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            num_groups=num_groups
        )

    def forward(self, x: torch.Tensor, bridge: torch.Tensor):
        up = self.up(x)
        out = torch.cat([up, bridge], dim=1)
        out = self.conv_block(out)
        return out
