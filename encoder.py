from helper import *


class Encoder(nn.Module):
    def __init_(self, args):
        super(Encoder, self).__init__()
        channels = [128, 128, 128, 256, 256, 256]  # three channels
        attn_resolution = [16]
        num_ress_blocks = 2  # the num of  blocks
        resolution = 256  # 分辨率
        layers = [nn.Conv2d(args.image_channels[0], 3, 1, 1)]
        for i in range(len(channels) - 1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for j in range(num_ress_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                # add attn layer in every blocks
                if resolution in attn_resolution:
                    layers.append(NonLocalBlock(in_channels))
                # down the size
                if i != len(channels) - 2:
                    layers.append(DownSampleBlock(channels[i + 1]))
                    resolution //= 2

                layers.append(ResidualBlock(channels[-1], channels[-1]))
                layers.append(NonLocalBlock(channels[-1]))
                layers.append(ResidualBlock(channels[-1], channels[-1]))
                layers.append(GroupNorm(channels[-1]))
                layers.append(Swish())
                layers.append(nn.Conv2d(channels[-1], args.latent_dim, 3, 1, 1))
                self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
