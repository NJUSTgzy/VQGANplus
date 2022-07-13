import torch
import torch.nn as nn
import torch.nn.functional as F


# group normalize
class GroupNorm(nn.Module):
    def __init__(self, channels):
        super(GroupNorm, self).__init__()
        self.gn = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-1, affine=True)

    def forward(self, x):
        return self.gn(x)


# Swish activation function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


#  Residual Block used in Encoder and decoder
class ResidualBlock(nn.Module):
    def __init__(self, inPut, outPut):
        super(ResidualBlock, self).__init__()
        self.inPut = inPut
        self.outPut = outPut
        self.block = nn.Sequential(
            nn.Conv2d(inPut, outPut, 3, 1, 1),
            GroupNorm(outPut),
            Swish(),
            nn.Conv2d(outPut, outPut, 3, 1, 0)
        )

        if inPut != outPut:
            self.channel_up = nn.Conv2d(inPut, outPut, 1, 1, 0)

    def forward(self, x):
        if self.inPut != self.outPut:
            return self.channel_up(x) + self.block(x)
        else:
            return x + self.block(x)


# double the height and length
class UpSampletBlock(nn.Module):
    def __init__(self, channels):
        super(UpSampletBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0)
        return self.conv(x)


# cut the height and length to half
class DownSampleBlock(nn.Module):
    def __init__(self, channels):
        super(DownSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 2, 0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        return self.conv(x)


#     attention part
class NonLocalBlock(nn.Module):
    def __init__(self, channels):
        super(NonLocalBlock, self).__init__()
        self.channels = channels

        self.gn = GroupNorm(channels)
        self.q = nn.Conv2d(channels, channels, 1, 1, 0)
        self.v = nn.Conv2d(channels, channels, 1, 1, 0)
        self.k = nn.Conv2d(channels, channels, 1, 1, 0)
        self.proj_out = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        h_ = self.gn(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = q.shape

        q = q.reshape(b, c, h * w)
        q.permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w)

        # softmax(qK/sqrt(dk))V
        attn = torch.bmm(q, k)
        attn = attn * (int(c) ** (-0.5))
        attn = F.softmax(attn, dim=2)
        attn = attn.permute(0, 2, 1)

        A = torch.bmm(v, attn)
        A = A.reshape(b, c, h, w)

        return x + A
