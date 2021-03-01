import torch
from torch import nn
from torch.nn import functional as F
from math import ceil


# 2D alternative to https://www.mathworks.com/help/signal/ref/upfirdn.html
# uses PixelShuffle operation for UpSampling instead of reshape + padding https://github.com/rosinality/stylegan2-pytorch/blob/master/op/upfirdn2d.py
# Anti Alias filter adapted from https://github.com/adobe/antialiased-cnns/blob/master/antialiased_cnns/blurpool.py

class SubSample(nn.Module):
    def __init__(
            self,
            factor: int = 2,
    ):
        super(SubSample, self).__init__()
        self.factor = factor

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        return x[:, :, ::self.factor, ::self.factor]


class ShuffleUpsample(nn.Module):
    def __init__(
            self,
            factor: int = 2,
    ):
        super(ShuffleUpsample, self).__init__()
        self.shuffle = nn.PixelShuffle(factor)
        self.factor = factor

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.view(b * c, 1, h, w)
        pad = torch.zeros(b * c, self.factor ** 2 - 1, h, w, device=x.device)
        x = self.shuffle(torch.cat([x, pad], dim=1))
        x = x.view(b, c, h * self.factor, w * self.factor)
        return x


class Filter(nn.Module):

    def __kernels(
            self,
            size: int = 4,
    ) -> torch.Tensor:
        if size == 1:
            k = torch.tensor([1., ], dtype=torch.float32)
        elif size == 2:
            k = torch.tensor([1., 1.], dtype=torch.float32)
        elif size == 3:
            k = torch.tensor([1., 2., 1.], dtype=torch.float32)
        elif size == 4:
            k = torch.tensor([1., 3., 3., 1.], dtype=torch.float32)
        elif size == 5:
            k = torch.tensor([1., 4., 6., 4., 1.], dtype=torch.float32)
        elif size == 6:
            k = torch.tensor([1., 5., 10., 10., 5., 1.], dtype=torch.float32)
        elif size == 7:
            k = torch.tensor([1., 6., 15., 20., 15., 6., 1.], dtype=torch.float32)

        return k

    def __make_kernel(
            self,
            k: torch.Tensor,
            scale: int
    ) -> torch.Tensor:
        if k.ndim == 1:
            k = k[None, :] * k[:, None]

        k /= k.sum()
        k *= scale ** 2

        return k

    def __init__(
            self,
            channels: int,
            filt_size: int = 4,
            scale: int = 1,
    ):
        super(Filter, self).__init__()
        k = self.__kernels(filt_size)
        k = self.__make_kernel(k, scale)
        self.register_buffer('filter', k[None, None, :, :].repeat(channels, 1, 1, 1))

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        b, c, h, w = x.shape
        return F.conv2d(
            x,
            self.filter,
            stride=1,
            padding=0,
            groups=c,
        )


class UpFirDn2D(nn.Module):
    def __init__(
            self,
            channels: int,
            up_factor: int = 1,
            dn_factor: int = 1,
            flt_dim: int = 4,
    ):
        super(UpFirDn2D, self).__init__()

        pad_0 = None
        pad_1 = None
        self.up = None
        self.dn = None

        if up_factor != 1:
            self.up = ShuffleUpsample(factor=up_factor)
            pad = flt_dim - up_factor
            pad_0 = (pad + 1) // 2 + up_factor - 1
            pad_1 = pad // 2
        if dn_factor != 1:
            self.dn = SubSample(factor=dn_factor)
            pad = flt_dim - dn_factor
            pad_0 = (pad + 1) // 2
            pad_1 = pad // 2

        if pad_0 is None and pad_1 is None:
            pad_0 = (flt_dim - 1) / 2
            pad_1 = (flt_dim - 1) / 2
            self.pad = [int(pad_0), ceil(pad_0), int(pad_1), ceil(pad_1)]
        else:
            self.pad = [pad_0, pad_1, pad_0, pad_1]
        self.flt = Filter(channels=channels, filt_size=flt_dim, scale=up_factor)

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        if self.up:
            x = self.up(x)
        x = F.pad(x, self.pad)
        x = self.flt(x)
        if self.dn:
            x = self.dn(x)
        return x


if __name__ == '__main__':
    print(torch.__version__)
    h, w = 256, 256
    x = torch.randn(size=(8, 3, h, w))
    n1 = UpFirDn2D(channels=3, up_factor=1, dn_factor=1, flt_dim=4)
    assert n1(x).shape == (8, 3, h, w), "failed filtering operation"
    n2 = UpFirDn2D(channels=3, up_factor=2, dn_factor=1, flt_dim=4)
    assert n2(x).shape == (8, 3, h * 2, w * 2), "failed up+flt operation"
    n3 = UpFirDn2D(channels=3, up_factor=1, dn_factor=2, flt_dim=4)
    assert n3(x).shape == (8, 3, h // 2, w // 2), "failed flt+dn operation"
    n4 = UpFirDn2D(channels=3, up_factor=2, dn_factor=2, flt_dim=4)
    assert n4(x).shape == (8, 3, h, w), "failed up+flt+dn operation"
