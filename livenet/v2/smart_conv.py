import math
import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SmartConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple[int, int],
                 padding: int| tuple[int, int]=0, groups=1, bias=True):
        # if input padding is None, padding will be auto-chosen to make input and output resolution the same
        super().__init__()
        assert in_channels % groups == 0, f"in_channels ({in_channels}) must be divisible by groups ({groups})"
        assert out_channels % groups == 0, f"out_channels ({out_channels}) must be divisible by groups ({groups})"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.kernel_size = nn.modules.utils._pair(kernel_size)
        kh, kw = self.kernel_size
        self.padding = nn.modules.utils._pair(padding)

        self.weight = nn.Parameter(
            torch.empty(groups, out_channels // groups, in_channels // groups, kh, kw)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        self._reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        # output = self._forward_im2col(x)
        output = self._forward_direct_loop(x)
        return output

    def _forward_direct_loop(self, x: Tensor) -> Tensor:
        N, C_in, H, W = x.shape
        assert C_in == self.in_channels, f"Actual data input channels ({C_in}) does not match conv input channels ({self.in_channels})"
        kh, kw = self.kernel_size
        ph, pw = self.padding
        groups = self.groups
        out_h = H + 2 * ph - kh + 1
        out_w = W + 2 * pw - kw + 1
        wg = self.weight
        len_g = wg.shape[1]
        xg = x.view(N, groups, C_in // groups, H, W)
        out = torch.empty(N, self.out_channels, out_h, out_w, device=x.device)
        for g in range(groups):
            for h0 in range(-ph, -ph + out_h):
                for w0 in range(-pw, -pw + out_w):
                    h1 = h0 + kh
                    w1 = w0 + kw
                    dh0 = max(0, -h0)
                    dw0 = max(0, -w0)
                    dh1 = max(h1 - H, 0)
                    dw1 = max(w1 - W, 0)
                    feature_x = xg[:, g, :, h0 + dh0:h1 - dh1, w0 + dw0:w1 - dw1]
                    w = wg[g, :, :, dh0:kh-dh1, dw0:kw-dw1]
                    feature_x_ = feature_x.reshape(N, -1)
                    w_ = w.reshape(len_g, -1)
                    out_hw = feature_x_ @ w_.T
                    out[:, g*len_g:(g+1)*len_g, h0+ph, w0+pw] = out_hw
        if self.bias is not None:
            out += self.bias[:, None, None]
        return out


    def _forward_im2col(self, x: Tensor) -> Tensor:
        N, C_in, H, W = x.shape
        assert C_in == self.in_channels, f"Actual data input channels ({C_in}) does not match conv input channels ({self.in_channels})"
        kh, kw = self.kernel_size
        ph, pw = self.padding
        groups = self.groups
        out_h = H + 2 * ph - kh + 1
        out_w = W + 2 * pw - kw + 1
        x_unfold = F.unfold(x, kernel_size=(kh, kw), padding=(ph, pw))
        expected_unfolded_shape = (N, self.in_channels * kh * kw, out_h * out_w)
        assert x_unfold.shape == expected_unfolded_shape, f"Internal error. Unfolded data shape {x_unfold.shape} while expected {expected_unfolded_shape}"
        wg = self.weight.view(self.groups, self.out_channels // groups, self.in_channels // groups * kh * kw)
        xg = x_unfold.view(N, self.groups, self.in_channels // groups * kh * kw, out_h * out_w)
        out_groups = []
        for g in range(groups):
            outg = wg[g] @ xg[:, g, :, :]
            out_groups.append(outg)
        out_unfold = torch.cat(out_groups, dim=1)
        if self.bias is not None:
                    out_unfold += self.bias[:, None]
        out = out_unfold.view(N, self.out_channels, out_h, out_w)
        return out

    def _reset_parameters(self) -> None:
       kh, kw = self.kernel_size
       fan_in = (self.in_channels // self.groups) * kh * kw
       bound = math.sqrt(6.0 / fan_in)
       with torch.no_grad():
           self.weight.uniform_(-bound, bound)
           if self.bias is not None:
                    self.bias.zero_()
                
                
def pytest_configure(config):
    torch.manual_seed(0)

@pytest.fixture(params=[
    {'in_channels': 8, 'out_channels': 6, 'kernel_size': 3, 'padding': 1, 'groups': 1, 'batch_size': 1, 'H': 5, 'W': 5},
    {'in_channels': 8, 'out_channels': 6, 'kernel_size': 3, 'padding': 1, 'groups': 2, 'batch_size': 2, 'H': 5, 'W': 4},
    {'in_channels': 8, 'out_channels': 6, 'kernel_size': (3, 1), 'padding': (1, 0), 'groups': 2, 'batch_size': 7, 'H': 8, 'W': 5},
    {'in_channels': 8, 'out_channels': 16, 'kernel_size': (3, 5), 'padding': (0, 0), 'groups': 8, 'batch_size': 7, 'H': 11, 'W': 5},
])
def conv_pairs_and_input(request):
    params = request.param
    in_channels = params['in_channels']
    out_channels = params['out_channels']
    kernel_size = params['kernel_size']
    padding = params['padding']
    groups = params['groups']
    batch_size = params['batch_size']
    H = params['H']
    W = params['W']

    conv_ref = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, groups=groups)
    conv_custom = SmartConv(in_channels, out_channels, kernel_size=kernel_size, padding=padding, groups=groups)

    with torch.no_grad():
        conv_custom.weight.view(-1)[:] = conv_ref.weight.view(-1)
        if conv_ref.bias is not None:
            conv_custom.bias.copy_(conv_ref.bias)

    x = torch.randn(batch_size, in_channels, H, W)
    return conv_ref, conv_custom, x

def test_smart_conv(conv_pairs_and_input):
    conv_ref, conv_custom, x = conv_pairs_and_input
    out_ref = conv_ref(x)
    out_custom = conv_custom(x)
    max_diff = (out_ref - out_custom).abs().max().cpu().item()
    atol = 1e-6
    assert max_diff < atol, f"Max diff {max_diff} exceeds tolerance {atol}"