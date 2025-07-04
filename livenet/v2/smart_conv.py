import math
import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch import compile

from livenet.v2.context2 import Context2

torch._dynamo.config.force_parameter_static_shapes = False


class SmartConv2d(nn.Module):
    def __init__(self, context: Context2, in_channels: int, out_channels: int, kernel_size: int | tuple[int, int],
                 padding: int| tuple[int, int]=0, groups=1, bias=True):
        # if input padding is None, padding will be auto-chosen to make input and output resolution the same
        super().__init__()
        self.context = context
        self.forward_type = "torch"
        assert in_channels % groups == 0, f"in_channels ({in_channels}) must be divisible by groups ({groups})"
        assert out_channels % groups == 0, f"out_channels ({out_channels}) must be divisible by groups ({groups})"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.kernel_size = nn.modules.utils._pair(kernel_size)
        kh, kw = self.kernel_size
        self.padding = nn.modules.utils._pair(padding)
        self.stats = {"count": 0}

        self.weight = nn.Parameter(
            torch.empty(groups, out_channels // groups, in_channels // groups, kh, kw)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=False)
        self._reset_parameters()
        # self.forward = compile(self.forward, dynamic=True)
        # self._forward_im2col = compile(self._forward_im2col, dynamic=True)

    def forward(self, x: Tensor) -> Tensor:
        match self.forward_type:
            case "torch":
                output = self._forward_torch(x)
            case "naive":
                output = self._forward_direct_loop(x)
            case "im2col":
                output = self._forward_im2col(x)
            case _:
                raise NotImplementedError(self.forward_type)
        self._update_observation(x)
        return output

    def _update_observation(self, x: Tensor):
        with torch.no_grad():
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
            # wg = self.weight.view(self.groups, self.out_channels // groups, self.in_channels // groups * kh * kw)
            xg = x_unfold.view(N, self.groups, self.in_channels // groups * kh * kw, out_h * out_w)

            xg_mask = (xg != 0.0).int()
            ops_amount = xg_mask.sum(dim=(0, 3))
            ops_amount = ops_amount.view(groups, 1, self.in_channels // groups, kh, kw)
            self.stats["ops_amount"] = ops_amount
            if self.stats["count"] == 0:
                self.stats["ops_amount_sum"] = ops_amount
                self._update_observation = compile(self._update_observation, dynamic=True)
            else:
                self.stats["ops_amount_sum"] += ops_amount
            self.stats["batch_size"] = N
            self.stats["out_h"] = out_h
            self.stats["out_w"] = out_w
            self.stats["count"] += 1

    def clear_stats(self):
        self.stats["count"] = 0

    def get_stats_str(self):
        kh, kw = self.kernel_size
        shape = (self.stats["count"], self.stats["batch_size"], self.groups, self.in_channels // self.groups, kh, kw, self.stats["out_h"], self.stats["out_w"])
        sum_ = self.stats["ops_amount_sum"].sum().cpu().item()
        ratio = sum_ / math.prod(shape)
        return f"shape={shape[2:]}, non_zero_ratio={round(100 * ratio)}%"

    def _forward_torch(self, x: Tensor) -> Tensor:
        w = self.weight.view(self.out_channels, self.in_channels // self.groups, *self.kernel_size)
        output = F.conv2d(x, w, self.bias, 1, self.padding, groups=self.groups)
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
        # if self.bias is not None:
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
        out_unfold += self.bias[:, None]
        out = out_unfold.view(N, self.out_channels, out_h, out_w)
        return out

    def internal_loss(self):
        return torch.tensor(0.0)
        w_ops = self.weight * self.stats["ops_amount"] / self.stats["batch_size"]
        sum_ = w_ops.abs().sum()
        l1 = self.context.regularization_l1
        loss = l1 * sum_
        return loss

    def _reset_parameters(self) -> None:
       kh, kw = self.kernel_size
       fan_in = (self.in_channels // self.groups) * kh * kw
       bound = math.sqrt(6.0 / fan_in)
       with torch.no_grad():
           self.weight.uniform_(-bound, bound)
           if self.bias.requires_grad:
               self.bias.uniform_(-bound / math.sqrt(6), bound / math.sqrt(6))
                
                
def pytest_configure(config):
    torch.manual_seed(0)

@pytest.fixture(params=[
    {'in_channels': 8, 'out_channels': 6, 'kernel_size': 3, 'padding': 1, 'groups': 2, 'batch_size': 7, 'H': 5, 'W': 4},
    {'in_channels': 8, 'out_channels': 6, 'kernel_size': 3, 'padding': 1, 'groups': 1, 'batch_size': 1, 'H': 5, 'W': 5},
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

    context = Context2()
    conv_ref = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, groups=groups)
    conv_custom = SmartConv2d(context, in_channels, out_channels, kernel_size=kernel_size, padding=padding, groups=groups)
    conv_custom.forward_type = "torch"

    with torch.no_grad():
        conv_custom.weight.view(-1)[:] = conv_ref.weight.view(-1)
        if conv_ref.bias is not None:
            conv_custom.bias.copy_(conv_ref.bias)

    x = torch.randn(batch_size, in_channels, H, W)
    return conv_ref, conv_custom, x

def test_smart_conv(conv_pairs_and_input):
    conv_ref, conv_custom, x = conv_pairs_and_input
    out_ref = conv_ref(x)
    conv_custom.forward_type = "torch"
    out_custom_torch = conv_custom(x)
    conv_custom.forward_type = "naive"
    out_custom_naive = conv_custom(x)
    conv_custom.forward_type = "im2col"
    out_custom_im2col = conv_custom(x)
    atol = 1e-6
    max_diff = (out_ref - out_custom_torch).abs().max().cpu().item()
    assert max_diff < atol, f"Max diff {max_diff} exceeds tolerance {atol}"
    max_diff = (out_ref - out_custom_naive).abs().max().cpu().item()
    assert max_diff < atol, f"Max diff {max_diff} exceeds tolerance {atol}"
    max_diff = (out_ref - out_custom_im2col).abs().max().cpu().item()
    assert max_diff < atol, f"Max diff {max_diff} exceeds tolerance {atol}"
