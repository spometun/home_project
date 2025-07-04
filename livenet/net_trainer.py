import math
import typing
from collections import deque

import numpy as np
import torch
import pandas as pd

from ai_libs.simple_log import LOG
from . import utils


def get_summary_stat(life_stat: pd.DataFrame):
    low_cut = life_stat[life_stat["type"] == "output_low_cut_ratio"]
    low_cut = low_cut["value"].mean()
    high_cut = life_stat[life_stat["type"] == "output_high_cut_ratio"]
    high_cut = high_cut["value"].mean()
    gradient = life_stat[life_stat["type"] == "gradient"]
    av_abs_grad = gradient["value"].abs().mean()
    parameter = life_stat[life_stat["type"] == "parameter"]
    max_abs_param = parameter["value"].abs().max()
    delta = life_stat[life_stat["type"] == "delta"]
    av_abs_delta = delta["value"].abs().mean()
    av_abs_output = life_stat[life_stat["type"] == "output_av"]
    av_abs_output = av_abs_output["value"].mean()
    max_abs_output = life_stat[life_stat["type"] == "output_max"]
    max_abs_output = max_abs_output["value"].mean()
    res = f"av_abs_grad {av_abs_grad:.1g} av_abs_delta: {av_abs_delta:.1g} av_abs_output {av_abs_output:.1g} max_abs_output {max_abs_output:.1g} max_abs_param {max_abs_param:.1f} hcut {high_cut:.1g} lcut {low_cut:.1f}"
    return res


class NetTrainer:
    def __init__(self, network: torch.nn.Module, data_loader,
                 criterion: typing.Callable, optimizer, adaptive_lr=True):
        self.network = network
        self.data_loader = data_loader
        self.data_iter = iter(data_loader)
        self.criterion = criterion
        self.optimizer = optimizer
        self.history = []
        self.counter = 0
        self.counter_good = 0
        self.loss_criterion = 0.0
        self.loss_network = 0.0
        self.adaptive_lr = adaptive_lr
        self.adaptive_lr_increase_step = 1.005
        self.adaptive_lr_decrease_step = 2
        self.adaptive_lr_max_lr = 0.01
        self.adaptive_lr_min_lr = 1e-5
        self.last_epoch_tick = self.network.context.tick
        self.last_epoch_all_loss = math.inf
        self._n_loss_increases = 0
        self.loss_history = deque(maxlen=6)
        self._need_to_stop = False
        self.clear_life_stat = True

    def step(self, n_steps=1):
        self._need_to_stop = False
        for _ in range(n_steps):
            if self._need_to_stop:
                LOG("stopped")
                break
            self._step()

    def _step(self):
        try:
            data, labels = next(self.data_iter)
        except StopIteration:
            self._on_epoch()
            self.data_iter = iter(self.data_loader)
            data, labels = next(self.data_iter)
        self.network.context.tick += 1

        device = next(self.network.parameters()).device
        data = data.to(device)
        labels = labels.to(device)
        pred = self.network.forward(data)

        loss = self.criterion(pred, labels)
        self.loss_criterion += loss.detach().item()
        loss_network = self.network.internal_loss()
        self.loss_network += loss_network.detach().item()
        all_loss = loss + loss_network

        self.optimizer.zero_grad()
        all_loss.backward()
        self.optimizer.step()

        if self.adaptive_lr:
            pass
            # self._adjust_lr(data, labels, all_loss)

        self.counter += 1

    def _on_epoch(self):
        params = utils.get_parameters_dict(self.network)
        grads = utils.get_gradients_dict(self.network)
        epoch_loss_criterion = self.loss_criterion
        epoch_loss_network = self.loss_network
        # good_ratio = self.counter_good / self.epoch_size
        # if self.counter != 0:
        tick = self.network.context.tick
        epoch_size = tick - self.last_epoch_tick
        epoch_loss_criterion /= epoch_size
        epoch_loss_network /= epoch_size
        self.history.append({"params": params,
                             "loss": epoch_loss_criterion,
                             "loss_reg": epoch_loss_network})
        msg = f"{tick}"
        msg += f" {epoch_loss_criterion:.3f}+{epoch_loss_network:.3f}reg"
        msg += f" params={len(params)}"
        if self.adaptive_lr:
            all_loss = epoch_loss_criterion + epoch_loss_network
            self.loss_history.append(all_loss)
            n = self.loss_history.maxlen
            if len(self.loss_history) == n:
                h = list(self.loss_history)
                past = min(h[:n//2])
                cur = min(h[n//2:])
                if cur >= past:
                    self.optimizer.learning_rate /= 2
                    self.loss_history.clear()
            msg += f" lr={self.optimizer.learning_rate:.5f}"
            self.last_epoch_all_loss = all_loss
            if self.optimizer.learning_rate < self.adaptive_lr_min_lr:
                self._need_to_stop = True

        LOG(msg)
        m = self.network.get_stats_strs(clear=True)
        LOG(m)
        self.loss_criterion = 0.0
        self.loss_network = 0.0
        self.counter_good = 0
        self.last_epoch_tick = tick


