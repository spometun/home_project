import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import importlib
from . import core
import typing
importlib.reload(core)
from simple_log import LOG
import math
from matplotlib import pyplot as plt
import pytest


def test_dev():
    # simple_log.level = simple_log.LogLevel.DEBUG
    train_x, train_y = core.datasets.get_odd()
    network = core.nets.create_livenet_odd_2()
    network.context.alpha_l1 = 0.01
    batch_iterator = core.gen_utils.batch_iterator(train_x, train_y, batch_size=len(train_x))
    criterion = core.nets.criterion_n
    optimizer = core.optimizers.LiveNetOptimizer(network, lr=0.02)
    trainer = core.net_trainer.NetTrainer(network, batch_iterator, criterion, optimizer, epoch_size=500)
    trainer.step(3001)
