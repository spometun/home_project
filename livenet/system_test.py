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
LOG = core.simple_log.LOG
import math
from matplotlib import pyplot as plt
import pytest


def test_bug():
    LOG("mercy and Spirit")
    pass


