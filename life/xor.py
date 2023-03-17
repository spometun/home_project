from typing import List, Self
import numpy as np
import pytest
import abc
import torch
import torch.nn as nn
import random
import pickle

import lib


def test_dev():
    print("O merciful God")


class Visitable:
    def visit(self, func_name: str):
        try:
            exec(f"self.{func_name}()")
        except AttributeError:
            pass


class Neuron():
    def __init__(self, module: "LiveNet"):
        super().__init__()
        self.module = module
        self._output = None

    def compute_output(self) -> torch.Tensor:
        if self._output is None:
            self._output = self._compute_output()
        return self._output

    def clear_output(self):
        print(f"clear {type(self)} {self._output is None}")
        self._output = None

    @abc.abstractmethod
    def _compute_output(self) -> torch.Tensor: ...


# noinspection PyAbstractClass
class SourceNeuron(Neuron):
    def __init__(self, module: "LiveNet"):
        super().__init__(module)
        self.axons: List[Synapse] = []

    def connect_to(self, destination: "DestinationNeuron"):
        synapse = Synapse(self, destination)
        return synapse


# noinspection PyAbstractClass
class DestinationNeuron(Neuron):
    def __init__(self, module: "LiveNet"):
        super().__init__(module)
        self.dendrites: List[Synapse] = []

    def connect_from(self, source: SourceNeuron):
        synapse = Synapse(source, self)
        return synapse

    def clear_output(self):
        print(f"try {type(self)} {self._output is None}")
        if self._output is not None:
            for synapse in self.dendrites:
                synapse.source.clear_output()
            super().clear_output()


class DataNeuron(SourceNeuron):
    def __init__(self, module: "LiveNet"):
        super().__init__(module)

    def set_output(self, value: torch.Tensor):
        assert len(value.shape) == 1
        self._output = value

    def _compute_output(self) -> torch.Tensor:
        raise NotImplementedError("Should never be called")


class Synapse():
    def __init__(self, source: SourceNeuron, destination: DestinationNeuron):
        assert source != destination
        self.source = source
        self.destination = destination
        assert source not in (synapse.source for synapse in destination.dendrites), "Connection already exists"
        destination.dendrites.append(self)
        assert destination not in (synapse.destination for synapse in source.axons), "Connection already exists"
        source.axons.append(self)
        self.k = source.module.obtain_float_parameter("s")
        self.k.data = torch.tensor(random.random())

    def output(self):
        output = self.k * self.source.compute_output()
        return output


class RegularNeuron(SourceNeuron, DestinationNeuron):
    def __init__(self, module: "LiveNet"):
        super().__init__(module)
        self.b = module.obtain_float_parameter("n")

    def _compute_output(self) -> torch.Tensor:
        output = self.b
        for synapse in self.dendrites:
            output = output + synapse.output()
        output = torch.nn.functional.relu(output)
        return output


class LiveNet(nn.Module):
    def __init__(self, n_inputs, n_middle, n_outputs):
        super().__init__()
        self._n_params = 0
        self.inputs = [DataNeuron(self) for _ in range(n_inputs)]
        self.outputs = [RegularNeuron(self) for _ in range(n_outputs)]
        '''
        for input_ in self.inputs:
            for output in self.outputs:
                input_.connect_to(output)
                '''
        for i in range(n_middle):
            neuron = RegularNeuron(self)
            for input_ in self.inputs:
                input_.connect_to(neuron)
            for output in self.outputs:
                output.connect_from(neuron)
        self.p = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        assert len(x.shape) == 2, "Invalid input shape"
        assert x.shape[1] == len(self.inputs), "Invalid input dimension"
        for output in self.outputs:
            output.clear_output()
        for i in range(x.shape[1]):
            self.inputs[i].set_output(x[:, i])
        for output in self.outputs:
            output.compute_output()

    def obtain_float_parameter(self, name_prefix: str) -> nn.Parameter:
        name = f"{name_prefix}{self._n_params}"
        self._n_params += 1
        param = nn.Parameter(torch.tensor(0.0))
        self.register_parameter(name, param)
        return param


if __name__ == "__main__":
    lib.utils.set_seed()
    x = torch.tensor([[0., 0], [0, 1], [1, 0], [1, 1]])
    net = LiveNet(2, 1, 2)
    net.forward(x)
    net.forward(x)
    s = pickle.dumps(net.p)
    p = [p for p in net.named_parameters()]

    print("Here")
    # net = LiveNet(2, 1)
