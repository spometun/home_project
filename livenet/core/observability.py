from typing import Any

import torch

from ai_libs.simple_log import LOG, LOGD


class TopologyStat:
    def __init__(self):
        self.dangle = set()
        self.useless = set()

    def on_dangle_neuron(self, dangle: "DestinationNeuron"):
        LOG(f"{dangle.name} became dangle")
        assert dangle not in self.dangle
        self.dangle.add(dangle)

    def off_dangle_neuron(self, dangle: "DestinationNeuron"):
        LOGD(f"{dangle.name} is not dangle any more")
        self.dangle.remove(dangle)

    def on_useless_neuron(self, useless: "SourceNeuron"):
        LOG(f"{useless.name} became useless")
        assert useless not in self.useless
        self.useless.add(useless)

    def off_useless_neuron(self, useless: "SourceNeuron"):
        LOGD(f"{useless.name} is not useless any more")
        self.useless.remove(useless)

    def get_stat(self) -> dict:
        stat = {
            "dangle": {"RegularNeuron": 0, "DestinationNeuron": 0},
            "useless": {"RegularNeuron": 0, "InputNeuron": 0}
        }
        for neuron in self.dangle:
            class_name = neuron.__class__.__name__
            stat["dangle"][class_name] += 1
        for neuron in self.useless:
            class_name = neuron.__class__.__name__
            stat["useless"][class_name] += 1
        return stat


class LifeStat:
    def __init__(self):
        self.data: dict[str, list[dict[str, Any]]] = dict()
        self.reset()

    def reset(self):
        self.data = dict()

    def add_entry(self, name: str, data: dict[str, Any]):
        if name not in self.data.keys():
            self.data[name] = []
        self.data[name].append(data)


class LifeStatContributor:
    def add_life_stat_entry(self, _type: str, tensor: torch.Tensor):
        self.context: "Context"
        self.name: str
        value = float(tensor.detach().numpy())
        assert type(value) == float, f"Invalid input type {type(value)}"
        entry = {
            "type": _type,
            "value": value,
            "name": self.name,
            "class_name": self.__class__.__name__,
            "tick": self.context.tick
        }
        self.context.life_stat.append(entry)
