import torch

from .livenet import Context, RegularNeuron, DestinationNeuron
from simple_log import LOG, LOGD
from . import datasets, nets, gen_utils, optimizers, net_trainer, livenet


def test_die():
    context = Context(seed=42)
    context.module = torch.nn.Module()
    src = RegularNeuron(context, activation=torch.nn.ReLU())
    context.death_stat.dangle_neurons = 1  # src
    neuron = RegularNeuron(context, activation=torch.nn.ReLU)
    dst = DestinationNeuron(context, activation=None)
    src.connect_to(neuron)
    neuron.connect_to(dst)
    assert len(src.axons) == 1
    dst.dendrites[0].die()
    assert len(src.axons) == 0
    assert context.death_stat.dangle_neurons == 1  # dst


def test_system_die_all():
    # core.simple_log.level = core.simple_log.LogLevel.DEBUG
    train_x, train_y = datasets.get_odd()
    network = nets.create_livenet_odd_2()
    network.context.alpha_l1 = 1.0  # big alpha will lead to quick death, even with big 'b'
    batch_iterator = gen_utils.batch_iterator(train_x, train_y, batch_size=len(train_x))
    criterion = nets.criterion_n
    optimizer = optimizers.LiveNetOptimizer(network, lr=0.02)
    trainer = net_trainer.Trainer(network, batch_iterator, criterion, optimizer, epoch_size=100)
    trainer.step(401)
    assert network.context.death_stat.dangle_neurons == 2
    assert len(network.inputs[0].axons) == 0
    assert len(network.outputs[0].dendrites) == 0
    assert len(network.outputs[1].dendrites) == 0


def test_system():
    # core.simple_log.level = core.simple_log.LogLevel.DEBUG
    train_x, train_y = datasets.get_odd()
    context = livenet.Context()
    context.liveness_die_after_n_sign_changes = 2
    context.alpha_l1 = 0.01
    network = nets.create_livenet_odd_2(context)
    batch_iterator = gen_utils.batch_iterator(train_x, train_y, batch_size=len(train_x))
    criterion = nets.criterion_n
    optimizer = optimizers.LiveNetOptimizer(network, lr=0.05)
    trainer = net_trainer.Trainer(network, batch_iterator, criterion, optimizer, epoch_size=100)
    trainer.step(401)
    assert len(trainer.history[0]["params"]) > len(trainer.history[-1]["params"])  # some stuff must be dead
    assert trainer.history[0]["loss"] > 0.04
    assert trainer.history[-1]["loss"] < 0.02
    assert trainer.history[-1]["loss_reg"] > 0.0
