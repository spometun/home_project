import pytest
from typing import List
import abc

from life.lib.simple_log import LOG


class GraphNode:

    @abc.abstractmethod
    def get_adjacent_nodes(self) -> List["GraphNode"]: ...

    # visited function must comply one of the following:
    # A. Do not modify graph structure
    # or
    # B. Visited function may disconnect visited node (and may disconnect some or all of its children)
    # siblings of visited node must be intact
    # or
    # C. ... about creation ?
    def visit(self, function_name: str, *args):
        visited_ids = set()
        self._visit(function_name, args, visited_ids=visited_ids)
        del visited_ids

    def _visit(self, function_name: str, args: tuple, visited_ids: set):
        try:
            name = self.name
        except AttributeError:
            name = "NoName"
        if id(self) in visited_ids:
            LOG(f"{name} already visited")
            return

        LOG(f"visiting {name}")
        try:
            function = type(self).__dict__[function_name]
        except KeyError:
            function = None
        if function is not None:
            function(self, *args)

        visited_ids.add(id(self))
        # make a copy because self.get_adjacent_nodes() may change
        # (some visited nodes (but not siblings!) may be disconnected - and removed from adjacent during the visit)
        adjacent_nodes = self.get_adjacent_nodes()[:]
        for node in adjacent_nodes:
            node._visit(function_name, args, visited_ids=visited_ids)


class NodesHolder(GraphNode):
    def __init__(self, name, nodes: List[GraphNode]):
        self.nodes = nodes
        self.name = name

    def get_adjacent_nodes(self) -> List[GraphNode]:
        return self.nodes


def test_graph():
    class GL(GraphNode):
        _call_counter = 0

        def __init__(self, nodes, num):
            super().__init__()
            self.nodes = nodes
            self.num = num

        def print(self):
            print(f"{self.num} {super()._global_counter}")

        def func(self):
            GL._call_counter += 1

        def summator(self, v):
            v[0] += self.num

        def get_adjacent_nodes(self) -> List["GraphNode"]:
            return self.nodes

    n1 = GL([], 1)
    n2 = GL([], 2)
    n3 = GL([n1, n2], 3)
    n4 = GL([n1, n2], 4)
    n5 = GL([n3, n4], 5)
    n5.visit("func")
    assert GL._call_counter == 5
    n5.visit("func")
    assert GL._call_counter == 10

    v = [0]
    n5.visit("summator", v)
    assert v[0] == 15


def test_except():
    class N(GraphNode):
        def get_adjacent_nodes(self) -> List["GraphNode"]:
            return []
        def f(self):
            v = self.__dict__["ku43"]

    node = N()
    with pytest.raises(KeyError):
        node.visit("f")


def test_bug():
    class C:
        def f(self):
            LOG("f...")
            v = self.ku32
        def g(self):
            try:
                exec("self.f()")
            except AttributeError:
                pass

    c = C()
    c.g()
