"""SiphonNet: a hydraulic reservoir computer built from self-emptying siphons."""

from .abstract import AbstractParams, AbstractSiphonNet
from .inputs import ComplementEncoder, pump_inflow, random_input_matrix
from .physical import PhysicalParams, PhysicalSiphonNet
from .recording import Recording
from .topology import Topology, random_topology

__all__ = [
    "AbstractParams",
    "AbstractSiphonNet",
    "ComplementEncoder",
    "PhysicalParams",
    "PhysicalSiphonNet",
    "Recording",
    "Topology",
    "pump_inflow",
    "random_input_matrix",
    "random_topology",
]
