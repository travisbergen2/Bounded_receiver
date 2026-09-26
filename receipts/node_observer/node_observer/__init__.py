"""node_observer — quadratic-pair nodes driven by a hidden rotating world, read by a
three-block bounded observer (ESTIMATE, DETECT, COMMIT) whose commit prior starts blank
and accumulates a narrative.

Exploratory design-stage toy (v0.1). Nothing here bears on the Riemann Hypothesis or on
any market. See README.md for the design, the primitive-to-code map, and the fences.
"""
from .model import (
    PLUS, TIMES, QuadraticPair, Node, Network, WorldParams, World,
    ObserverParams, Observer, RunParams, run, gaussian_table, eisenstein_table,
)

__all__ = [
    "PLUS", "TIMES", "QuadraticPair", "Node", "Network", "WorldParams", "World",
    "ObserverParams", "Observer", "RunParams", "run", "gaussian_table", "eisenstein_table",
]
__version__ = "0.1.0"
