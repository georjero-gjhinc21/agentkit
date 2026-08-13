"""
agentkit.prebuilt
=================

Ready-made agent architectures assembled from the core primitives.

These are *starting points*, not black boxes. Each one is short enough to read
in a sitting, and the intended workflow when your needs diverge is to copy the
file and edit the graph — not to add another keyword argument here. A prebuilt
that grows twenty flags has become a framework of its own, which is the thing
this package is trying to avoid.
"""

from .react import create_agent
from .supervisor import create_supervisor

__all__ = ["create_agent", "create_supervisor"]
