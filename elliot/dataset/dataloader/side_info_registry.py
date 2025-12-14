"""
Side information registry and helpers.

This module introduces a small registry to declare the capabilities and
requirements of side-information loaders. It also centralizes alignment
policies and materialization hints for large feature sets.

Capabilities capture what a loader provides (e.g., item attributes), while
alignment policies express how to handle missing users/items relative to
the training set (drop/pad/impute). Materialization hints describe whether
the loader prefers lazy loading, full in-memory, or memory-mapped data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class AlignmentMode(str, Enum):
    DROP = "drop"      # intersect with train (current behavior)
    PAD = "pad"        # add UNK/zero rows for missing users/items
    IMPUTE = "impute"  # fill missing with statistics/learned defaults


class Materialization(str, Enum):
    LAZY = "lazy"
    MEMORY = "memory"
    MMAP = "mmap"


@dataclass
class SideInfoDescriptor:
    name: str
    provides: str  # e.g., "item_features", "user_features", "kg_edges"
    format: str    # e.g., "sparse", "dense", "graph"
    dims: Optional[int] = None
    alignment: AlignmentMode = AlignmentMode.DROP
    materialization: Materialization = Materialization.MEMORY
    requires_alignment: bool = True
    notes: Dict[str, str] = field(default_factory=dict)


class SideInfoRegistry:
    def __init__(self):
        self._registry: Dict[str, SideInfoDescriptor] = {}

    def register(self, key: str, desc: SideInfoDescriptor) -> None:
        self._registry[key] = desc

    def get(self, key: str) -> Optional[SideInfoDescriptor]:
        return self._registry.get(key)

    def all(self) -> Dict[str, SideInfoDescriptor]:
        return dict(self._registry)


# Default registry instance to be populated by loaders
side_info_registry = SideInfoRegistry()


def register_default_side_sources():
    """
    Populate the registry with known loaders and sensible defaults.
    Loaders can also call `side_info_registry.register` directly.
    """
    side_info_registry.register(
        "ItemAttributes",
        SideInfoDescriptor(
            name="ItemAttributes",
            provides="item_features",
            format="sparse",
            alignment=AlignmentMode.DROP,
            materialization=Materialization.MEMORY,
        ),
    )
    side_info_registry.register(
        "VisualAttribute",
        SideInfoDescriptor(
            name="VisualAttribute",
            provides="item_features",
            format="dense",
            dims=None,
            alignment=AlignmentMode.PAD,
            materialization=Materialization.MMAP,
        ),
    )
    side_info_registry.register(
        "TextualAttribute",
        SideInfoDescriptor(
            name="TextualAttribute",
            provides="item_features",
            format="sparse",
            alignment=AlignmentMode.PAD,
            materialization=Materialization.LAZY,
        ),
    )
    side_info_registry.register(
        "KGINTSVLoader",
        SideInfoDescriptor(
            name="KGINTSVLoader",
            provides="kg_edges",
            format="graph",
            alignment=AlignmentMode.DROP,
            materialization=Materialization.MMAP,
        ),
    )


# Initialize defaults on import
register_default_side_sources()
