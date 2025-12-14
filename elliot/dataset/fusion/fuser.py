"""
Feature fusion utilities for CB/Hybrid models.

This module provides a simple adapter that can merge multiple side-information
sources into unified tensors/maps, with basic support for padding/UNK rows when
side sources request PAD alignment. It does not perform heavy computation; it
aggregates existing side namespaces.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import scipy.sparse as sp


class FeatureFuser:
    def __init__(self, side_information) -> None:
        self.side_information = side_information

    def fuse_item_features(
        self,
        sources: Optional[List[str]] = None,
        pad_to: Optional[int] = None,
    ) -> Tuple[Optional[sp.csr_matrix], Dict[str, int]]:
        """
        Fuse item-side features from the requested sources.

        - Sparse features are horizontally stacked.
        - Dense features are concatenated column-wise.
        - For PAD alignment modes, missing items get zero/UNK rows.

        Returns:
            (fused_features, source_offsets)
                fused_features: sparse matrix (n_items x total_dims) or None
                source_offsets: starting column index per source
        """
        if not self.side_information:
            return None, {}

        matrices = []
        offsets: Dict[str, int] = {}
        cursor = 0

        for name, side_ns in self.side_information.__dict__.items():
            if sources and name not in sources:
                continue
            feat_map = getattr(side_ns, "feature_map", None)
            dense = getattr(side_ns, "features_matrix", None)
            alignment = getattr(side_ns, "alignment_mode", None)
            n_items = getattr(side_ns, "num_items", None)

            if feat_map:
                rows = []
                cols = []
                data = []
                for item, feats in feat_map.items():
                    mapped_item = side_ns.item_mapping.get(item)
                    if mapped_item is None and alignment == "pad":
                        continue
                    if mapped_item is None:
                        continue
                    for f in feats:
                        rows.append(mapped_item)
                        cols.append(f)
                        data.append(1.0)
                if cols:
                    mat = sp.csr_matrix((data, (rows, cols)), shape=(n_items, max(cols) + 1))
                    matrices.append(mat)
                    offsets[name] = cursor
                    cursor += mat.shape[1]
            elif dense is not None:
                mat = sp.csr_matrix(dense)
                matrices.append(mat)
                offsets[name] = cursor
                cursor += mat.shape[1]

        if not matrices:
            return None, {}

        fused = sp.hstack(matrices).tocsr()

        if pad_to and fused.shape[0] < pad_to:
            pad_rows = pad_to - fused.shape[0]
            fused = sp.vstack([fused, sp.csr_matrix((pad_rows, fused.shape[1]))])

        return fused, offsets
