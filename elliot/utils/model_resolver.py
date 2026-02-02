"""
Utility to resolve recommender model classes, including external models.
"""

import importlib
import sys

from elliot.utils.folder import path_relative


def resolve_model_class(key: str, base_namespace, here):
    if key.startswith("external."):
        spec = importlib.util.spec_from_file_location(
            "external",
            path_relative(base_namespace.external_models_path, here)
        )
        external = importlib.util.module_from_spec(spec)
        external.backend = base_namespace.backend
        sys.modules[spec.name] = external
        spec.loader.exec_module(external)
        return getattr(importlib.import_module("external"), key.split(".", 1)[1])
    return getattr(importlib.import_module("elliot.recommender"), key)
