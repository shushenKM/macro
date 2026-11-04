"""Reuse the existing model architectures and expose a common logits interface.

The selected sibling project is activated before importing any of its modules.
Model instances are fresh; pretrained paths for video are resolved by the macro
caller. No source model files, weights, or caches are written by this module.
"""
from copy import deepcopy
from macro_tools.paths import resolve_project_path
import torch
from torch import nn
from macro_tools.sources import activate_source


class LogitsModel(nn.Module):
    """Adapt signal models' (logits, features) output to the video-style logits."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        output = self.model(inputs)
        return output[0] if isinstance(output, tuple) else output


def build_model(config, num_classes):
    modality = config["modality"]
    activate_source(modality)
    points = (config.get("input") or {}).get("num_points", 128)
    parameters = config.get("model_parameters") or {}
    if modality == "imu":
        from imu.model import build_model as upstream_build
        model = upstream_build(num_classes, points, parameters, sensor_count=len(config.get("sensor_ids", ["E45A", "E477"])))
    elif modality == "semg":
        from semg.model import build_model as upstream_build
        model = upstream_build(num_classes, points, parameters, grid_count=len(config.get("grid_ids", ["grid1", "grid2"])))
    elif modality in ("rgb", "depth"):
        from vision.models import build_model as upstream_build
        effective = deepcopy(config)
        weights = (effective.get("model_parameters") or {}).get("pretrained_backbone")
        if weights:
            effective["model_parameters"]["pretrained_backbone"] = str(resolve_project_path(weights))
        model = upstream_build(effective, num_classes)
    else:
        raise ValueError("Unknown modality: {}".format(modality))
    return LogitsModel(model)
