"""Compatibility imports for checkpoint and inference helpers.

New code should import from ``training.checkpoints`` and
``training.inference`` directly.
"""

from __future__ import annotations

from training.checkpoints import ModelArtifacts, load_checkpoint, save_checkpoint, save_training_summary
from training.inference import default_inference_device, forward_predict, windowed_predict

__all__ = [
    "ModelArtifacts",
    "default_inference_device",
    "forward_predict",
    "load_checkpoint",
    "save_checkpoint",
    "save_training_summary",
    "windowed_predict",
]
