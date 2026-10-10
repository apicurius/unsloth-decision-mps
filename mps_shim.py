"""Import Unsloth's PyTorch decision-model code on Apple Silicon.

unsloth/__init__.py routes Macs to its MLX backend, where FastDecisionModel is not
supported, and its GPU path only accepts NVIDIA, AMD and Intel GPUs. This module
skips that file: it registers bare `unsloth` packages that point at the installed
sources, stubs the few GPU-only helpers decision.py imports, and then imports
FastDecisionModel and DecisionTrainer directly.
"""

import importlib.util
import pathlib
import sys
import types

UNSLOTH_DIR = pathlib.Path(importlib.util.find_spec("unsloth").origin).parent


def _register_package(name, path):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
    return module


def _register_module(name, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    sys.modules[name] = module
    return module


def _mark_full_finetuning(model, full_finetuning=False):
    # Must honour the flag: get_peft_model skips LoRA when it is True.
    model._unsloth_full_finetuning = bool(full_finetuning)
    return model


_register_package("unsloth", UNSLOTH_DIR)
_register_package("unsloth.models", UNSLOTH_DIR / "models")
_register_package("unsloth._vendor", UNSLOTH_DIR / "_vendor")

_register_module(
    "unsloth.models._utils",
    _mark_full_finetuning=_mark_full_finetuning,
    apply_unsloth_gradient_checkpointing=lambda mode, *args, **kwargs: bool(mode),
    is_bfloat16_supported=lambda: False,
    # Only needed for transformers 5.0 - 5.5.
    patch_gradient_accumulation_fix=lambda *args, **kwargs: None,
)
_register_module(
    "unsloth.models.loader_utils",
    is_distributed=lambda: False,
    prepare_device_map=lambda *args, **kwargs: (None, None),
)

from unsloth.models.decision import DecisionTrainer, FastDecisionModel

__all__ = ["DecisionTrainer", "FastDecisionModel"]
