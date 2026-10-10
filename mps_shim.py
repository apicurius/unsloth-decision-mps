
"""Load Unsloth's pure-PyTorch decision-model code without unsloth/__init__.py (which refuses Apple Silicon)."""
import sys, types, importlib.util, pathlib, torch
SP = pathlib.Path(importlib.util.find_spec("unsloth").origin).parent
def _pkg(name, path):
    m = types.ModuleType(name); m.__path__ = [str(path)]; sys.modules[name] = m; return m
_pkg("unsloth", SP); _pkg("unsloth.models", SP / "models"); _pkg("unsloth._vendor", SP / "_vendor")
u = types.ModuleType("unsloth.models._utils")
u._mark_full_finetuning = lambda model, full_finetuning=False: (setattr(model, "_unsloth_full_finetuning", bool(full_finetuning)), model)[1]
u.apply_unsloth_gradient_checkpointing = lambda mode, *a, **k: True if mode else False
u.is_bfloat16_supported = lambda: False
u.patch_gradient_accumulation_fix = lambda *a, **k: None   # only needed for transformers 5.0-5.5
sys.modules[u.__name__] = u
lu = types.ModuleType("unsloth.models.loader_utils")
lu.is_distributed = lambda: False
lu.prepare_device_map = lambda *a, **k: (None, None)
sys.modules[lu.__name__] = lu
import unsloth.models.decision as decision
from unsloth.models.decision import FastDecisionModel, DecisionTrainer
