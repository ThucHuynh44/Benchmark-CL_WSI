"""
FEATHER model helpers for MergeSlide.

This module is intentionally separate from the TITAN wrappers.  FEATHER uses
CONCH v1.5 patch features directly and does not depend on TITAN's
``vision_encoder`` or text prompt APIs.
"""

import importlib
import inspect
import os
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn


DEFAULT_FEATHER_MODEL_NAME = "abmil.base.conch_v15.pc108-24k"
DEFAULT_FEATURE_DIM = 768


def prepare_hf_token_env() -> Optional[str]:
    """Expose HF_TOKEN under common Hugging Face env names without persisting it."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
        os.environ.setdefault("HF_HUB_TOKEN", token)
    return token


def _load_from_spec(spec: str) -> Callable[..., nn.Module]:
    if ":" not in spec:
        raise ImportError(
            "FEATHER_CREATE_MODEL must use 'module.submodule:function_name' format."
        )
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    create_model = getattr(module, function_name)
    if not callable(create_model):
        raise ImportError(f"{spec} is not callable.")
    return create_model


def import_create_model() -> Callable[..., nn.Module]:
    """Import MIL-Lab's create_model with a small set of known fallbacks."""
    custom_spec = os.environ.get("FEATHER_CREATE_MODEL")
    if custom_spec:
        return _load_from_spec(custom_spec)

    candidates = [
        ("mil_lab", "create_model"),
        ("mil_lab.models", "create_model"),
        ("mil_lab.models.builder", "create_model"),
        ("mil_lab.models.registry", "create_model"),
        ("millab", "create_model"),
        ("millab.models", "create_model"),
        ("mil", "create_model"),
        ("mil.models", "create_model"),
        ("mil_models", "create_model"),
        ("mil_models.models", "create_model"),
    ]
    errors = []
    for module_name, function_name in candidates:
        try:
            module = importlib.import_module(module_name)
            create_model = getattr(module, function_name)
            if callable(create_model):
                return create_model
        except Exception as exc:
            errors.append(f"{module_name}.{function_name}: {type(exc).__name__}: {exc}")

    raise ImportError(
        "Could not import MIL-Lab create_model. Install MIL-Lab, or set "
        "FEATHER_CREATE_MODEL='module.path:create_model'. Tried:\n  "
        + "\n  ".join(errors)
    )


def _signature_accepts(signature: inspect.Signature, name: str) -> bool:
    return name in signature.parameters or any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in signature.parameters.values()
    )


def create_feather_model(
    model_name: str = DEFAULT_FEATHER_MODEL_NAME,
    *,
    num_classes: int,
    from_pretrained: bool = True,
    **extra_kwargs: Any,
) -> nn.Module:
    """Create a FEATHER MIL-Lab model."""
    prepare_hf_token_env()
    create_model = import_create_model()
    kwargs: Dict[str, Any] = {"num_classes": num_classes}

    try:
        signature = inspect.signature(create_model)
    except (TypeError, ValueError):
        signature = None

    if signature is None or _signature_accepts(signature, "from_pretrained"):
        kwargs["from_pretrained"] = from_pretrained
    elif _signature_accepts(signature, "pretrained"):
        kwargs["pretrained"] = from_pretrained

    if signature is None:
        kwargs.update(extra_kwargs)
    else:
        for key, value in extra_kwargs.items():
            if _signature_accepts(signature, key):
                kwargs[key] = value

    try:
        return create_model(model_name, **kwargs)
    except TypeError:
        # Some registries accept only the model name and task kwargs.
        kwargs.pop("from_pretrained", None)
        kwargs.pop("pretrained", None)
        return create_model(model_name, **kwargs)


def sample_patch_bag(
    features: torch.Tensor,
    coords: Optional[torch.Tensor],
    k: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Sample a WSI feature bag before moving it to GPU."""
    if k is None or k <= 0 or features.shape[0] <= k:
        return features, coords
    indices = torch.randperm(features.shape[0])[:k]
    sampled_coords = coords[indices] if coords is not None else None
    return features[indices], sampled_coords


def freeze_feather_backbone(model: nn.Module) -> Tuple[int, int]:
    """Freeze likely backbone parameters while keeping classifier-like heads trainable."""
    head_tokens = ("head", "classifier", "classif", "fc", "logit", "output")
    frozen, trainable = 0, 0
    for name, param in model.named_parameters():
        keep_trainable = any(token in name.lower() for token in head_tokens)
        param.requires_grad = keep_trainable
        if keep_trainable:
            trainable += param.numel()
        else:
            frozen += param.numel()
    return frozen, trainable


class FeatherMILWrapper(nn.Module):
    """Adapter from MergeSlide bags to FEATHER/MIL-Lab logits."""

    _FORWARD_MODES = (
        "positional",
        "features_kw",
        "x_kw",
        "feats_kw",
        "h_kw",
        "dict_features",
        "dict_x",
    )

    def __init__(
        self,
        model: nn.Module,
        *,
        num_classes: Optional[int] = None,
        feature_dim: int = DEFAULT_FEATURE_DIM,
    ) -> None:
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self._forward_mode: Optional[str] = None

    def _call_model(self, mode: str, x: torch.Tensor) -> Any:
        if mode == "positional":
            return self.model(x)
        if mode == "features_kw":
            return self.model(features=x)
        if mode == "x_kw":
            return self.model(x=x)
        if mode == "feats_kw":
            return self.model(feats=x)
        if mode == "h_kw":
            return self.model(h=x)
        if mode == "dict_features":
            return self.model({"features": x})
        if mode == "dict_x":
            return self.model({"x": x})
        raise ValueError(f"Unknown FEATHER forward mode: {mode}")

    def _choose_tensor(self, tensors) -> torch.Tensor:
        if not tensors:
            raise TypeError("FEATHER output has no tensor values.")
        if self.num_classes is not None:
            for tensor in tensors:
                if tensor.dim() > 0 and tensor.shape[-1] == self.num_classes:
                    return tensor
                if tensor.numel() == self.num_classes:
                    return tensor
        return tensors[0]

    def _extract_logits(self, output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            logits = output
        elif isinstance(output, dict):
            preferred = ("logits", "logit", "scores", "score", "output", "pred", "y_hat")
            logits = None
            for key in preferred:
                value = output.get(key)
                if isinstance(value, torch.Tensor):
                    logits = value
                    break
            if logits is None:
                tensors = [value for value in output.values() if isinstance(value, torch.Tensor)]
                try:
                    logits = self._choose_tensor(tensors)
                except TypeError as exc:
                    raise TypeError(f"FEATHER output dict has no tensor values: {output.keys()}") from exc
        elif isinstance(output, (tuple, list)):
            tensors = [value for value in output if isinstance(value, torch.Tensor)]
            logits = self._choose_tensor(tensors)
        else:
            raise TypeError(f"Unsupported FEATHER output type: {type(output).__name__}")

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        elif logits.dim() > 2:
            logits = logits.reshape(-1, logits.shape[-1])

        if self.num_classes is not None and logits.shape[-1] != self.num_classes:
            if logits.numel() == self.num_classes:
                logits = logits.reshape(1, self.num_classes)
            else:
                raise RuntimeError(
                    f"FEATHER logits have shape {tuple(logits.shape)}, expected "
                    f"last dim {self.num_classes}."
                )
        return logits

    def forward(
        self,
        features: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        patch_size: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _ = coords, patch_size
        if features.dim() == 2:
            x = features.unsqueeze(0)
        elif features.dim() == 3:
            x = features
        else:
            raise ValueError(
                f"Expected FEATHER features with shape [N, D] or [B, N, D], got {tuple(features.shape)}"
            )
        if x.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected CONCH v1.5 feature dim {self.feature_dim}, got {x.shape[-1]}."
            )
        x = x.float()

        modes: Iterable[str]
        if self._forward_mode is not None:
            modes = (self._forward_mode,)
        else:
            modes = self._FORWARD_MODES

        errors = []
        for mode in modes:
            try:
                logits = self._extract_logits(self._call_model(mode, x))
                self._forward_mode = mode
                return logits
            except Exception as exc:
                errors.append(f"{mode}: {type(exc).__name__}: {exc}")
                if self._forward_mode is not None:
                    self._forward_mode = None

        raise RuntimeError(
            "Could not run FEATHER forward with any supported calling convention.\n  "
            + "\n  ".join(errors)
        )
