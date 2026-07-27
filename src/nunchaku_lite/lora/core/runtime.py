"""Shared runtime LoRA lifecycle helpers for Nunchaku Lite adapters."""

from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path
from types import MethodType

import torch
from torch import nn

from ...linear import AWQW4A16Linear, SVDQW4A4Linear
from ...utils import load_state_dict_in_safetensors
from .layout import (
    fit_lora_tensor,
    lora_modules,
    pack_lowrank_weight,
    svdq_down_to_logical,
    svdq_up_to_logical,
)


TRANSFORMER_LORA_METHODS = (
    "load_lora",
    "load_lora_adapter",
    "set_lora_strength",
    "set_adapters",
    "reset_lora",
    "delete_adapters",
    "unload_lora",
    "enable_lora",
    "disable_lora",
    "get_list_adapters",
    "get_active_adapters",
    "fuse_lora",
    "_convert_lora_to_nunchaku",
)

PIPELINE_LORA_METHODS = (
    "load_lora_weights",
    "load_lora_adapter",
    "set_adapters",
    "delete_adapters",
    "unload_lora_weights",
    "enable_lora",
    "disable_lora",
    "get_list_adapters",
    "get_active_adapters",
    "fuse_lora",
    "unfuse_lora",
    "_pipeline_transformer",
)

RUNTIME_LORA_LABEL = "Nunchaku LoRA"


def bind_mixin_methods(instance, mixin_cls: type, method_names: tuple[str, ...]) -> None:
    """Bind selected methods from a mixin class onto one runtime instance.

    Args:
        instance: Object that should receive bound methods.
        mixin_cls: Class providing unbound method implementations.
        method_names: Names to copy from ``mixin_cls`` onto ``instance``.
    """

    for method_name in method_names:
        setattr(instance, method_name, MethodType(getattr(mixin_cls, method_name), instance))


def bind_transformer_lora_methods(
    transformer: nn.Module,
    mixin_cls: type,
) -> None:
    """Attach transformer-level LoRA runtime methods to a patched module.

    Args:
        transformer: Patched transformer that will own adapter state.
        mixin_cls: Model-specific mixin implementing conversion hooks.
    """

    bind_mixin_methods(transformer, mixin_cls, TRANSFORMER_LORA_METHODS)
    ensure_lora_runtime(transformer)


def bind_pipeline_lora_methods(
    pipeline,
    mixin_cls: type,
    *,
    component_name: str = "transformer",
) -> None:
    """Attach Diffusers-compatible LoRA methods to a pipeline object.

    Args:
        pipeline: Pipeline instance whose component should receive runtime LoRA
            operations.
        mixin_cls: Model-specific pipeline mixin implementation.
        component_name: Name of the pipeline component containing the patched
            transformer or UNet.
    """

    pipeline._nunchaku_lite_lora_component_name = component_name
    bind_mixin_methods(pipeline, mixin_cls, PIPELINE_LORA_METHODS)
    pipeline._nunchaku_lite_lora_pipeline_api_bound = True


class NunchakuLoraMixin:
    """Mixin-style method provider for quantized transformer LoRA runtime."""

    def load_lora(
        self,
        path_or_state_dict: str | Path | dict[str, torch.Tensor],
        *,
        strength: float = 1.0,
        name: str | None = None,
        replace: bool = False,
        defer_recompose: bool = False,
    ) -> str:
        """Load a runtime LoRA adapter into this patched transformer.

        Args:
            path_or_state_dict: Safetensors path or already loaded LoRA state dict.
            strength: Initial adapter scale.
            name: Optional adapter name.
            replace: Whether to clear existing adapters before loading.
            defer_recompose: When True, skip immediate recompose for batch loading.
        """

        return load_lora(
            self, path_or_state_dict, strength=strength, name=name, replace=replace, defer_recompose=defer_recompose
        )

    def load_lora_adapter(
        self,
        pretrained_model_name_or_path_or_dict,
        prefix: str = "transformer",
        hotswap: bool = False,
        **kwargs,
    ) -> str:
        """Load a Diffusers/PEFT-style adapter dict through the transformer API.

        Args:
            pretrained_model_name_or_path_or_dict: State dict passed by the
                Diffusers loader path.
            prefix: Component prefix to strip from keys before conversion.
            hotswap: Unsupported PEFT hotswap flag.
            **kwargs: Diffusers loader options; supported adapter metadata is
                consumed and unsupported runtime options are ignored.
        """

        if hotswap:
            raise NotImplementedError(
                f"nunchaku_lite {RUNTIME_LORA_LABEL} runtime does not support PEFT hotswap."
            )
        adapter_name = kwargs.pop("adapter_name", None)
        network_alphas = kwargs.pop("network_alphas", None)
        kwargs.pop("metadata", None)
        kwargs.pop("_pipeline", None)
        kwargs.pop("low_cpu_mem_usage", None)
        state_dict = strip_component_prefixes(dict(pretrained_model_name_or_path_or_dict), prefix=prefix)
        if network_alphas:
            state_dict.update(network_alphas)
        raise_if_text_encoder_lora(state_dict)
        return self.load_lora(state_dict, name=adapter_name)

    def set_lora_strength(self, strength: float = 1.0, name: str | None = None) -> None:
        """Set the scale for one loaded adapter.

        Args:
            strength: New adapter scale.
            name: Adapter name, required when multiple adapters are active.
        """

        set_lora_strength(self, strength=strength, name=name)

    def set_adapters(self, adapter_names: list[str] | str, weights=None) -> None:
        """Select active adapters and optional per-adapter weights.

        Args:
            adapter_names: Adapter name or names to enable.
            weights: Optional scalar/list/dict weights matching Diffusers API.
        """

        set_active_adapters(self, adapter_names, weights)

    def reset_lora(self, name: str | None = None) -> None:
        """Remove one adapter or all adapters and restore base low-rank state.

        Args:
            name: Adapter name to remove, or ``None`` to remove all adapters.
        """

        reset_lora(self, name=name)

    def delete_adapters(self, adapter_names: list[str] | str) -> None:
        """Delete one or more named adapters.

        Args:
            adapter_names: Adapter name or names to remove.
        """

        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]
        for adapter_name in adapter_names:
            self.reset_lora(adapter_name)

    def unload_lora(self) -> None:
        """Unload every runtime LoRA adapter from this transformer."""

        self.reset_lora()

    def enable_lora(self) -> None:
        """Enable runtime LoRA composition and reapply active adapters."""

        ensure_lora_runtime(self)
        self._nunchaku_lite_lora_enabled = True
        recompose_loras(self)

    def disable_lora(self) -> None:
        """Disable runtime LoRA composition while keeping loaded adapters."""

        ensure_lora_runtime(self)
        self._nunchaku_lite_lora_enabled = False
        recompose_loras(self)

    def get_list_adapters(self) -> list[str]:
        """Return loaded adapter names in insertion order."""

        ensure_lora_runtime(self)
        return list(self._nunchaku_lite_loras)

    def get_active_adapters(self) -> list[str]:
        """Return currently active adapter names."""

        return active_lora_names(self)

    def fuse_lora(self, *args, **kwargs) -> None:
        """Reject Diffusers fuse requests because runtime LoRAs stay separate."""

        raise NotImplementedError(
            f"nunchaku_lite {RUNTIME_LORA_LABEL} runtime keeps adapters as low-rank branches."
        )

    def _convert_lora_to_nunchaku(
        self,
        path_or_state_dict: str | Path | dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Convert model-specific LoRA input into Nunchaku Lite tensors.

        Args:
            path_or_state_dict: Safetensors path or loaded LoRA state dict.
        """

        raise NotImplementedError


class NunchakuPipelineLoraMixin:
    """Mixin-style provider for Diffusers-compatible pipeline LoRA APIs."""

    _nunchaku_lite_lora_component_name = "transformer"

    def load_lora_weights(
        self,
        pretrained_model_name_or_path_or_dict: str | dict[str, torch.Tensor],
        adapter_name: str | None = None,
        hotswap: bool = False,
        **kwargs,
    ) -> None:
        """Load LoRA weights through a Diffusers-compatible pipeline API.

        Args:
            pretrained_model_name_or_path_or_dict: Path, repo id, or state dict
                accepted by the pipeline's ``lora_state_dict`` loader.
            adapter_name: Optional runtime adapter name.
            hotswap: Unsupported PEFT hotswap flag.
            **kwargs: Extra loader options forwarded to ``lora_state_dict``.
                Pass ``defer_recompose=True`` to skip immediate recompose for
                batch loading.
        """

        if hotswap:
            raise NotImplementedError(
                f"nunchaku_lite {RUNTIME_LORA_LABEL} runtime does not support PEFT hotswap."
            )
        transformer = self._pipeline_transformer()
        ensure_lora_runtime(transformer)
        # Local files and in-memory state dicts are loaded directly through the
        # Nunchaku Lite converter, which natively handles Kohya/ai-toolkit
        # native Z-Image LoRA formats (including ``.diff``/``.diff_b`` deltas)
        # that Diffusers' ``lora_state_dict`` cannot convert. Hub repo ids still
        # fall back to the Diffusers loader so cached/remote weights resolve.
        if _is_local_lora_source(pretrained_model_name_or_path_or_dict):
            state_dict = load_lora_state_dict(pretrained_model_name_or_path_or_dict)
            raise_if_text_encoder_lora(state_dict)
            transformer.load_lora(
                state_dict, name=adapter_name, defer_recompose=kwargs.pop("defer_recompose", False)
            )
            return
        kwargs["return_lora_metadata"] = True
        result = self.lora_state_dict(
            pretrained_model_name_or_path_or_dict,
            return_alphas=True,
            **kwargs,
        )
        network_alphas = None
        if len(result) == 3:
            state_dict, network_alphas, _metadata = result
        else:
            state_dict, maybe_alphas_or_metadata = result
            if isinstance(maybe_alphas_or_metadata, dict) and all(
                isinstance(value, torch.Tensor) for value in maybe_alphas_or_metadata.values()
            ):
                network_alphas = maybe_alphas_or_metadata
        component_name = getattr(self, "_nunchaku_lite_lora_component_name", "transformer")
        raise_if_text_encoder_lora(state_dict)
        transformer_state = {
            key: value
            for key, value in state_dict.items()
            if key.startswith(
                (
                    f"{component_name}.",
                    f"base_model.model.{component_name}.",
                )
            )
        }
        if not transformer_state:
            transformer_state = state_dict
        if network_alphas:
            transformer_state = dict(transformer_state)
            transformer_state.update(network_alphas)
        transformer.load_lora(
            transformer_state, name=adapter_name, defer_recompose=kwargs.pop("defer_recompose", False)
        )

    def load_lora_adapter(
        self,
        pretrained_model_name_or_path_or_dict,
        adapter_name: str | None = None,
        hotswap: bool = False,
        **kwargs,
    ) -> None:
        """Alias ``load_lora_weights`` for PEFT/Diffusers compatibility.

        Args:
            pretrained_model_name_or_path_or_dict: LoRA source accepted by
                ``load_lora_weights``.
            adapter_name: Optional runtime adapter name.
            hotswap: Unsupported PEFT hotswap flag.
            **kwargs: Extra loader options forwarded to ``load_lora_weights``.
        """

        self.load_lora_weights(
            pretrained_model_name_or_path_or_dict,
            adapter_name=adapter_name,
            hotswap=hotswap,
            **kwargs,
        )

    def set_adapters(
        self,
        adapter_names: list[str] | str,
        adapter_weights: float | dict | list[float] | list[dict] | None = None,
    ) -> None:
        """Select active pipeline adapters and optional component weights.

        Args:
            adapter_names: Adapter name or names to activate.
            adapter_weights: Scalar/list/dict weights accepted by Diffusers APIs.
        """

        transformer = self._pipeline_transformer()
        component_name = getattr(self, "_nunchaku_lite_lora_component_name", "transformer")
        weights = transformer_adapter_weights(adapter_weights, component_name)
        transformer.set_adapters(adapter_names, weights)

    def delete_adapters(self, adapter_names: list[str] | str) -> None:
        """Delete one or more adapters from the patched pipeline component.

        Args:
            adapter_names: Adapter name or names to remove.
        """

        self._pipeline_transformer().delete_adapters(adapter_names)

    def unload_lora_weights(self, reset_to_overwritten_params: bool = False) -> None:
        """Unload all LoRA weights from the patched pipeline component.

        Args:
            reset_to_overwritten_params: Unsupported Diffusers option because
                this runtime never overwrites dense parameters.
        """

        if reset_to_overwritten_params:
            raise NotImplementedError(
                f"nunchaku_lite {RUNTIME_LORA_LABEL} runtime does not overwrite dense params."
            )
        self._pipeline_transformer().unload_lora()

    def enable_lora(self) -> None:
        """Enable LoRA composition on the patched pipeline component."""

        self._pipeline_transformer().enable_lora()

    def disable_lora(self) -> None:
        """Disable LoRA composition on the patched pipeline component."""

        self._pipeline_transformer().disable_lora()

    def get_list_adapters(self) -> dict[str, list[str]]:
        """Return loaded adapter names grouped by pipeline component."""

        component_name = getattr(self, "_nunchaku_lite_lora_component_name", "transformer")
        return {component_name: self._pipeline_transformer().get_list_adapters()}

    def get_active_adapters(self) -> list[str]:
        """Return active adapter names for the patched pipeline component."""

        return self._pipeline_transformer().get_active_adapters()

    def fuse_lora(self, *args, **kwargs) -> None:
        """Reject Diffusers fuse requests for quantized runtime LoRAs."""

        raise NotImplementedError(
            f"nunchaku_lite {RUNTIME_LORA_LABEL} runtime does not support fusing into quantized weights."
        )

    def unfuse_lora(self, *args, **kwargs) -> None:
        """Reject Diffusers unfuse requests for quantized runtime LoRAs."""

        raise NotImplementedError(
            f"nunchaku_lite {RUNTIME_LORA_LABEL} runtime does not support fusing into quantized weights."
        )

    def _pipeline_transformer(self) -> nn.Module:
        """Return the patched pipeline component with transformer LoRA APIs."""

        component_name = getattr(self, "_nunchaku_lite_lora_component_name", "transformer")
        transformer = getattr(self, component_name, None)
        if transformer is None:
            transformer_name = getattr(self, "transformer_name", component_name)
            transformer = getattr(self, transformer_name)
        if not callable(getattr(transformer, "load_lora", None)):
            raise RuntimeError(
                f"Pipeline component {component_name!r} is not bound to the nunchaku_lite "
                "transformer LoRA runtime. Patch the transformer with its adapter before "
                "binding pipeline LoRA methods."
            )
        return transformer


def load_lora(
    transformer: nn.Module,
    path_or_state_dict: str | Path | dict[str, torch.Tensor],
    *,
    strength: float = 1.0,
    name: str | None = None,
    replace: bool = False,
    defer_recompose: bool = False,
) -> str:
    """Load a LoRA into a patched transformer and return its adapter name.

    Args:
        transformer: Patched transformer with bound Nunchaku Lite LoRA methods.
        path_or_state_dict: Safetensors path or in-memory LoRA state dict.
        strength: Initial adapter scale used when composing active LoRAs.
        name: Optional adapter name. If omitted, a path stem or generated name
            is used.
        replace: Whether to remove existing adapters before loading this one.
        defer_recompose: When True, skip the immediate ``recompose_loras``
            call.  Useful for batch-loading multiple adapters: set True on all
            but the last load, then call ``recompose_loras(transformer)`` once.
            This reduces O(M·N²) recompose calls to O(N²) for M adapters (fix #2).
    """

    ensure_lora_base_state(transformer)
    if replace:
        transformer._nunchaku_lite_loras.clear()
        transformer._nunchaku_lite_active_loras.clear()

    lora_name = resolve_lora_name(transformer, path_or_state_dict, name)
    converted = transformer._convert_lora_to_nunchaku(path_or_state_dict)
    converted = {key: value.detach().cpu() for key, value in converted.items()}
    transformer._nunchaku_lite_loras[lora_name] = {"state_dict": converted, "strength": float(strength)}
    transformer._nunchaku_lite_active_loras = [
        active_name for active_name in transformer._nunchaku_lite_active_loras if active_name != lora_name
    ]
    transformer._nunchaku_lite_active_loras.append(lora_name)
    # Invalidate adapter cache — the set of loaded adapters changed (fix #5).
    _invalidate_adapter_cache(transformer)
    if not defer_recompose:
        recompose_loras(transformer)
    return lora_name


def set_lora_strength(transformer: nn.Module, strength: float = 1.0, name: str | None = None) -> None:
    """Set one adapter strength and recompose all active LoRAs.

    When all active modules are SVDQ (the common case for quantized models),
    this uses an O(1) fast path: instead of rebuilding the packed LoRA tensors,
    it only updates the per-16-rank-block ``_nunchaku_lite_lora_scales`` lists
    on each module.  The native kernel applies the new strength at GEMM time.

    For AWQ/dense modules (or when no SVDQ modules exist), it falls back to
    a full ``recompose_loras`` because those paths bake strength into the weight.

    Args:
        transformer: Patched transformer containing runtime adapter state.
        strength: New scalar multiplier for the adapter.
        name: Adapter name. Required when more than one adapter is active.
    """

    ensure_lora_runtime(transformer)
    if not transformer._nunchaku_lite_loras:
        return
    if name is None:
        active_names = active_lora_names(transformer)
        if len(active_names) != 1:
            raise ValueError("Multiple LoRAs are active; pass name=... to set one strength.")
        name = active_names[0]
    try:
        transformer._nunchaku_lite_loras[name]["strength"] = float(strength)
    except KeyError as exc:
        raise ValueError(f"No loaded LoRA named {name!r}.") from exc

    # Fast path (fix #1): if all modules are SVDQ, just update lora_scales.
    modules = transformer._nunchaku_lite_lora_module_cache
    if modules is not None:
        active_entries = active_lora_entries(transformer)
        active_names_ordered = active_lora_names(transformer)
        # Attach names for _build_lora_scales.
        for i, entry in enumerate(active_entries):
            active_entries[i] = {**entry, "name": active_names_ordered[i]}

        has_non_svdq = any(not isinstance(m, SVDQW4A4Linear) for m in modules.values())
        if not has_non_svdq:
            # O(1) fast path: update lora_scales on every SVDQ module.
            for module_name, module in modules.items():
                if not isinstance(module, SVDQW4A4Linear):
                    continue
                base_logical = transformer._nunchaku_lite_lora_base_logical
                base_rank = base_logical[f"{module_name}.down"].shape[0]
                base_blocks = math.ceil(base_rank / 16)
                # Build adapter ranks for this module from the cached adapter cache.
                adapter_cache = _ensure_adapter_cache(transformer)
                adapter_ranks: dict[str, int] = {}
                for entry in active_entries:
                    adapter_name = entry["name"]
                    adapter_tensors = adapter_cache.get(adapter_name, {})
                    if module_name in adapter_tensors:
                        adapter_ranks[adapter_name] = adapter_tensors[module_name]["down"].shape[0]
                adapter_scales = _build_lora_scales(active_entries, adapter_ranks)
                module._nunchaku_lite_lora_scales = [1.0] * base_blocks + adapter_scales
            return

    # Fallback: full recompose (AWQ/dense modules present).
    recompose_loras(transformer)


def reset_lora(transformer: nn.Module, name: str | None = None) -> None:
    """Remove adapters from a patched transformer and restore base low-rank state.

    Args:
        transformer: Patched transformer containing runtime adapter state.
        name: Optional adapter name. When omitted, all adapters are removed.
    """

    ensure_lora_runtime(transformer)
    if name is None:
        transformer._nunchaku_lite_loras.clear()
        transformer._nunchaku_lite_active_loras.clear()
    else:
        try:
            del transformer._nunchaku_lite_loras[name]
        except KeyError as exc:
            raise ValueError(f"No loaded LoRA named {name!r}.") from exc
        transformer._nunchaku_lite_active_loras = [
            active_name for active_name in transformer._nunchaku_lite_active_loras if active_name != name
        ]
    # Invalidate adapter cache — the set of loaded adapters changed (fix #5).
    _invalidate_adapter_cache(transformer)
    recompose_loras(transformer)


def load_lora_state_dict(path_or_state_dict: str | Path | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Load a LoRA state dict from memory or a safetensors file.

    Args:
        path_or_state_dict: Existing state dict or path to a safetensors file.
    """

    if isinstance(path_or_state_dict, dict):
        return dict(path_or_state_dict)
    return load_state_dict_in_safetensors(path_or_state_dict)


def _is_local_lora_source(pretrained_model_name_or_path_or_dict: str | dict[str, torch.Tensor]) -> bool:
    """Return whether a LoRA source can be loaded without the Diffusers loader.

    Local safetensors files and in-memory state dicts are read directly by the
    Nunchaku Lite converter, bypassing Diffusers' ``lora_state_dict``. That
    converter is lossy for native Kohya/ai-toolkit Z-Image LoRAs (it cannot
    consume ``.diff``/``.diff_b`` delta keys), whereas the Lite converter
    handles them natively.

    Args:
        pretrained_model_name_or_path_or_dict: Source passed to
            ``load_lora_weights``.

    Returns:
        ``True`` for in-memory dicts and existing local file paths.
    """

    if isinstance(pretrained_model_name_or_path_or_dict, dict):
        return True
    if isinstance(pretrained_model_name_or_path_or_dict, (str, Path)):
        path = Path(pretrained_model_name_or_path_or_dict)
        return path.is_file()
    return False


def strip_component_prefixes(
    state_dict: dict[str, torch.Tensor], prefix: str = "transformer"
) -> dict[str, torch.Tensor]:
    """Remove pipeline component prefixes from LoRA state-dict keys.

    Args:
        state_dict: LoRA tensors keyed by pipeline or model component names.
        prefix: Component prefix to remove, usually ``transformer``.
    """

    prefixes = (
        f"{prefix}.",
        f"base_model.model.{prefix}.",
    )
    stripped = {}
    for key, value in state_dict.items():
        new_key = key
        for current_prefix in prefixes:
            if new_key.startswith(current_prefix):
                new_key = new_key[len(current_prefix) :]
                break
        stripped[new_key] = value
    return stripped


def raise_if_text_encoder_lora(state_dict: dict[str, torch.Tensor]) -> None:
    """Reject LoRA state dicts that target text encoder modules.

    Args:
        state_dict: Candidate LoRA tensors to inspect before transformer-only
            runtime loading.
    """

    text_keys = [
        key
        for key in state_dict
        if key.startswith(("text_encoder.", "text_encoder_2.", "base_model.model.text_encoder."))
    ]
    if text_keys:
        sample = ", ".join(text_keys[:5])
        raise NotImplementedError(
            f"nunchaku_lite {RUNTIME_LORA_LABEL} runtime supports transformer LoRA weights only; "
            f"text encoder LoRA keys are not supported: {sample}"
        )


def transformer_adapter_weights(adapter_weights, component_name: str = "transformer"):
    """Extract the transformer component weights from Diffusers adapter weights.

    Args:
        adapter_weights: Scalar, list, or component-name dict accepted by
            Diffusers pipeline APIs.
        component_name: Component key to read when weights are dictionaries.
    """

    if isinstance(adapter_weights, dict):
        return adapter_weights.get(component_name)
    if isinstance(adapter_weights, list):
        return [
            weight.get(component_name) if isinstance(weight, dict) else weight
            for weight in adapter_weights
        ]
    return adapter_weights


def set_active_adapters(transformer: nn.Module, adapter_names: list[str] | str, weights=None) -> None:
    """Select active adapters and update their runtime weights.

    Args:
        transformer: Patched transformer containing loaded adapters.
        adapter_names: Adapter name or ordered list of adapter names to enable.
        weights: Optional scalar/list/dict weights matching Diffusers adapter
            API conventions.
    """

    ensure_lora_runtime(transformer)
    adapter_names = [adapter_names] if isinstance(adapter_names, str) else list(adapter_names)
    if not isinstance(weights, list):
        weights = [weights] * len(adapter_names)
    if len(adapter_names) != len(weights):
        raise ValueError(
            f"Length of adapter names {len(adapter_names)} is not equal to the length of the weights {len(weights)}."
        )

    missing = [name for name in adapter_names if name not in transformer._nunchaku_lite_loras]
    if missing:
        loaded = set(transformer._nunchaku_lite_loras)
        raise ValueError(f"Adapter name(s) {set(missing)} not in the list of present adapters: {loaded}.")

    transformer._nunchaku_lite_active_loras = list(adapter_names)
    for adapter_name, weight in zip(adapter_names, weights):
        if isinstance(weight, dict):
            weight = weight.get("transformer", 1.0)
        transformer._nunchaku_lite_loras[adapter_name]["strength"] = 1.0 if weight is None else float(weight)
    # Invalidate adapter cache — the active set (and thus cached tensors) changed.
    _invalidate_adapter_cache(transformer)
    recompose_loras(transformer)


def active_lora_names(transformer: nn.Module) -> list[str]:
    """Return currently active adapter names, respecting the global enable flag.

    Args:
        transformer: Patched transformer containing runtime adapter state.
    """

    ensure_lora_runtime(transformer)
    if not transformer._nunchaku_lite_lora_enabled:
        return []
    return [name for name in transformer._nunchaku_lite_active_loras if name in transformer._nunchaku_lite_loras]


def active_lora_entries(transformer: nn.Module) -> list[dict]:
    """Return active adapter metadata entries in composition order.

    Args:
        transformer: Patched transformer containing runtime adapter state.
    """

    return [transformer._nunchaku_lite_loras[name] for name in active_lora_names(transformer)]


def ensure_lora_runtime(transformer: nn.Module) -> None:
    """Initialize runtime LoRA bookkeeping fields if they are missing.

    Args:
        transformer: Patched transformer that may not yet have adapter state.
    """

    if not hasattr(transformer, "_nunchaku_lite_loras"):
        transformer._nunchaku_lite_loras = OrderedDict()
    if not hasattr(transformer, "_nunchaku_lite_lora_base_state"):
        transformer._nunchaku_lite_lora_base_state = None
    if not hasattr(transformer, "_nunchaku_lite_active_loras"):
        transformer._nunchaku_lite_active_loras = list(transformer._nunchaku_lite_loras)
    if not hasattr(transformer, "_nunchaku_lite_lora_enabled"):
        transformer._nunchaku_lite_lora_enabled = True
    # --- Performance caches (fix #3, #4, #5) ---
    # Cached lora_modules() scan result — avoids 3× named_modules() per recompose.
    if not hasattr(transformer, "_nunchaku_lite_lora_module_cache"):
        transformer._nunchaku_lite_lora_module_cache = None
    # Cached base state in *logical* (unpacked) form for SVDQ modules — avoids
    # re-unpacking every recompose.
    if not hasattr(transformer, "_nunchaku_lite_lora_base_logical"):
        transformer._nunchaku_lite_lora_base_logical = None
    # Cached per-adapter logical tensors, GPU-resident — avoids re-unpacking
    # from CPU state_dict every recompose.
    if not hasattr(transformer, "_nunchaku_lite_lora_adapter_cache"):
        transformer._nunchaku_lite_lora_adapter_cache = None
    # Whether the next recompose should actually rebuild tensors. When False
    # (deferred batch loading), recompose is skipped until the caller re-enables.
    if not hasattr(transformer, "_nunchaku_lite_defer_recompose"):
        transformer._nunchaku_lite_defer_recompose = False


def ensure_lora_base_state(transformer: nn.Module) -> None:
    """Snapshot base low-rank tensors before runtime LoRA composition.

    Also caches the module scan and unpacked logical forms for SVDQ modules
    so that subsequent ``recompose_loras`` calls avoid repeated tree scans
    and unpack operations.

    Args:
        transformer: Patched transformer whose SVDQ/AWQ modules should be
            restorable after adapters are disabled or unloaded.
    """

    ensure_lora_runtime(transformer)
    if transformer._nunchaku_lite_lora_base_state is not None:
        return

    # Cache the module scan (fix #3): lora_modules() does 3× named_modules()
    # calls. We cache it here so recompose never re-scans.
    modules = lora_modules(transformer)
    transformer._nunchaku_lite_lora_module_cache = modules

    base_state = {}
    base_logical = {}  # fix #4: cache unpacked SVDQ logical forms
    for name, module in modules.items():
        if isinstance(module, SVDQW4A4Linear):
            base_down = module.proj_down.detach().clone()
            base_up = module.proj_up.detach().clone()
            base_state[f"{name}.proj_down"] = base_down
            base_state[f"{name}.proj_up"] = base_up
            # Pre-unpack to logical form so recompose doesn't repeat this work.
            base_logical[f"{name}.down"] = svdq_down_to_logical(base_down, module, name)
            base_logical[f"{name}.up"] = svdq_up_to_logical(base_up, module, name)
        elif isinstance(module, AWQW4A16Linear):
            device = module.qweight.device
            dtype = module.wscales.dtype
            base_state[f"{name}.proj_down"] = torch.empty(module.in_features, 0, device=device, dtype=dtype)
            base_state[f"{name}.proj_up"] = torch.empty(module.out_features, 0, device=device, dtype=dtype)
        else:
            base_state[f"{name}.proj_down"] = torch.empty(
                module.in_features,
                0,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
            base_state[f"{name}.proj_up"] = torch.empty(
                module.out_features,
                0,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
    transformer._nunchaku_lite_lora_base_state = base_state
    transformer._nunchaku_lite_lora_base_logical = base_logical


def resolve_lora_name(
    transformer: nn.Module,
    path_or_state_dict: str | Path | dict[str, torch.Tensor],
    name: str | None,
) -> str:
    """Choose and validate the adapter name for a newly loaded LoRA.

    Args:
        transformer: Patched transformer containing existing adapter names.
        path_or_state_dict: Source used to derive a default name when needed.
        name: User-provided adapter name, or ``None`` to auto-generate one.
    """

    if name is None:
        if isinstance(path_or_state_dict, (str, Path)):
            name = Path(path_or_state_dict).stem
        else:
            name = f"lora_{len(transformer._nunchaku_lite_loras) + 1}"
    if name in transformer._nunchaku_lite_loras:
        raise ValueError(f"A LoRA named {name!r} is already active. Use replace=True or choose another name.")
    return name


def _invalidate_adapter_cache(transformer: nn.Module) -> None:
    """Clear the per-adapter logical tensor cache.

    Called when the set of loaded adapters changes (load/delete/reset) so
    that the next recompose rebuilds the cache from the current state_dict.

    Args:
        transformer: Patched transformer with runtime adapter state.
    """

    transformer._nunchaku_lite_lora_adapter_cache = None


def _ensure_adapter_cache(transformer: nn.Module) -> dict:
    """Build (or return cached) per-adapter logical GPU tensors.

    For each active adapter, pre-unpacks every SVDQ proj_down/proj_up to
    logical form and moves them to the module's device.  The cache is keyed
    by adapter name and invalidated whenever the adapter set changes.

    For AWQ/dense adapters the raw state_dict tensors are stored as-is
    (they are cheap to fit_lora_tensor at recompose time).

    Args:
        transformer: Patched transformer with runtime adapter state.

    Returns:
        Dict mapping adapter_name → {module_name → {"down": tensor, "up": tensor}}.
    """

    cache = transformer._nunchaku_lite_lora_adapter_cache
    if cache is not None:
        return cache

    modules = transformer._nunchaku_lite_lora_module_cache
    assert modules is not None, "module cache must be populated by ensure_lora_base_state"
    active_names = active_lora_names(transformer)
    cache = {}
    for adapter_name in active_names:
        entry = transformer._nunchaku_lite_loras[adapter_name]
        state_dict = entry["state_dict"]
        adapter_tensors: dict[str, dict[str, torch.Tensor]] = {}
        for name, module in modules.items():
            down_key = f"{name}.proj_down"
            up_key = f"{name}.proj_up"
            if down_key not in state_dict and up_key not in state_dict:
                continue
            if down_key not in state_dict or up_key not in state_dict:
                raise ValueError(f"Incomplete LoRA tensors for {name}.")
            if isinstance(module, SVDQW4A4Linear):
                down_logical = svdq_down_to_logical(state_dict[down_key], module, name)
                up_logical = svdq_up_to_logical(state_dict[up_key], module, name)
                if down_logical.shape[0] != up_logical.shape[1]:
                    raise ValueError(
                        f"LoRA rank mismatch for {name}: "
                        f"proj_down={tuple(down_logical.shape)}, proj_up={tuple(up_logical.shape)}"
                    )
                device = module.proj_down.device
                dtype = module.proj_down.dtype
                adapter_tensors[name] = {
                    "down": down_logical.to(device=device, dtype=dtype),
                    "up": up_logical.to(device=device, dtype=dtype),
                }
            else:
                # AWQ / dense: keep raw tensors; fit at recompose time.
                adapter_tensors[name] = {
                    "down": state_dict[down_key],
                    "up": state_dict[up_key],
                }
        cache[adapter_name] = adapter_tensors
    transformer._nunchaku_lite_lora_adapter_cache = cache
    return cache


def _build_lora_scales(active_entries: list[dict], adapter_ranks: dict[str, int]) -> list[float]:
    """Build the per-16-rank-block lora_scales list from adapter strengths.

    The native kernel applies one scale per 16-rank-block.  Each adapter
    occupies a contiguous range of rank blocks; its strength fills every
    block it owns.

    Args:
        active_entries: Active adapter metadata entries in composition order.
        adapter_ranks: Map from adapter name → per-module rank (int).

    Returns:
        Flat list of floats, one per 16-rank-block across the concatenated rank.
    """

    scales: list[float] = []
    for entry in active_entries:
        rank = adapter_ranks.get(entry.get("name", ""), 0)
        if rank == 0:
            continue
        num_blocks = math.ceil(rank / 16)
        scales.extend([float(entry["strength"])] * num_blocks)
    return scales


def recompose_loras(transformer: nn.Module) -> None:
    """Rebuild module LoRA tensors from base state plus all active adapters.

    Optimisations vs. the original implementation:

    - **Cached module scan** (fix #3): uses ``_nunchaku_lite_lora_module_cache``
      instead of calling ``lora_modules()`` (3× named_modules) every time.
    - **Cached base logical form** (fix #4): SVDQ base tensors are pre-unpacked
      in ``ensure_lora_base_state`` and reused here.
    - **Cached adapter logical form** (fix #5): active adapter tensors are
      unpacked once and kept GPU-resident in ``_nunchaku_lite_lora_adapter_cache``.
    - **lora_scales instead of strength baking** (fix #1): SVDQ adapter tensors
      are concatenated *unscaled*; per-adapter strength is applied at kernel
      time via ``_nunchaku_lite_lora_scales``.  This makes strength changes O(1).
    - **Single cat for AWQ/dense** (fix #6): collects all down/up tensors in a
      list and does one ``torch.cat`` instead of O(N²) incremental cats.
    - **In-place parameter update** (fix #8): updates ``.data`` on existing
      ``nn.Parameter`` objects instead of creating new ones.

    Args:
        transformer: Patched transformer whose quantized modules should receive
            the currently active adapter composition.
    """

    ensure_lora_base_state(transformer)

    # Deferred recompose support (fix #2): batch loading skips recompose.
    if transformer._nunchaku_lite_defer_recompose:
        return

    modules = transformer._nunchaku_lite_lora_module_cache
    active_entries = active_lora_entries(transformer)

    # Build adapter cache (fix #5) — also adds "name" to each entry for convenience.
    for i, entry in enumerate(active_entries):
        active_entries[i] = {**entry, "name": active_lora_names(transformer)[i]}
    adapter_cache = _ensure_adapter_cache(transformer)

    for name, module in modules.items():
        if isinstance(module, SVDQW4A4Linear):
            base_logical = transformer._nunchaku_lite_lora_base_logical
            base_down_logical = base_logical[f"{name}.down"]
            base_up_logical = base_logical[f"{name}.up"]

            if not active_entries:
                # No active adapters — restore the original base packed form
                # directly from the cached snapshot (fix: avoids re-packing
                # from logical form, which would pad rank to 16 and produce a
                # different shape than the original base weight).
                base_state = transformer._nunchaku_lite_lora_base_state
                module.proj_down.data = base_state[f"{name}.proj_down"].clone().to(
                    device=module.proj_down.device, dtype=module.proj_down.dtype
                )
                module.proj_up.data = base_state[f"{name}.proj_up"].clone().to(
                    device=module.proj_up.device, dtype=module.proj_up.dtype
                )
                module.rank = base_state[f"{name}.proj_down"].shape[1]
                # No lora scales needed (base has zero LoRA contribution).
                if hasattr(module, "_nunchaku_lite_lora_scales"):
                    delattr(module, "_nunchaku_lite_lora_scales")
                continue

            # Concatenate unscaled logical tensors (fix #1: strength via lora_scales).
            logical_downs = [base_down_logical]
            logical_ups = [base_up_logical]
            adapter_ranks: dict[str, int] = {}
            for entry in active_entries:
                adapter_name = entry["name"]
                adapter_tensors = adapter_cache.get(adapter_name, {})
                if name not in adapter_tensors:
                    continue
                down_logical = adapter_tensors[name]["down"]
                up_logical = adapter_tensors[name]["up"]
                logical_downs.append(down_logical)
                logical_ups.append(up_logical)
                adapter_ranks[adapter_name] = down_logical.shape[0]  # rank is dim 0 for down

            down_cat = torch.cat(logical_downs, dim=0)
            up_cat = torch.cat(logical_ups, dim=1)

            # Pack once (fix #10: only changed tensors would need repack, but
            # cat always changes, so pack is unavoidable here).
            down_packed = pack_lowrank_weight(down_cat, down=True)
            up_packed = pack_lowrank_weight(up_cat, down=False)

            # Update existing Parameter .data (fix #8: no new Parameter allocation).
            module.proj_down.data = down_packed.to(device=module.proj_down.device, dtype=module.proj_down.dtype)
            module.proj_up.data = up_packed.to(device=module.proj_up.device, dtype=module.proj_up.dtype)
            module.rank = down_packed.shape[1]

            # Build and attach lora_scales (fix #1): one scale per 16-rank-block.
            # Base rank contributes scale=0 (no LoRA effect for base weights).
            base_rank = base_down_logical.shape[0]
            base_blocks = math.ceil(base_rank / 16)
            adapter_scales = _build_lora_scales(active_entries, adapter_ranks)
            # Base blocks get 0.0 (they represent the original model, not a LoRA).
            # But actually base_down/up are the original model's LoRA params (rank r),
            # which should always be active at scale 1.0. Let's set them to 1.0.
            full_scales = [1.0] * base_blocks + adapter_scales
            module._nunchaku_lite_lora_scales = full_scales
        else:
            # AWQ / dense path — fix #6: single cat instead of O(N²) incremental.
            base_down = transformer._nunchaku_lite_lora_base_state[f"{name}.proj_down"]
            base_up = transformer._nunchaku_lite_lora_base_state[f"{name}.proj_up"]

            if not active_entries:
                if isinstance(module, AWQW4A16Linear):
                    device = module.qweight.device
                    dtype = module.wscales.dtype
                else:
                    device = module.weight.device
                    dtype = module.weight.dtype
                module._nunchaku_lite_lora_down = base_down.to(device=device, dtype=dtype)
                module._nunchaku_lite_lora_up = base_up.to(device=device, dtype=dtype)
                continue

            down_parts = [base_down]
            up_parts = [base_up]
            for entry in active_entries:
                adapter_name = entry["name"]
                adapter_tensors = adapter_cache.get(adapter_name, {})
                if name not in adapter_tensors:
                    continue
                lora_down = fit_lora_tensor(
                    adapter_tensors[name]["down"], module.in_features, down=True, module_name=name
                )
                lora_up = fit_lora_tensor(
                    adapter_tensors[name]["up"], module.out_features, down=False, module_name=name
                )
                if lora_down.shape[1] != lora_up.shape[1]:
                    raise ValueError(
                        f"LoRA rank mismatch for {name}: "
                        f"proj_down={tuple(lora_down.shape)}, proj_up={tuple(lora_up.shape)}"
                    )
                # Strength is baked for AWQ/dense (no kernel lora_scales support).
                lora_down = lora_down.to(device=base_down.device, dtype=base_down.dtype) * float(entry["strength"])
                lora_up = lora_up.to(device=base_up.device, dtype=base_up.dtype)
                down_parts.append(lora_down)
                up_parts.append(lora_up)

            # Single cat (fix #6).
            down = torch.cat(down_parts, dim=1)
            up = torch.cat(up_parts, dim=1)

            if isinstance(module, AWQW4A16Linear):
                device = module.qweight.device
                dtype = module.wscales.dtype
            else:
                device = module.weight.device
                dtype = module.weight.dtype
            module._nunchaku_lite_lora_down = down.to(device=device, dtype=dtype)
            module._nunchaku_lite_lora_up = up.to(device=device, dtype=dtype)
