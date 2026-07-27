# Multi-LoRA Performance Paths — Logic Map

> **Technique**: Logic Mapping (Trace-Through-Before-You-Build), Phase 1.
> Every function listed below has been verified by reading the actual source
> at the listed file:line.  This document is the single source of truth for
> understanding where time and memory are spent during multi-LoRA operations.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Path A — Adapter Loading](#path-a--adapter-loading)
3. [Path B — Adapter Activation / Composition](#path-b--adapter-activation--composition)
4. [Path C — Strength Tuning](#path-c--strength-tuning)
5. [Path D — Inference Forward (SVDQ W4A4)](#path-d--inference-forward-svdq-w4a4)
6. [Path E — Inference Forward (AWQ W4A16 / Dense)](#path-e--inference-forward-awq-w4a16--dense)
7. [Path F — Disable / Enable / Unload](#path-f--disable--enable--unload)
8. [Path G — LoRA Weight Conversion](#path-g--lora-weight-conversion)
9. [Hot-Spot Summary](#hot-spot-summary)
10. [Glossary](#glossary)

---

## 1. Architecture Overview

### State Fields (all on the transformer object)

| Field | Type | Purpose | Initialized at |
|-------|------|---------|-----------------|
| `_nunchaku_lite_loras` | `OrderedDict[str, dict]` | All loaded adapters, insertion order. Each entry: `{"state_dict": {...}, "strength": float}` | `ensure_lora_runtime` (runtime.py:680) |
| `_nunchaku_lite_active_loras` | `list[str]` | Ordered list of currently-active adapter names | `ensure_lora_runtime` (runtime.py:688) |
| `_nunchaku_lite_lora_enabled` | `bool` | Global on/off flag; when `False`, `active_lora_names` returns `[]` | `ensure_lora_runtime` (runtime.py:689) |
| `_nunchaku_lite_lora_base_state` | `dict[str, Tensor] \| None` | Snapshot of original `proj_down`/`proj_up` per module | `ensure_lora_base_state` (runtime.py:697) |

### Per-Module State

| Module type | LoRA storage | Set by |
|-------------|-------------|--------|
| `SVDQW4A4Linear` | `module.proj_down` / `module.proj_up` (nn.Parameter) | `recompose_loras` (runtime.py:805-806) |
| `AWQW4A16Linear` | `module._nunchaku_lite_lora_down` / `_up` (plain Tensor) | `recompose_loras` (runtime.py:836-837) |
| `DenseRuntimeLoraLinear` | `module._nunchaku_lite_lora_down` / `_up` (plain Tensor) | `recompose_loras` (runtime.py:836-837) |

### Composition Model

**Rank-concatenation**, NOT sequential application. Each adapter's
`proj_down`/`proj_up` are concatenated along the rank dimension. Strength is
multiplied into `proj_down` before concatenation. The fused GEMM computes a
single `up @ (down @ x)` in one pass.

```
Adapter A (rank=16, strength=0.5)    Adapter B (rank=16, strength=0.8)
  proj_down_A * 0.5                   proj_down_B * 0.8
       |                                    |
       v                                    v
  [down_A_packed | down_B_packed]  ← torch.cat(dim=0) → pack_lowrank_weight
  [up_A_packed   | up_B_packed]    ← torch.cat(dim=1) → pack_lowrank_weight
       |                                    |
       +----------------+-------------------+
                        v
              Single fused GEMM in forward
```

---

## Path A — Adapter Loading

**Entry**: `transformer.load_lora(path, strength=1.0, name=None, replace=False)`
or `pipeline.load_lora_weights(path, adapter_name=None)`

```
User calls load_lora / load_lora_weights
    |
    v
[NunchakuLoraMixin.load_lora] (runtime.py:110)
    |  OR
    |  [NunchakuPipelineLoraMixin.load_lora_weights] (runtime.py:257)
    |      → _pipeline_transformer() (runtime.py:425)
    |      → _is_local_lora_source() (runtime.py:535)
    |      → load_lora_state_dict() (runtime.py:523)        ← reads safetensors file
    |      → transformer.load_lora()                        ← delegates to mixin
    |
    v
[load_lora()] (runtime.py:440)
    |
    +--→ ensure_lora_base_state(transformer) (runtime.py:697)
    |       |
    |       +--→ ensure_lora_runtime() (runtime.py:680)     ← inits OrderedDict, list, flag
    |       +--→ if base_state is not None: RETURN          ← idempotent guard
    |       +--→ lora_modules(transformer) (layout.py:27)   ← FULL TREE SCAN every call
    |       |       +--→ modules_by_class(SVDQW4A4Linear)   (layout.py:16)
    |       |       +--→ modules_by_class(AWQW4A16Linear)
    |       |       +--→ modules_by_class(DenseRuntimeLoraLinear)
    |       +--→ for each module: clone proj_down/proj_up   ← SVDQ only; AWQ/dense get empty(0)
    |       +--→ store in _nunchaku_lite_lora_base_state
    |
    +--→ (if replace) clear loras + active_loras
    |
    +--→ resolve_lora_name() (runtime.py:735)               ← derive name from path stem
    |
    +--→ transformer._convert_lora_to_nunchaku(path_or_state_dict)
    |       |
    |       v
    |    [NunchakuZImageTransformerLoraMixin._convert_lora_to_nunchaku] (z_image.py:42)
    |       +--→ load_lora_state_dict() (runtime.py:523)    ← re-reads if path
    |       +--→ is_nunchaku_lite_lora_state_dict() (convert.py:63)
    |       |       TRUE → normalize_nunchaku_lora_state_dict() (convert.py:86)
    |       |       FALSE → convert_z_image_diffusers_lora_state_dict() (z_image.py:60)
    |       |                   +--→ normalize_z_image_diffusers_lora_state_dict() (z_image.py:110)
    |       |                   +--→ lora_modules(transformer) (layout.py:27) ← FULL TREE SCAN
    |       |                   +--→ peft_lora_pairs() (core/peft.py)
    |       |                   +--→ group_fused_projection_pairs() (convert.py:108)  ← q/k/v → to_qkv
    |       |                   +--→ fuse_projection_branches() (convert.py:136)
    |       |                   +--→ set_standard_converted_lora_pair() (convert.py:191)
    |       |                   |       +--→ pack_lowrank_weight() (layout.py:122)   ← SVDQ: pad+permute
    |       |                   |       +--→ pad_lora_tensor() (layout.py:185)       ← AWQ: pad to 16
    |       |                   +--→ validate_nunchaku_lora_state_dict() (convert.py:223)
    |       +--→ return converted dict {name.proj_down, name.proj_up}
    |
    +--→ detach().cpu() every tensor in converted            ← (runtime.py:452)
    |
    +--→ store in _nunchaku_lite_loras[name] = {"state_dict": ..., "strength": ...}
    |
    +--→ dedup + append to _nunchaku_lite_active_loras
    |
    +--→ recompose_loras(transformer) (runtime.py:758)      ← FULL REBUILD (see Path B)
    |
    v
  Returns adapter name
```

### Cost Analysis — Loading N adapters in a loop

| Operation | Per-load cost | N loads total | Issue |
|-----------|--------------|---------------|-------|
| `ensure_lora_base_state` | Guarded — 1× only | O(1) | OK (idempotent) |
| `_convert_lora_to_nunchaku` | O(modules) conversion | N × O(M) | OK (necessary) |
| `lora_modules()` tree scan | O(all modules) | Called inside convert + inside recompose = 2×/load | **Wasteful** — rescans entire tree |
| `recompose_loras` | O(M × active) | N × O(M × N) = O(M·N²) | **Critical** — full rebuild every load |
| `pack_lowrank_weight` | O(rank × features) | Called inside recompose for EVERY previously-loaded adapter too | **Wasteful** — re-packs already-packed tensors |

**Total for loading N adapters sequentially**: O(M·N²) tensor operations, with
N full tree scans and N full recompositions where each recomposition re-processes
all previously loaded adapters.

---

## Path B — Adapter Activation / Composition

**Entry**: `transformer.set_adapters(names, weights)` or `pipeline.set_adapters(names, weights)`

```
User calls set_adapters / set_adapters
    |
    v
[NunchakuLoraMixin.set_adapters] (runtime.py:139)
    |  → set_active_adapters(transformer, adapter_names, weights)
    |
    v
[set_active_adapters()] (runtime.py:625)
    |
    +--→ ensure_lora_runtime()
    |
    +--→ normalize adapter_names to list
    +--→ normalize weights to list (scalar → [scalar] * len)
    +--→ validate: all names exist in _nunchaku_lite_loras
    |
    +--→ REPLACE _nunchaku_lite_active_loras = list(adapter_names)    ← (runtime.py:651)
    |
    +--→ for each (name, weight):
    |       _nunchaku_lite_loras[name]["strength"] = float(weight)    ← (runtime.py:654)
    |
    +--→ recompose_loras(transformer) (runtime.py:758)
    |
    v
[recompose_loras()] (runtime.py:758)
    |
    +--→ ensure_lora_base_state()           ← guarded (1× only)
    +--→ lora_modules(transformer)          ← FULL TREE SCAN (layout.py:27)
    +--→ active_lora_entries(transformer)   ← (runtime.py:670)
    |       +--→ active_lora_names() (runtime.py:657)
    |       |       → filter by _nunchaku_lite_lora_enabled
    |       +--→ return [loras[name] for name in active_names]
    |
    +--→ FOR EACH module (name, module):
    |
    +======== SVDQW4A4Linear branch (runtime.py:769) ========+
    |                                                        |
    |  base_down = base_state[f"{name}.proj_down"]           |
    |  base_up   = base_state[f"{name}.proj_up"]             |
    |                                                        |
    |  if no active entries:                                 |
    |      down = base_down.to(device, dtype)                |
    |      up   = base_up.to(device, dtype)                  |
    |  else:                                                 |
    |      logical_downs = [svdq_down_to_logical(base_down)] | ← UNPACK base every time
    |      logical_ups   = [svdq_up_to_logical(base_up)]     | ← (layout.py:81, 96)
    |                                                        |
    |      for entry in active_entries:                      |
    |          down_logical = svdq_down_to_logical(          | ← UNPACK adapter tensor
    |              state_dict[down_key], module, name)       |   (layout.py:81)
    |          up_logical   = svdq_up_to_logical(            | ← UNPACK adapter tensor
    |              state_dict[up_key], module, name)         |   (layout.py:96)
    |          .to(device, dtype)                            |
    |          logical_downs.append(                         |
    |              down_logical * float(entry["strength"]))  | ← NEW TENSOR for strength (runtime.py:798)
    |          logical_ups.append(up_logical)                |
    |                                                        |
    |      down = pack_lowrank_weight(                       | ← RE-PACK (layout.py:122)
    |          torch.cat(logical_downs, dim=0), down=True)   |   pad + view + permute + .contiguous()
    |      up = pack_lowrank_weight(                         | ← RE-PACK (layout.py:122)
    |          torch.cat(logical_ups, dim=1), down=False)    |
    |      .to(device, dtype)                                |
    |                                                        |
    |  module.proj_down = nn.Parameter(down, ...)            | ← NEW Parameter (runtime.py:805)
    |  module.proj_up   = nn.Parameter(up, ...)              | ← NEW Parameter (runtime.py:806)
    |  module.rank = down.shape[1]                           |
    |                                                        |
    +========================================================+
    |                                                        |
    +======== AWQ/Dense branch (runtime.py:809) =============+
    |                                                        |
    |  down = base_state[f"{name}.proj_down"]                | ← starts as empty(0) tensor
    |  up   = base_state[f"{name}.proj_up"]                  |
    |                                                        |
    |  for entry in active_entries:                          |
    |      lora_down = fit_lora_tensor(state_dict[down_key]) | ← (layout.py:58)
    |      lora_up   = fit_lora_tensor(state_dict[up_key])   |
    |      lora_down = .to(...) * float(entry["strength"])   | ← NEW TENSOR (runtime.py:827)
    |      down = torch.cat([down, lora_down], dim=1)        | ← O(N²) INCREMENTAL CAT (runtime.py:828)
    |      up   = torch.cat([up, lora_up], dim=1)            | ← O(N²) INCREMENTAL CAT (runtime.py:829)
    |                                                        |
    |  .to(device, dtype)                                    |
    |  module._nunchaku_lite_lora_down = down                |
    |  module._nunchaku_lite_lora_up   = up                  |
    |                                                        |
    +========================================================+
```

### Cost Analysis — recompose_loras with A active adapters across M modules

| Operation | Cost | Issue |
|-----------|------|-------|
| `lora_modules()` | O(M_total) tree scan | **Rescans every call** — no cache |
| `svdq_down_to_logical` (base) | O(rank × features) per module | **Re-unpacks base every time** |
| `svdq_down_to_logical` (each adapter) | O(rank × features) × A | Re-unpacks from CPU state_dict every time |
| `down_logical * strength` | Allocates new tensor | **Not in-place** — could use `.mul_` |
| `pack_lowrank_weight` | O(rank × features) — pad+permute+contiguous | Expensive layout transform, redone every call |
| `nn.Parameter(down, ...)` | New wrapper object | **Could update `.data` in-place** |
| AWQ `torch.cat` in loop | O(N²) — each cat copies all prior data | **Should use list + single cat** |

---

## Path C — Strength Tuning

**Entry**: `transformer.set_lora_strength(strength, name=None)`

```
User calls set_lora_strength
    |
    v
[NunchakuLoraMixin.set_lora_strength] (runtime.py:131)
    |  → set_lora_strength(transformer, strength, name)
    |
    v
[set_lora_strength()] (runtime.py:476)
    |
    +--→ ensure_lora_runtime()
    +--→ resolve name (if None, require exactly 1 active)
    +--→ _nunchaku_lite_loras[name]["strength"] = float(strength)
    +--→ recompose_loras(transformer)                    ← FULL REBUILD (Path B)
    |
    v
  Done — but EVERY module was fully rebuilt just to change one scalar
```

### Cost Analysis

Changing a single adapter's strength triggers the **entire recompose_loras**
pipeline: tree scan → unpack base → unpack all adapters → multiply → re-pack →
new Parameters. This is the most common interactive operation (tuning strength
during generation) and it has O(M × A) cost when it should be O(1) if
`lora_scales` were wired through.

### The Unused Optimization: `lora_scales`

The GEMM kernel already supports per-rank-block scales:

```
svdq_gemm_w4a4_cuda() (gemm.py:10)
    parameter: lora_scales: list[float] | None  (gemm.py:31)
    default:   [1.0] * ceil(rank / 16)           (gemm.py:79-81)
    passed to: ops.gemm_w4a4(..., lora_scales)   (gemm.py:110)
```

But `forward_quant` **never passes `lora_scales`**:

```
SVDQW4A4Linear.forward_quant() (linear.py:193)
    → svdq_gemm_w4a4_cuda(
          act=..., wgt=..., out=..., ascales=..., wscales=...,
          lora_act_in=..., lora_up=...,
          bias=..., fp4=..., alpha=..., wcscales=..., act_unsigned=...
          # NO lora_scales= argument  ← ALWAYS defaults to [1.0]*ceil(rank/16)
      )
```

If `lora_scales` were passed, strength changes would only need to update a
Python list — no tensor rebuild at all. This eliminates the entire recompose
path for strength-tuning.

---

## Path D — Inference Forward (SVDQ W4A4)

**Entry**: `module.forward(x)` during transformer generation

```
SVDQW4A4Linear.forward(x, output=None) (linear.py:233)
    |
    +--→ reshape x to (batch*seq, channels)
    +--→ allocate output if None
    |
    +--→ self.quantize(x) (linear.py:173)
    |       |
    |       v
    |    svdq_quantize_w4a4_act_fuse_lora_cuda() (quantize.py:9)
    |       +--→ rank = lora_down.shape[1]                    (quantize.py:42)
    |       +--→ lora_act_out = torch.empty(                  (quantize.py:56)
    |       |       batch_size_pad, rank,
    |       |       dtype=torch.float32, ...)                 ← ALWAYS float32
    |       +--→ ops.quantize_w4a4_act_fuse_lora(             (quantize.py:58)
    |       |       input, output, oscales,
    |       |       lora_down, lora_act_out,                  ← FUSED: quantize + lora_down matmul
    |       |       smooth, fuse_glu, fp4)
    |       +--→ return (output, oscales, lora_act_out)
    |
    +--→ self.forward_quant(quantized_x, ascales, lora_act, output) (linear.py:196)
    |       |
    |       v
    |    svdq_gemm_w4a4_cuda() (gemm.py:10)
    |       +--→ lora_scales defaults to [1.0]*ceil(rank/16)  (gemm.py:79)
    |       |       ← strength was ALREADY baked into proj_down at recompose time
    |       +--→ ops.gemm_w4a4(                               (gemm.py:85)
    |       |       act, wgt, out, qout, ascales, wscales, oscales, poolout,
    |       |       lora_act_in, lora_up, lora_down,
    |       |       lora_act_out, norm_q, norm_k, rotary_emb, bias,
    |       |       smooth_factor, out_vk, out_linearattn,
    |       |       act_unsigned, lora_scales,                ← PASSED but always [1.0,...]
    |       |       fuse_silu, fp4, alpha_arg, wcscales,
    |       |       out_q, out_k, out_v, attn_tokens)
    |       +--→ output filled in-place
    |
    +--→ return output.reshape(batch, seq, -1)
```

### Key Observation

The forward path is **already optimized** — it uses fused quantize+LoRA-down and
fused GEMM+LoRA-up. The only inefficiency is that `lora_scales` is hardcoded to
`[1.0, ...]` because `forward_quant` doesn't pass it, forcing strength to be
baked into weights at compose time (Path B/C).

---

## Path E — Inference Forward (AWQ W4A16 / Dense)

**Entry**: `module.forward(x)` during transformer generation

```
AWQW4A16Linear.forward(x) (linear.py:357)
    |
    +--→ reshape x to (-1, in_features)
    +--→ base GEMM: awq_gemm_w4a16_g64_int32() or _forward_gemv_chunks()
    +--→ add bias
    |
    +--→ lora_down = getattr(self, "_nunchaku_lite_lora_down", None)
    +--→ lora_up   = getattr(self, "_nunchaku_lite_lora_up", None)
    |
    +--→ if lora_down is not None and lora_up is not None and shape[1] > 0:
    |       |
    |       +--→ device check: .to(x.device) if needed     ← lazy GPU transfer
    |       +--→ lora = torch.matmul(x.to(lora_down.dtype), lora_down)   ← SEPARATE matmul #1
    |       +--→ lora = torch.matmul(lora, lora_up.transpose(0, 1))      ← SEPARATE matmul #2
    |       +--→ output.add_(lora.to(output.dtype))                      ← in-place add
    |
    +--→ return output

DenseRuntimeLoraLinear.forward(input) (linear.py:34)
    |
    +--→ output = F.linear(input, self.weight, self.bias)     ← base dense matmul
    +--→ lora = torch.matmul(input.to(lora_down.dtype), lora_down)        ← SEPARATE matmul #1
    +--→ lora = torch.matmul(lora, lora_up.transpose(0, 1))               ← SEPARATE matmul #2
    +--→ return output + lora.to(output.dtype)
```

### Key Observation

AWQ and dense paths use **2 separate matmuls** (not fused into the base GEMM).
This is inherent to the AWQ kernel design — no fused AWQ+LoRA kernel exists.
The strength is already baked into `lora_down` at recompose time.

---

## Path F — Disable / Enable / Unload

```
disable_lora() (runtime.py:148)
    |
    +--→ _nunchaku_lite_lora_enabled = False
    +--→ recompose_loras()                           ← FULL REBUILD to base state
            → active_lora_names() returns []         (runtime.py:659)
            → every module restored to base_state    (runtime.py:773-776)

enable_lora() (runtime.py:142)
    |
    +--→ _nunchaku_lite_lora_enabled = True
    +--→ recompose_loras()                           ← FULL REBUILD with active adapters

unload_lora() / reset_lora(name) (runtime.py:127, 500)
    |
    +--→ clear _nunchaku_lite_loras (or remove one)
    +--→ update _nunchaku_lite_active_loras
    +--→ recompose_loras()                           ← FULL REBUILD
```

### Cost

Every enable/disable/unload triggers a full `recompose_loras`. For disable, this
means unpacking base state and repacking it just to zero out the LoRA branch.

---

## Path G — LoRA Weight Conversion

This path runs once per `load_lora` call, converting external LoRA formats
(Diffusers/PEFT, Kohya/ai-toolkit) into Nunchaku Lite packed tensors.

```
_convert_lora_to_nunchaku(path_or_state_dict) (z_image.py:42)
    |
    +--→ load_lora_state_dict()                      ← read safetensors if path
    |
    +--→ is_nunchaku_lite_lora_state_dict() (convert.py:63)
    |       TRUE  → normalize_nunchaku_lora_state_dict() (convert.py:86)
    |       FALSE → convert_z_image_diffusers_lora_state_dict() (z_image.py:60)
    |
    v
[convert_z_image_diffusers_lora_state_dict] (z_image.py:60)
    |
    +--→ normalize_z_image_diffusers_lora_state_dict() (z_image.py:110)
    |       → rewrite keys: lora_down→lora_A, lora_up→lora_B, strip prefixes
    |
    +--→ lora_modules(transformer)                   ← FULL TREE SCAN (layout.py:27)
    |
    +--→ peft_lora_pairs(diffusers_state)            ← extract (lora_A, lora_B) pairs
    |
    +--→ for QKV specs: group_fused_projection_pairs() (convert.py:108)
    |       → match to_q/to_k/to_v → to_qkv target
    |       → fuse_projection_branches() (convert.py:136)
    |       → set_standard_converted_lora_pair() (convert.py:191)
    |           → pack_lowrank_weight() (layout.py:122)    ← SVDQ packing
    |
    +--→ for SwiGLU specs: same flow (w1+w3 → net.0.proj)
    |
    +--→ for remaining direct targets:
    |       → set_standard_converted_lora_pair() (convert.py:191)
    |
    +--→ validate_nunchaku_lora_state_dict() (convert.py:223)
    |
    +--→ return {name.proj_down: packed_tensor, name.proj_up: packed_tensor}
```

### pack_lowrank_weight internals (layout.py:122)

This is the layout transformation for SVDQ W4A4 kernels:

```
pack_lowrank_weight(weight, down) (layout.py:122)
    |
    +--→ if float32: convert to bfloat16              (layout.py:139)
    +--→ if not fp16/bf16: raise ValueError
    |
    +--→ frag_n = 16, frag_k = 16                     (layout.py:142-146)
    |
    +--→ pad_lora_tensor(weight, divisor=(16,16))     (layout.py:147)
    |       → pad rank/features to multiples of 16
    |
    +--→ if down:
    |       view(rank_frags, 16, channel_frags, 16)
    |       .permute(2, 0, 1, 3)                      ← memory layout rearrange
    |       .reshape → .permute → .contiguous()       ← FINAL .contiguous() = copy
    |
    +--→ if up:
    |       view(channel_frags, 16, rank_frags, 16)
    |       .permute(0, 2, 1, 3)
    |       .reshape → .permute → .contiguous()       ← FINAL .contiguous() = copy
    |
    +--→ return packed tensor (channels, rank)
```

The `.contiguous()` call at the end forces a full copy. This runs inside
`recompose_loras` for every module, every recompose — even for tensors that
were already packed and haven't changed.

---

## Hot-Spot Summary

Ranked by combined impact on setup speed and interactive responsiveness:

### 🔴 Critical

| # | Hot-spot | Location | Why it's expensive |
|---|----------|----------|--------------------|
| 1 | **`lora_scales` not wired through `forward_quant`** | linear.py:196 (missing arg) | Forces full `recompose_loras` on every strength change. Should be O(1) list update. |
| 2 | **`recompose_loras` called per-load in batch loops** | runtime.py:457 (inside `load_lora`) | Loading N adapters = N full rebuilds = O(M·N²). Should be 1 rebuild. |
| 3 | **`lora_modules()` full tree scan every recompose** | layout.py:27, called at runtime.py:764 | `named_modules()` walks entire transformer tree. Should be cached. |
| 4 | **Base state re-unpacked from packed form every recompose** | runtime.py:777-778 (`svdq_down_to_logical(base_down)`) | `unpack_lowrank_weight` does view+permute+contiguous = copy. Base never changes. |
| 5 | **Adapter tensors re-unpacked from CPU state_dict every recompose** | runtime.py:785-786 | Each `svdq_down_to_logical(state_dict[key])` unpacks from CPU + transfers to GPU. Should cache GPU-resident logical form. |

### 🟡 Medium

| # | Hot-spot | Location | Why it's expensive |
|---|----------|----------|--------------------|
| 6 | **AWQ/dense O(N²) incremental `torch.cat`** | runtime.py:828-829 | Each `torch.cat([down, lora_down])` copies all prior data. N adapters = N² copies. Should use list + single `torch.cat`. |
| 7 | **Strength multiply allocates new tensor** | runtime.py:798 (`down_logical * float(strength)`) | Creates new tensor. Could use `.mul_()` in-place. |
| 8 | **New `nn.Parameter` created every recompose** | runtime.py:805-806 | Wraps tensor in new Parameter object. Could update `.data` in-place. |
| 9 | **Transient memory spike during recompose** | runtime.py:790-803 | `logical_downs` list + `torch.cat` + `pack_lowrank_weight` all alive simultaneously. Peak = 3× final tensor size. |
| 10 | **`pack_lowrank_weight` re-runs for unchanged tensors** | layout.py:122, called at runtime.py:801-802 | If adapter set hasn't changed, packing produces identical output. Could cache. |

### 🟢 Low

| # | Hot-spot | Location | Why it's expensive |
|---|----------|----------|--------------------|
| 11 | **Inactive adapters stay on CPU in state_dict** | runtime.py:452 (`.detach().cpu()`) | All loaded adapters consume CPU RAM even when inactive. No eviction. |
| 12 | **Base state stores full SVDQ packed tensors** | runtime.py:703-704 (`.clone()`) | For SVDQ, base proj_down/proj_up are full-size packed tensors. Could store logical (unpacked) form to avoid re-unpacking. |

---

## Path Interaction Diagram

```
                    ┌─────────────────────────────────────────────┐
                    │           USER OPERATIONS                    │
                    └──────┬──────┬──────┬──────┬──────┬──────────┘
                           │      │      │      │      │
                     load   │ set  │ set  │ dis/ │ unld │
                     lora   │ adapters│ strength│ en  │ lora  │
                           │      │      │      │      │
                           v      v      v      v      v
                    ┌─────────────────────────────────────────────┐
                    │        recompose_loras()                     │
                    │        (runtime.py:758)                      │
                    │                                             │
                    │  ┌─── lora_modules() ──── FULL TREE SCAN ──┐ │
                    │  │                                        │ │
                    │  │  FOR EACH MODULE:                       │ │
                    │  │  ┌─ SVDQ path ──────────────────────┐  │ │
                    │  │  │ unpack base (svdq_down_to_logical)│  │ │
                    │  │  │ for each active adapter:           │  │ │
                    │  │  │   unpack (svdq_down_to_logical)    │  │ │
                    │  │  │   strength * down (NEW TENSOR)     │  │ │
                    │  │  │ cat all downs                      │  │ │
                    │  │  │ pack_lowrank_weight (COPY)         │  │ │
                    │  │  │ new nn.Parameter                   │  │ │
                    │  │  └────────────────────────────────────┘  │ │
                    │  │  ┌─ AWQ/Dense path ─────────────────┐   │ │
                    │  │  │ for each active adapter:           │   │ │
                    │  │  │   fit_lora_tensor                  │   │ │
                    │  │  │   strength * down (NEW TENSOR)     │   │ │
                    │  │  │   torch.cat [O(N²)]                │   │ │
                    │  │  │ set _nunchaku_lite_lora_down/up    │   │ │
                    │  │  └────────────────────────────────────┘   │ │
                    │  └──────────────────────────────────────────┘ │
                    └─────────────────────┬─────────────────────────┘
                                          │
                                          v
                    ┌─────────────────────────────────────────────┐
                    │        INFERENCE FORWARD                     │
                    │                                             │
                    │  SVDQ: quantize (fused lora_down)            │
                    │       → forward_quant (fused gemm+lora_up)  │
                    │       → lora_scales = [1.0, ...]  ← UNUSED   │
                    │                                             │
                    │  AWQ/Dense: base GEMM                        │
                    │       → matmul(x, lora_down)  [separate]     │
                    │       → matmul(lora, lora_up) [separate]     │
                    │       → output + lora                        │
                    └─────────────────────────────────────────────┘
```

---

## Glossary

| Term | Meaning |
|------|---------|
| **Rank-concatenation** | Multiple LoRA adapters are combined by concatenating their down/up projections along the rank dimension, not by sequential application or weighted sum |
| **SVDQ** | Singular Value Decomposition Quantization — W4A4 (4-bit weights, 4-bit activations) |
| **AWQ** | Activation-aware Weight Quantization — W4A16 (4-bit weights, 16-bit activations) |
| **Packed layout** | SVDQ LoRA tensors use a permuted memory layout (frag_n=16, frag_k=16) for kernel alignment |
| **Logical form** | The mathematical rank-by-features layout before packing — what you get after `unpack_lowrank_weight` |
| **`lora_scales`** | Per-16-rank-block scalar multipliers supported by the GEMM kernel but never passed by `forward_quant` |
| **Recompose** | Full rebuild of all module LoRA tensors from base state + active adapters |
| **Fused projection** | Multiple Diffusers LoRA branches (e.g. to_q, to_k, to_v) merged into one Nunchaku module (to_qkv) |
