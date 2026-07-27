# Multi-LoRA Loading — Logic Map

> **Technique**: Trace-Through-Before-You-Build (Logic Mapping Technique).
> **Scope**: How nunchaku-lite loads, composes, and applies multiple runtime
> LoRA adapters end-to-end, from user API call to fused kernel.
> **Source verified against**: `src/nunchaku_lite/lora/core/runtime.py`,
> `src/nunchaku_lite/linear.py` (as of 2026-07-27).

---

## Entry Point → Exit Point (Pipeline-level)

This is the path used by `generate_multi.py`, `generate.py`, and all
documented examples in `docs/models/flux.md`.

```
User script
  pipe.load_lora_weights(path, adapter_name="A")   ← called once per LoRA
  pipe.load_lora_weights(path, adapter_name="B")
  pipe.set_adapters(["A","B"], adapter_weights=[wA, wB])
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  INFERENCE: pipe(prompt, ...)                           │
│  → transformer forward → SVDQW4A4Linear.forward        │
│  → fused quantize+LoRA-down + fused GEMM+LoRA-up       │
└─────────────────────────────────────────────────────────┘
```

---

## Flow 1 — Loading one adapter: `load_lora_weights`

```
pipe.load_lora_weights(path, adapter_name="A")
    │
    ▼  runtime.py:257  NunchakuPipelineLoraMixin.load_lora_weights
    │
    ├── _is_local_lora_source(path)?   runtime.py:540
    │       │
    │   YES (local .safetensors)              NO (hub repo id)
    │       │                                     │
    │       ▼                                     ▼
    │   load_lora_state_dict(path)         self.lora_state_dict(repo, **kwargs)
    │   runtime.py:533                      (Diffusers loader)
    │       │                                     │
    │       ▼                                     ▼
    │   raise_if_text_encoder_lora(sd)     raise_if_text_encoder_lora(sd)
    │   runtime.py:593                      runtime.py:593
    │       │                                     │
    │       ▼                                     ▼
    │   transformer.load_lora(sd, name="A")  transformer.load_lora(sd, name="A")
    │       │
    │       ▼  runtime.py:88  NunchakuLoraMixin.load_lora  →  load_lora()  runtime.py:440
    │       │
    │       ▼
    │   ┌──────────── LOAD_LORA (module function) ────────────┐
    │   │ 1. ensure_lora_base_state(transformer)  :445        │
    │   │    → snapshots base proj_down/proj_up ONCE          │
    │   │      (SVDQ: clone real tensors; AWQ/dense: empty)   │
    │   │                                                      │
    │   │ 2. resolve_lora_name(transformer, path, name) :450  │
    │   │    → name = Path(path).stem if name is None         │
    │   │    → ValueError if name already in _nunchaku_lite_loras
    │   │                                                      │
    │   │ 3. _convert_lora_to_nunchaku(path_or_sd)  :451      │
    │   │    → model-specific (abstract in base mixin :256)   │
    │   │    → e.g. convert_z_image_diffusers_lora_state_dict │
    │   │    → returns {module.proj_down, module.proj_up}     │
    │   │                                                      │
    │   │ 4. Store:                                            │
    │   │    _nunchaku_lite_loras["A"] = {                     │
    │   │        "state_dict": converted (CPU, detached),      │
    │   │        "strength": 1.0   ← DEFAULT, not user's yet   │
    │   │    }                                                 │
    │   │    runtime.py:452-454                                │
    │   │                                                      │
    │   │ 5. Append to active list (dedup first):             │
    │   │    _active_loras = [x for x in _active if x != "A"] │
    │   │    _active_loras.append("A")                        │
    │   │    runtime.py:455-456                                │
    │   │                                                      │
    │   │ 6. recompose_loras(transformer)  :457               │
    │   └──────────────────────────────────────────────────────┘
    │
    ▼  (returns to load_lora_weights, which returns None)
```

### Data contract at storage

```python
transformer._nunchaku_lite_loras: OrderedDict[str, {
    "state_dict": dict[str, Tensor],   # {f"{mod}.proj_down": [...], f"{mod}.proj_up": [...]}
    "strength": float,                  # 1.0 at load time
}]
transformer._nunchaku_lite_active_loras: list[str]  # ["A"] then ["A","B"]
```

---

## Flow 2 — Activating multiple: `set_adapters`

```
pipe.set_adapters(["A","B"], adapter_weights=[0.65, 0.35])
    │
    ▼  runtime.py:348  NunchakuPipelineLoraMixin.set_adapters
    │
    ├── transformer_adapter_weights(adapter_weights, "transformer")  runtime.py:580
    │     → extracts scalar from dict if needed; passes list through
    │
    ▼  transformer.set_adapters(["A","B"], [0.65, 0.35])
    │  runtime.py:171  NunchakuLoraMixin.set_adapters
    │
    ▼  set_active_adapters(transformer, ["A","B"], [0.65, 0.35])
    │  runtime.py:628
    │
    ├── ensure_lora_runtime(transformer)                 :636
    ├── normalize: adapter_names = list; weights = list   :637-638
    ├── LENGTH CHECK: len(names) == len(weights)          :641  ← raises if mismatch
    ├── EXISTENCE CHECK: all names in _nunchaku_lite_loras :647  ← raises if missing
    │
    ├── REPLACE active list:
    │     _nunchaku_lite_active_loras = ["A", "B"]         :651
    │     (completely replaces, NOT append)
    │
    ├── SET STRENGTHS (overwrites the 1.0 defaults):
    │     _nunchaku_lite_loras["A"]["strength"] = 0.65     :654
    │     _nunchaku_lite_loras["B"]["strength"] = 0.35     :654
    │     (dict weights → weight.get("transformer", 1.0))  :653
    │
    ▼  recompose_loras(transformer)                       :655
```

**Key:** `set_adapters` is where user strengths are actually applied.
`load_lora_weights` always stores `strength=1.0`.

---

## Flow 3 — The composition engine: `recompose_loras`

```
recompose_loras(transformer)                         runtime.py:758
    │
    ├── ensure_lora_base_state(transformer)            :763
    ├── modules = lora_modules(transformer)            :764  (all SVDQ/AWQ/dense linears)
    ├── active_entries = active_lora_entries(transformer):765
    │     → [ _nunchaku_lite_loras[name] for name in active_lora_names() ]
    │     → active_lora_names respects _lora_enabled flag
    │
    ▼  FOR EACH (name, module) in modules:
    │
    ├─── SVDQW4A4Linear branch                         :767
    │    │
    │    ├── base_down, base_up from _lora_base_state   :768-769
    │    │
    │    ├── IF no active entries:                      :771
    │    │     down = base_down; up = base_up  (restore base)
    │    │
    │    └── ELSE (active adapters exist):              :776
    │          logical_downs = [svdq_down_to_logical(base_down)]  :777
    │          logical_ups   = [svdq_up_to_logical(base_up)]      :778
    │          │
    │          FOR EACH entry in active_entries:        :779
    │            ├── down_logical = svdq_down_to_logical(sd[f"{name}.proj_down"])  :785
    │            ├── up_logical   = svdq_up_to_logical(sd[f"{name}.proj_up"])      :786
    │            ├── rank check: down.shape[0] == up.shape[1]                      :788
    │            ├── ★ STRENGTH APPLIED HERE ★:
    │            │     logical_downs.append(down_logical * float(entry["strength"]))  :798
    │            │     logical_ups.append(up_logical)              :799
    │            │     (strength multiplies proj_down ONLY, not proj_up)
    │          │
    │          down = pack_lowrank_weight(torch.cat(logical_downs, dim=0))  :801
    │          up   = pack_lowrank_weight(torch.cat(logical_ups,   dim=1))  :802
    │          │     (dim=0 for down = rank concat; dim=1 for up = rank concat)
    │          │
    │          module.proj_down = nn.Parameter(down)    :805
    │          module.proj_up   = nn.Parameter(up)      :806
    │          module.rank      = down.shape[1]         :807
    │
    └─── AWQ / dense branch                            :809
         │
         ├── down = base_down (empty rank-0); up = base_up (empty rank-0)
         │
         FOR EACH entry in active_entries:             :812
           ├── lora_down = fit_lora_tensor(sd[down_key], in_features)  :818
           ├── lora_up   = fit_lora_tensor(sd[up_key], out_features)   :819
           ├── rank check                               :821
           ├── ★ STRENGTH APPLIED HERE ★:
           │     lora_down = lora_down * float(entry["strength"])     :825
           │     lora_up   = lora_up                                   :826
           ├── down = torch.cat([down, lora_down], dim=1)             :827
           └── up   = torch.cat([up, lora_up],   dim=1)               :828
         │
         module._nunchaku_lite_lora_down = down        :836
         module._nunchaku_lite_lora_up   = up          :837
```

---

## Flow 4 — Forward (inference): where the composed tensors are consumed

```
SVDQW4A4Linear.forward(x)                              linear.py:~226
    │
    ├── quantize(x, pad_size)                          linear.py:188
    │     → svdq_quantize_w4a4_act_fuse_lora_cuda(
    │         x, lora_down=self.proj_down, ...)        linear.py:194
    │     ★ FUSED: quantize activation + matmul(x, proj_down)
    │     Returns: (quantized_x, ascales, lora_act)
    │            where lora_act = x_quantized @ proj_down  (low-rank)
    │
    ▼
    forward_quant(quantized_x, ascales, lora_act)      linear.py:196
    │
    └── svdq_gemm_w4a4_cuda(
            act=quantized_x, wgt=self.qweight, ...     linear.py:213
            lora_act_in=lora_act,                      linear.py:216
            lora_up=self.proj_up, ...)                 linear.py:217
        ★ FUSED: base W4A4 GEMM + matmul(lora_act, proj_up)
        → output = base_GEMM(x, W) + lora_act @ proj_up.T

─────────────────────────────────────────────────────────

DenseRuntimeLoraLinear.forward(input)                 linear.py:34
    │
    ├── output = F.linear(input, self.weight, self.bias)    :42
    ├── lora = matmul(input, _nunchaku_lite_lora_down)      :52
    ├── lora = matmul(lora, lora_up.transpose(0,1))         :53
    └── return output + lora                                :54
    (2 separate matmuls, NOT fused)
```

---

## Complete end-to-end data flow

```
                    ┌─────────────────────────┐
                    │  N LoRA .safetensors     │
                    │  files on disk           │
                    └────────────┬─────────────┘
                                 │  load_lora_weights × N
                                 ▼
              ┌──────────────────────────────────────┐
              │  _nunchaku_lite_loras (OrderedDict)  │
              │  "A": {sd, strength=1.0}             │
              │  "B": {sd, strength=1.0}             │
              │  ... (N entries, insertion order)    │
              └──────────────┬───────────────────────┘
                             │  set_adapters(["A","B"], [wA, wB])
                             ▼
              ┌──────────────────────────────────────┐
              │  strengths overwritten:              │
              │  "A".strength = wA                   │  runtime.py:654
              │  "B".strength = wB                   │
              │  _active_loras = ["A", "B"]          │  runtime.py:651
              └──────────────┬───────────────────────┘
                             │  recompose_loras()
                             ▼
         ┌─────────────────────────────────────────────────┐
         │  PER MODULE (SVDQ path):                        │
         │  proj_down = pack(cat([base_down,              │
         │                        A_down × wA,            │  runtime.py:798
         │                        B_down × wB], dim=0))   │
         │  proj_up   = pack(cat([base_up,                │
         │                        A_up,                    │  runtime.py:799
         │                        B_up], dim=1))          │
         │  rank = base_rank + rankA + rankB              │
         └────────────────────┬────────────────────────────┘
                              │  forward()
                              ▼
         ┌─────────────────────────────────────────────────┐
         │  lora_act = quantize(x) @ proj_down   (fused)  │  linear.py:194
         │  output   = GEMM(x, W) + lora_act @ proj_up    │  linear.py:216-217
         │           = base + Σᵢ (x @ Aᵢ_down×wᵢ) @ Aᵢ_up │
         └─────────────────────────────────────────────────┘
```

### Mathematical formulation

Composition is rank-concatenation, NOT a weighted sum of independent deltas:

$$\text{output} = W_{\text{base}} \cdot x + \sum_{i} w_i \cdot (A_i^{\text{down}} \cdot x) \cdot A_i^{\text{up}}$$

implemented as a single concatenated low-rank matmul (fused into the W4A4
GEMM for SVDQ), not N sequential applications. The strength scalar $w_i$
multiplies only the down-projection of each adapter before concatenation.

---

## Branch analysis (all paths traced)

| Branch | Location | Trigger | Outcome |
|--------|----------|---------|---------|
| Local file vs hub repo | `runtime.py:284` | `_is_local_lora_source()` | Local → direct converter; Hub → Diffusers `lora_state_dict()` |
| Text-encoder LoRA keys | `runtime.py:593` | `raise_if_text_encoder_lora()` | Raises `ValueError` — only transformer LoRA supported |
| No active adapters | `runtime.py:771` | `active_entries == []` | Restores base proj_down/proj_up (no LoRA) |
| `enable_lora` / `disable_lora` | `runtime.py:240/246` | `_lora_enabled` flag | `active_lora_names` returns `[]` when disabled → recompose restores base |
| Duplicate name at load | `runtime.py:510` | `resolve_lora_name` | Raises `ValueError` — must use `replace=True` or unique name |
| `hotswap=True` | `runtime.py:131, 265` | PEFT hotswap flag | Raises `NotImplementedError` |
| `fuse_lora()` | `runtime.py:235` | Diffusers fuse request | Raises `NotImplementedError` — adapters stay as low-rank branches |
| `reset_to_overwritten_params` | `runtime.py:385` | Diffusers unload option | Raises `NotImplementedError` — never overwrites dense params |

---

## Key findings from the trace

### 1. Strength is applied exactly once, at the right place

`load_lora_weights` stores `strength=1.0` (`runtime.py:453`). User strengths
are applied by `set_adapters` (`runtime.py:654`) which overwrites the field,
then `recompose_loras` multiplies `proj_down * strength` (`runtime.py:798` for
SVDQ, `:825` for AWQ). No double-application.

### 2. Composition is rank-concatenation, NOT weighted sum

Each adapter's `proj_down`/`proj_up` are concatenated along the rank
dimension (`torch.cat(..., dim=0)` for SVDQ down, `dim=1` for SVDQ up and all
AWQ/dense). The strength scalar multiplies only the down-projection of each
adapter before concatenation.

### 3. Order is mathematically immaterial

Because concatenation along the rank dimension followed by matmul is invariant
to block ordering, the order of names in `set_adapters(["A","B"], ...)` vs
`["B","A"], ...` produces identical output. Only the **set** of active
adapters and their **strengths** matter.

### 4. `recompose_loras` is called N+1 times when loading N LoRAs

Each `load_lora_weights` → `load_lora` → `recompose_loras`
(`runtime.py:457`), then `set_adapters` → `recompose_loras` again
(`runtime.py:655`). For N adapters, that's N+1 full recompositions. Each
recompose re-copies all base + adapter tensors and re-concatenates. This is a
**performance inefficiency** (not a correctness bug) — the final state is
identical regardless.

### 5. `set_adapters` REPLACES the active list, doesn't append

`runtime.py:651`: `_nunchaku_lite_active_loras = list(adapter_names)`. This
means calling `set_adapters(["A"])` after `set_adapters(["A","B"])`
deactivates B (but B remains loaded in `_nunchaku_lite_loras`).

### 6. Base state is snapshotted once, on first load

`ensure_lora_base_state` (`runtime.py:467`) clones `proj_down`/`proj_up` into
`_nunchaku_lite_lora_base_state` on the first `load_lora` call and never
re-snapshots. All subsequent recomposes rebuild from this frozen base + active
adapters. This is what allows `reset_lora` / `delete_adapters` to cleanly
restore base state.

### 7. `generate_multi.py` is correct

Its pattern (`load_lora_weights` per adapter with
`adapter_name=Path.stem` → single `set_adapters(names, weights)`) matches the
source API exactly. The dedup guard
(`if adapter_name in loaded_adapters: continue`) correctly prevents the
`ValueError` at `runtime.py:510`. Strengths are applied once via
`set_adapters`, not at load time.

---

## File reference index

| File | Key symbols | Lines |
|------|-------------|-------|
| `src/nunchaku_lite/lora/core/runtime.py` | `NunchakuPipelineLoraMixin.load_lora_weights` | 257 |
| | `NunchakuPipelineLoraMixin.set_adapters` | 348 |
| | `NunchakuPipelineLoraMixin.delete_adapters` | 365 |
| | `NunchakuPipelineLoraMixin.unload_lora_weights` | 374 |
| | `NunchakuLoraMixin.load_lora` (method) | 88 |
| | `NunchakuLoraMixin.set_adapters` (method) | 171 |
| | `NunchakuLoraMixin.set_lora_strength` | 102 |
| | `NunchakuLoraMixin.reset_lora` | 177 |
| | `NunchakuLoraMixin.delete_adapters` | 190 |
| | `load_lora` (module function) | 440 |
| | `set_lora_strength` (module function) | 476 |
| | `reset_lora` (module function) | 500 |
| | `resolve_lora_name` | 498 |
| | `transformer_adapter_weights` | 580 |
| | `set_active_adapters` | 628 |
| | `active_lora_names` | 660 |
| | `active_lora_entries` | 670 |
| | `ensure_lora_runtime` | 676 |
| | `ensure_lora_base_state` | 690 |
| | `recompose_loras` | 758 |
| `src/nunchaku_lite/linear.py` | `DenseRuntimeLoraLinear.forward` | 34 |
| | `SVDQW4A4Linear.quantize` | 188 |
| | `SVDQW4A4Linear.forward_quant` | 196 |
| | `SVDQW4A4Linear.forward` | ~226 |
