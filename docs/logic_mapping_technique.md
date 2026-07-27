Logic Mapping Technique — Trace-Through-Before-You-Build

> **Problem**: When you start debugging or building inside an unfamiliar
> codebase, you patch symptoms instead of fixing root causes. Each "fix"
> breaks something else because you don't know the full call chain. You
> waste cycles guessing at data flow.

> **Solution**: Before writing or changing any code, trace through every
> function and data flow end-to-end to build a **logic map**. Then test each
> step with **real scraped data** to verify your understanding. Only then
> do you build — modularly, borrowing existing functions.

---

## Why This Works

| Without Logic Mapping                     | With Logic Mapping                           |
| ----------------------------------------- | -------------------------------------------- |
| Guess at data flow, patch symptoms        | Know exact call chain, fix root cause        |
| Break unseen callers with each fix        | See all callers before changing a signature  |
| Debug in a tight loop (slow, frustrating) | Debug once at the step level (fast, precise) |
| Reinvent functions that already exist     | Borrow and reuse existing utilities          |

The logic map is the **single source of truth** for understanding. Every bug
fix, every feature, every test references it. When you find something the map
doesn't cover, you update the map first, then proceed.

---

## The 4-Phase Workflow

### Phase 1 — Trace: Build the Logic Map

Goal: Understand the **complete** flow from entry point to exit, function by
function, with exact file/line references.

**Procedure**:

1. **Identify the entry point(s)** — the HTTP handler, CLI command, event
   listener, or public API function where the flow starts.
2. **Follow the call chain** — for each function, record:

   - File path and line number
   - Function signature (inputs → outputs)
   - What it calls next (the next function in the chain)
   - What data it produces or transforms
   - What side effects it has (logging, metrics, DB writes, state mutation)
3. **Trace ALL branches** — don't just follow the happy path. For every
   `if/else`, `try/except`, callback registration, and event handler, trace
   what happens on each branch. Mark dead branches (never called).
4. **Map data contracts** — what data shape enters each function, and what
   shape leaves? This catches type mismatches and missing fields that cause
   silent failures.
5. **Identify the exit point(s)** — the HTTP response, return value, persisted
   record, or emitted event where the flow ends.

**Output format** — a logic map document with:

```text
Entry Point → [function A (file:line)] → [function B (file:line)] → ... → Exit Point

With ASCII flow diagrams showing branching, parallel paths, and side effects.
```

**Example** (from Tier 2 metrics pipeline):

```
User request
    |
    v
[Proxy] --> litellm.acompletion()
    |              |
    |              v
    |        [Router] picks deployment
    |              |
    |              v
    |        [async_pre_call_deployment_hook]
    |              |
    |              v
    |        (2) _inc_deployment_in_progress  <-- INC gauge
    |              |
    |              v
    |        [Provider HTTP call]
    |              |
    |         +----+----+
    |         |         |
    |         v         v
    |    (3a) success  (3b) failure
    |         |         |
    |         v         v
    |    gauge.dec()  gauge.dec()
```

### Phase 2 — Test: Verify Each Step Against Real Data

Goal: Prove your logic map is **correct** by testing each step with real data
scraped from the live system.

**Procedure**:

1. **Port-forward access** to the live system (Prometheus, the service itself,
   databases) so you can query and scrape without deploying test code.
2. **Scrape real data** at each step of the flow:

   - Query Prometheus for the metrics your code produces/consumes
   - Capture the raw `/metrics/` output from the service pod
   - Capture endpoint responses (the API output your code generates)
   - Save everything to text/JSON files for offline analysis
3. **Save scraped data** in a `live-data/` directory alongside the logic map:

   ```text
   docs/dashboard-plan/live-data/
   ├── 01-all-metrics-instant.json      # All metrics at a point in time
   ├── 02-gauge-instant.json            # Specific gauge values
   ├── 03-gauge-range-1h.json           # Time series over 1 hour
   ├── 04-request-rate-range-1h.json
   ├── 07-limit-instant.json
   ├── 11-raw-pod-metrics.txt           # Raw /metrics/ endpoint output
   └── 12-endpoint-response-1h.json     # API endpoint response
   ```
4. **Test each step** — for each function in the logic map, load the relevant
   scraped data and verify:

   - Does the input data match what the function expects?
   - Does the output data match what the function produces?
   - Are there gaps (metrics that should exist but don't)?
   - Are there mismatches (labels that don't match between INC and DEC)?

   ```python
   # Load scraped data for offline testing
   import json
   with open("live-data/03-gauge-range-1h.json") as f:
       gauge_data = json.load(f)
   # Verify: does this match what _inc_deployment_in_progress produces?
   ```

**Why port-forward + scrape instead of live testing**: Live testing requires
deploying code, generating traffic, and waiting for Prometheus scrape cycles
(30s). Port-forwarding lets you query the live system instantly, save the data
once, and test offline repeatedly without re-scraping.

### Phase 3 — Build: Modular Vertical Slice Architecture

Goal: Implement the feature step-by-step, each step independently testable,
borrowing existing functions from the current codebase.

**Procedure**:

1. **Identify reusable functions** — scan the existing codebase for functions
   that already do what you need (parsing, validation, serialization, metric
   queries). Import and reuse them. Do NOT reinvent.
2. **Structure as vertical slices** — each slice is a self-contained feature
   that touches all layers (route → service → schema) but is narrow in scope:

   ```text
   feature_slice/
   ├── __init__.py
   ├── routes.py        # HTTP handler (thin)
   ├── service.py       # Business logic (borrows from codebase)
   └── schema.py        # Request/response models
   ```
3. **Build one slice at a time** — implement, test against scraped data, then
   move to the next slice. Each slice should be deployable and testable
   independently.
4. **Share infrastructure via a `_core/` package** — generic utilities (config,
   clients, auth, health) live in `_core/` and are imported by all slices:

   ```text
   _core/
   ├── config.py        # Env vars, single source of truth
   ├── prometheus.py    # Shared Prometheus client (borrowed)
   └── health/
       └── routes.py    # /health, /ready

   feature_a/           # Vertical slice A
   ├── routes.py
   ├── service.py
   └── schema.py

   feature_b/           # Vertical slice B (reuses _core/)
   ├── routes.py
   ├── service.py
   └── schema.py
   ```

### Phase 4 — Verify: Re-Test Against the Logic Map

Goal: Confirm the implementation matches the logic map and all steps produce
expected values.

**Procedure**:

1. Re-run the step tests from Phase 2 against the new implementation.
2. Verify each function in the logic map is exercised by at least one test.
3. Verify no new dead branches were introduced (cross-check with the
   [code smell audit technique](code_smell_detection_technique.md)).
4. Update the logic map if the implementation revealed flows the map missed.

---

## The Prompt Template

When starting a new feature in an unfamiliar codebase, give this prompt:

```text
Before you continue, I want you to map out the logic of implementation of
[FEATURE/TIER] in functions and flows, and test each step so the values are
expected — by port-forwarding access to the [PROMETHEUS / SERVICE] scraped
data, and save the scraped data somewhere for easy text file loading.

Then I want you to build this step by step in modularized vertical slice
architecture, and it should borrow as many functions as possible from the
current [CODEBASE] code base.

Save this methodology in the folder in [DOCS_PATH].
```

**Fill in**:

- `[FEATURE/TIER]` — what you're building (e.g., "tier 2 per-model metrics")
- `[PROMETHEUS / SERVICE]` — the live system to scrape data from
- `[CODEBASE]` — the existing codebase to borrow functions from
- `[DOCS_PATH]` — where to save the methodology and logic map

### What the prompt triggers

| Prompt phrase                                  | Technique phase | What happens                            |
| ---------------------------------------------- | --------------- | --------------------------------------- |
| "map out the logic ... in functions and flows" | Phase 1: Trace  | Build logic map with file:line refs     |
| "test each step so the values are expected"    | Phase 2: Test   | Verify each function against real data  |
| "port-forwarding access ... save scraped data" | Phase 2: Test   | Scrape live data to`live-data/` files |
| "build step by step in modularized VSA"        | Phase 3: Build  | One slice at a time,`_core/` + slices |
| "borrow as many functions as possible"         | Phase 3: Build  | Reuse existing codebase functions       |
| "save this methodology in [path]"              | Documentation   | Persist the technique for reuse         |

---

## Deliverables Checklist

When applying this technique, the following artifacts should be produced:

- [ ] **Logic map document** (`*-LOGIC-MAP.md`) — complete function-level flow
  with file:line references and ASCII diagrams
- [ ] **Live data directory** (`live-data/`) — scraped JSON/text files for
  offline testing, with numbered prefixes showing scrape order
- [ ] **Testing methodology document** (`*-TESTING-METHODOLOGY.md`) — how to
  port-forward, scrape, and load data for offline testing
- [ ] **VSA plan document** (`*-VSA-PLAN.md`) — slice breakdown, which
  functions to borrow, `_core/` vs slice boundary
- [ ] **Implementation** — code built one slice at a time, each tested
  against scraped data before moving to the next

---

## Worked Examples

### Example 1: PaddleX api_compat (Docling compatibility layer)

- **Logic map**: `PaddleX/deploy/hps/docs/api_compat_methodology.md`
  - Entry: `POST /v1/convert/file` → `routes.py:convert_file()`
  - Flow: route → `service.py:convert_image()` → `_core/inference.py:run_layout_detection()` → `converter/service.py:convert()` → export
  - Exit: `ConvertDocumentResponse` with 6 output formats

### Example 2: LiteLLM Tier 2 per-model metrics dashboard

- **Logic map**: `oicm-litellm-layer/docs/dashboard-plan/TIER2-LOGIC-MAP.md`
  - Entry: `proxy_server.py:chat_completion()` → `router.acompletion()`
  - Flow: `async_pre_call_deployment_hook` (INC gauge) → provider call → `success_handler`/`failure_handler` (DEC gauge) → Prometheus scrape → `get_per_model_metrics()` → endpoint response
  - Live data: `live-data/01-13-*.json` scraped from production Prometheus
  - Testing: `TIER2-TESTING-METHODOLOGY.md` documents port-forward + scrape commands
  - Found bug: INC gauge was in dead method `async_log_pre_api_call` (never called) — moved to `async_pre_call_deployment_hook`

### Key Insight from Both Examples

The logic map caught bugs that were invisible without tracing:

- **api_compat**: Missing `DOCLANG` exporter in dispatch table — only visible
  when you map every enum member to its consumer
- **LiteLLM Tier 2**: INC gauge in a dead method — only visible when you trace
  the full call chain and find the method has zero call sites

Both bugs required **understanding the full flow** before they became visible.
Neither could be found by reading a single file in isolation.
