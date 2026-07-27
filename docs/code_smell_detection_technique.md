# Code Smell Detection Technique — Reliable & Repeatable

> **Companion**: [Logic Mapping Technique](logic_mapping_technique.md) — trace
> the full code flow before debugging. The logic map is a prerequisite for the
> L2/L3 layers below: you can't cross-reference what you haven't mapped.

> **Problem**: Each manual code pass keeps finding *new* smells because there's no
> systematic baseline. The reviewer notices whatever catches their eye, fixes it,
> and the fix itself can cascade (e.g., removing a function makes its `import`
> dead). There's no way to know if you've found "everything" or just "what you
> happened to notice this time."

> **Solution**: A 4-layer technique that combines automated tools (for mechanical
> issues), a per-file checklist (for semantic issues), cross-reference analysis
> (for wiring gaps), and a fix-then-recheck loop (for cascading side effects).

---

## Why Smells Keep Appearing

| Root Cause | Example | Layer That Catches It |
|---|---|---|
| Ad-hoc review — no checklist | Missed stale comment referencing deleted function | L2: Per-file checklist |
| Fixes cascade — removing code orphans imports | Removing `_strip_markdown_syntax()` made `import re` dead | L4: Recheck loop |
| Semantic bugs invisible to linters | `OutputFormat.DOCLANG` in enum but missing from `_EXPORTERS` dispatch table | L3: Cross-reference |
| Style noise drowns real issues | 118 ruff findings (Optional→X\|None) hide 1 real bug | L1: Tiered filtering |

---

## L1 — Automated Tool Baseline (Mechanical Issues)

Run three tools. Pyflakes and vulture are the signal; ruff is for style.

### Setup (one-time)

```bash
python3 -m venv /tmp/lint-env
/tmp/lint-env/bin/pip install ruff pyflakes vulture
```

### Run (every pass)

```bash
API_DIR=<path-to-package>

# 1. Pyflakes — unused imports, undefined names (zero false positives)
/tmp/lint-env/bin/pyflakes $API_DIR

# 2. Vulture — dead code (high false-positive rate, filter aggressively)
/tmp/lint-env/bin/vulture $API_DIR --min-confidence 80

# 3. Ruff — style + some bug detection (filter to actionable rules only)
/tmp/lint-env/bin/ruff check \
  --select F,E9,PLC,PLE,PLR,PLW,B,SIM,RET,ARG \
  --ignore PLR0913,PLR2004,SIM117 \
  $API_DIR
```

### Rule Tier Filtering (critical — don't drown in noise)

Ruff's `--select ALL` produces 100+ findings that are mostly style preferences.
Filter to **actionable** rule groups only:

| Rule Group | Catches | Action |
|---|---|---|
| `F` | Pyflakes-equivalent (unused imports, undefined names) | Always fix |
| `E9` | Syntax errors, indentation | Always fix |
| `PLE` / `PLR` / `PLW` | Pylint errors/warnings (logic bugs, code smells) | Review each |
| `B` | Bugbear (common Python pitfalls) | Review each |
| `SIM` | Simplifications (collapsible ifs, ternary opportunities) | Fix if clear |
| `ARG` | Unused arguments | Review — often false positive in lambdas/callbacks |

**Ignore** (style noise that doesn't affect correctness):
- `UP045` (Optional → `X | None`) — cosmetic, breaks Python <3.10 compat
- `D*` (docstring formatting) — cosmetic
- `TID252` (relative vs absolute imports) — project convention
- `TC*` (type-checking block moves) — micro-optimization
- `COM812` (trailing commas) — cosmetic

### Vulture False-Positive Filtering

Vulture can't see:
- Pydantic `BaseModel` fields (used via serialization, not direct access)
- `enum.Enum` members (used via `OutputFormat.MARKDOWN` or iteration)
- FastAPI route functions (used via `@router.post()` decorator)
- Dataclass fields (used via serialization)

**Rule**: Only act on vulture findings for standalone functions, standalone
variables, or imports — never on class fields, enum members, or decorated
functions.

---

## L2 — Per-File Semantic Checklist

Apply this checklist to **every file**, every pass. This is the layer that
catches what linters can't.

### Import Section
- [ ] Every `import` is used somewhere in the file (grep each module name)
- [ ] No `from X import *` (star imports)
- [ ] Imports are grouped: stdlib → third-party → local (consistent style)
- [ ] `from __future__ import annotations` is present if using `X | Y` type hints

### Comments & Docstrings
- [ ] Every comment describes something that **still exists** in the code
      (grep for any function/class/variable named in comments — does it exist?)
- [ ] Every docstring is the **first statement** in its function/class/module
      (not preceded by setup calls — Python won't recognize it as `__doc__`)
- [ ] No commented-out code blocks
- [ ] Comments don't restate what the code already says (remove redundant ones)

### Functions & Methods
- [ ] Every function has a docstring (or is a trivial one-liner with an obvious name)
- [ ] `except Exception` blocks use `logger.exception()` (includes traceback),
      not `logger.error()` (doesn't) — UNLESS the exception is expected/benign
- [ ] No bare `except:` (always catch `Exception` at minimum)
- [ ] Return types match annotations (verify by reading, not just trusting)
- [ ] No function parameter is unused (if it's a callback signature, prefix with `_`)

### Data Structures
- [ ] Every `enum.Enum` member is **wired** — referenced in dispatch tables,
      conditionals, or serialization. An enum member with no consumer is dead code.
- [ ] Every `dataclass`/`BaseModel` field is **populated** somewhere — either by
      a constructor call, `setattr`, or Pydantic validation. A field nobody writes
      to is dead schema.
- [ ] Every dict constant (dispatch table, mapping) has matching **forward and
      reverse** coverage — if there's a `_FIELD_MAP` and an `_EXPORTERS` table,
      every key in one must appear in the other.
- [ ] Default mutable arguments use `Field(default_factory=...)` or
      `field(default_factory=...)`, never `[]` or `{}`

### Error Handling
- [ ] Error type constants (`ERR_FETCH = "FetchError"`) are **all used** — grep
      each constant name
- [ ] Error responses include the filename and processing_time (consistency)
- [ ] No `except Exception: pass` (swallowed errors)

---

## L3 — Cross-Reference Analysis (The Key Technique)

This is the technique that catches the bugs linters miss. It works by
**systematically verifying that every "definition" has a corresponding "usage."**

### Enum ↔ Dispatch Table Cross-Check

```python
# Given:
class OutputFormat(str, enum.Enum):
    MARKDOWN = "markdown"
    JSON = "json"
    TEXT = "text"
    HTML = "html"
    DOCTAGS = "doctags"
    DOCLANG = "doclang"      # ← is this in _EXPORTERS?

_EXPORTERS = {
    OutputFormat.MARKDOWN: ...,
    OutputFormat.JSON: ...,
    # ... is DOCLANG here?
}
```

**Technique**: For every `Enum` in the codebase, list all members. For every
dispatch table (`dict[EnumX, ...]`), list all keys. Diff the two lists.
Any enum member not in the dispatch table is either:
- Intentionally unsupported (should produce a user-facing error) → verify the
  error path handles it
- A wiring gap (bug) → fix it

**Automation**:
```bash
# Find all Enum classes and their members
grep -rn "class.*Enum" $API_DIR --include="*.py"
# Find all dispatch tables
grep -rn "dict\[Output" $API_DIR --include="*.py"
# Manually diff member list vs key list
```

### Constant ↔ Usage Cross-Check

```python
# Given error constants:
ERR_IMAGE_LOAD = "ImageLoadError"
ERR_LAYOUT_DETECTION = "LayoutDetectionError"
ERR_EXPORT = "ExportError"
ERR_FETCH = "FetchError"
ERR_DECODE = "DecodeError"

# Each must appear in at least one error_response() call or except block.
```

**Technique**: For every module-level constant (`UPPER_CASE`), grep for its
name across the entire package. If it only appears at its definition site,
it's dead code.

```bash
# List all UPPER_CASE constants
grep -rn "^[A-Z_]* =" $API_DIR --include="*.py"
# For each, check if it's used elsewhere
```

### Schema Field ↔ Population Cross-Check

```python
# Given response model fields:
class ExportDocumentResponse(BaseModel):
    md_content: Optional[str] = None
    json_content: Optional[DoclingDocument] = None
    # ...
    doclang_content: Optional[str] = None   # ← is this ever set?

# Given the ExportResult dataclass:
class ExportResult:
    doclang_content: Optional[str] = None   # ← is this in _FIELD_MAP?

# Given dispatch table:
_FIELD_MAP = {
    # ... is DOCLANG here?
}
```

**Technique**: For every field in a response/dataclass model, trace the
write path:
1. Is the field in the `_FIELD_MAP` / setter dispatch? → If no, it's never populated.
2. Is the `_FIELD_MAP` entry wired to an exporter? → If no, the field stays `None`.
3. Is the field copied from `ExportResult` to `ExportDocumentResponse`? → If no,
   the conversion is incomplete.

### Comment ↔ Symbol Cross-Check

```python
# Given comment:
# "Registered in _build_exporters() below to keep module-level clean."
# Does _build_exporters() exist?
```

**Technique**: For every comment that names a function, class, variable, or
file, grep for that name. If it doesn't exist, the comment is stale.

```bash
# Extract named symbols from comments
grep -rn "#.*_build_\|#.*Registered in\|#.*See .*\.py" $API_DIR --include="*.py"
```

---

## L4 — Fix-Then-Recheck Loop

Every fix can cascade. After applying fixes, **re-run L1 before declaring done.**

### Cascade Scenarios

| Fix Applied | Possible Cascade | Recheck |
|---|---|---|
| Removed a function | Its `import` is now unused | `pyflakes` |
| Removed a constant | An `if ERR_X` branch is now dead | `vulture` |
| Changed an enum | Dispatch table key is now orphaned | `ruff F841` |
| Removed a parameter | Caller still passes it | `ruff F811` |
| Changed a return type | Caller's unpacking breaks | Manual trace |

### Procedure

```
1. Run L1 (pyflakes + vulture + ruff) → record baseline
2. Run L2 (per-file checklist) → record findings
3. Run L3 (cross-reference) → record findings
4. Apply ALL fixes from L2 + L3
5. Re-run L1 → check for NEW findings introduced by fixes
6. If new findings → fix them → re-run L1 → repeat until clean
7. Run smoke test → verify behavioral correctness
```

**Rule**: Never declare "done" after a fix without re-running L1. The recheck
is what catches cascading dead imports, orphaned constants, and broken
references.

---

## Quick-Start: One-Command Audit

```bash
#!/bin/bash
# audit.sh — run this before every "code smell pass"
API_DIR="${1:?Usage: audit.sh <package-dir>}"
LINT=/tmp/lint-env/bin

echo "===== L1: AUTOMATED TOOLS ====="
echo "--- pyflakes ---"
$LINT/pyflakes "$API_DIR" 2>&1 | head -30
echo "--- vulture (conf>=80) ---"
$LINT/vulture "$API_DIR" --min-confidence 80 2>&1 | head -30
echo "--- ruff (actionable only) ---"
$LINT/ruff check --select F,E9,PLC,PLE,PLR,PLW,B,SIM,RET \
  --ignore PLR0913,PLR2004,SIM117,ARG002,ARG003,ARG005 \
  "$API_DIR" 2>&1 | head -50

echo ""
echo "===== L3: CROSS-REFERENCE CHECKS ====="
echo "--- Enum members vs dispatch tables ---"
grep -rn "class.*enum.Enum" "$API_DIR" --include="*.py"
echo "--- Upper-case constants ---"
grep -rn "^[A-Z_]\{3,\} =" "$API_DIR" --include="*.py" | grep -v "^.*:#"
echo "--- Comments naming symbols ---"
grep -rn "#.*_[a-z]" "$API_DIR" --include="*.py" | grep -v "coding:\|noqa\|type:\|pragma"
```

---

## What This Technique Caught (Historical Log)

| Pass | Issue Found | Layer | How |
|---|---|---|---|
| 1 | `app.py` docstring after `setup_logging()` — unreachable as `__doc__` | L2 | Docstring-first rule |
| 1 | `service.py` stale comment naming `_build_exporters()` — function doesn't exist | L3 | Comment↔Symbol cross-check |
| 2 | `OutputFormat.DOCLANG` missing from `_EXPORTERS` + `_FIELD_MAP` | L3 | Enum↔Dispatch cross-check |
| 2 | TEXT format used regex `_strip_markdown_syntax()` instead of `doc.export_to_text()` | L2 | Reinvented-wheel check |
| 2 | `import re` dead after removing `_strip_markdown_syntax()` | L4 | Recheck loop (pyflakes) |
| 2 | `need_md` checked TEXT but TEXT no longer uses cached_md | L4 | Recheck loop (manual) |
