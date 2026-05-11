"""cellPACK composite documents + composite-spec discovery.

Two flavors of composite construction live in this package:

1. **Hand-coded factory** -- `make_packing_document(recipe=...)` builds a
   PBG state-dict programmatically for callers that want full control
   over the recipe + wiring. Preserves the original pbg-cellpack API.

2. **Declarative `*.composite.yaml`** -- sibling files in this directory
   follow the pbg-superpowers composite-spec convention.
   `build_composite()` loads one by name and instantiates
   `process_bigraph.Composite` with parameter substitution. The
   dashboard's composite explorer discovers these automatically once
   the package is installed in a workspace.

Both flavors are equivalent -- pick the one that fits your use case.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any

import yaml
from process_bigraph import allocate_core
from process_bigraph.emitter import RAMEmitter

from pbg_cellpack.processes import CellPackStep


# ---------------------------------------------------------------------------
# Hand-coded composite factory (legacy / programmatic API)
# ---------------------------------------------------------------------------


def register_cellpack(core=None):
    """Return a core with CellPackStep, the RAM emitter, and the cellPACK
    Visualization Step registered."""
    if core is None:
        core = allocate_core()
    core.register_link('CellPackStep', CellPackStep)
    core.register_link('ram-emitter', RAMEmitter)
    # Register Visualization Steps so composites can wire them by name.
    try:
        from pbg_cellpack.visualizations import PackingPlots
        core.register_link('PackingPlots', PackingPlots)
    except ImportError:
        # pbg-superpowers may not be installed in minimal environments;
        # the legacy factory still works without the Viz Step.
        pass
    return core


def make_packing_document(
    recipe,
    place_method='jitter',
    seed=0,
):
    """Build a process-bigraph composite document for a cellPACK packing.

    Parameters
    ----------
    recipe : dict
        cellPACK v2 recipe specification (objects, composition, bounding_box).
    place_method : str
        Placement algorithm (``'jitter'`` or ``'spheresSST'``).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        A composite document dict ready for ``Composite({'state': doc})``.
    """
    return {
        'packer': {
            '_type': 'step',
            'address': 'local:CellPackStep',
            'config': {
                'recipe': recipe,
                'place_method': place_method,
            },
            'inputs': {
                'seed': ['stores', 'seed'],
            },
            'outputs': {
                'positions': ['stores', 'positions'],
                'radii': ['stores', 'radii'],
                'ingredient_names': ['stores', 'ingredient_names'],
                'ingredient_colors': ['stores', 'ingredient_colors'],
                'n_packed': ['stores', 'n_packed'],
                'packing_fraction': ['stores', 'packing_fraction'],
            },
        },
        'stores': {
            'seed': seed,
            'positions': [],
            'radii': [],
            'ingredient_names': [],
            'ingredient_colors': [],
            'n_packed': 0,
            'packing_fraction': 0.0,
        },
        'emitter': {
            '_type': 'step',
            'address': 'local:ram-emitter',
            'config': {
                'emit': {
                    'n_packed': 'integer',
                    'packing_fraction': 'float',
                },
            },
            'inputs': {
                'n_packed': ['stores', 'n_packed'],
                'packing_fraction': ['stores', 'packing_fraction'],
            },
        },
    }


# ---------------------------------------------------------------------------
# Declarative composite-spec loader (*.composite.yaml)
# ---------------------------------------------------------------------------

_COMPOSITES_DIR = Path(__file__).parent

_FULL_PLACEHOLDER = re.compile(r"^\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}$")
_INLINE_PLACEHOLDER = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _cast(value: Any, declared_type: str | None) -> Any:
    if declared_type is None:
        return value
    if declared_type == "float":
        return float(value)
    if declared_type == "int":
        return int(value)
    if declared_type in ("string", "str"):
        return str(value)
    if declared_type == "bool":
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes")
        return bool(value)
    return value


def _substitute(state: Any, params: dict, overrides: dict) -> Any:
    if isinstance(state, dict):
        return {k: _substitute(v, params, overrides) for k, v in state.items()}
    if isinstance(state, list):
        return [_substitute(v, params, overrides) for v in state]
    if isinstance(state, str):
        m = _FULL_PLACEHOLDER.match(state)
        if m:
            pname = m.group(1)
            pdef = params.get(pname, {})
            raw = overrides.get(pname, pdef.get("default"))
            return _cast(raw, pdef.get("type"))
        if _INLINE_PLACEHOLDER.search(state):
            return _INLINE_PLACEHOLDER.sub(
                lambda mm: str(overrides.get(mm.group(1), params.get(mm.group(1), {}).get("default", ""))),
                state,
            )
    return state


def list_composite_specs() -> list[str]:
    """Return short names of every `*.composite.yaml` shipped in this package."""
    out: list[str] = []
    for path in sorted(_COMPOSITES_DIR.glob("*.composite.yaml")):
        out.append(path.name[: -len(".composite.yaml")])
    return out


def load_composite_spec(name: str) -> dict:
    """Load and parse a named composite spec. `name` is the stem (no suffix)."""
    path = _COMPOSITES_DIR / f"{name}.composite.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"composite spec not found: {path}")
    return yaml.safe_load(path.read_text())


def build_composite(name: str, *, overrides: dict | None = None, core=None):
    """Load a *.composite.yaml by name and instantiate process_bigraph.Composite.

    overrides: parameter overrides (keys must match spec.parameters)
    core:      optional pre-built core; otherwise register_cellpack() is used
    """
    from process_bigraph import Composite

    spec = load_composite_spec(name)
    if not isinstance(spec, dict) or "state" not in spec or "name" not in spec:
        raise ValueError(f"composite '{name}' missing required keys (name, state)")

    if core is None:
        core = register_cellpack()

    params = spec.get("parameters") or {}
    state = _substitute(spec.get("state") or {}, params, overrides or {})
    return Composite({"state": state}, core=core)
