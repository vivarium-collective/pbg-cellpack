# pbg-cellpack

Process-bigraph wrapper for [cellPACK](https://github.com/mesoscope/cellpack), a mesoscale molecular packing tool that assembles computational models of biological environments by packing molecules according to recipes.

## Installation

```bash
git clone <repo-url>
cd pbg-cellpack
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick Start

```python
from process_bigraph import allocate_core
from pbg_cellpack import CellPackStep

core = allocate_core()
core.register_link('CellPackStep', CellPackStep)

recipe = {
    'version': '1.0.0',
    'format_version': '2.0',
    'name': 'example',
    'bounding_box': [[0, 0, 0], [500, 500, 500]],
    'objects': {
        'base': {'type': 'single_sphere', 'place_method': 'jitter',
                 'jitter_attempts': 10, 'packing_mode': 'random',
                 'max_jitter': [1, 1, 1]},
        'protein': {'type': 'single_sphere', 'inherit': 'base',
                    'color': [0.3, 0.6, 0.9], 'radius': 30},
    },
    'composition': {
        'space': {'regions': {'interior': ['A']}},
        'A': {'object': 'protein', 'count': 50},
    },
}

step = CellPackStep(config={'recipe': recipe}, core=core)
result = step.update({'seed': 42})

print(f"Packed {result['n_packed']} molecules")
print(f"Packing fraction: {result['packing_fraction']:.3f}")
```

## API Reference

### CellPackStep

A process-bigraph `Step` that runs a cellPACK packing simulation.

| Config Key | Type | Default | Description |
|---|---|---|---|
| `recipe` | dict | `{}` | cellPACK v2 recipe (objects, composition, bounding_box) |
| `place_method` | str | `'jitter'` | Placement algorithm (`'jitter'` or `'spheresSST'`) |
| `inner_grid_method` | str | `'trimesh'` | Grid computation method |
| `use_periodicity` | bool | `False` | Enable periodic boundary conditions |

| Port | Direction | Type | Description |
|---|---|---|---|
| `seed` | input | integer | Random seed for reproducible packing |
| `positions` | output | list | [[x,y,z], ...] packed molecule positions |
| `radii` | output | list | Radius of each packed molecule |
| `ingredient_names` | output | list | Ingredient name per molecule |
| `ingredient_colors` | output | list | RGB color per molecule |
| `n_packed` | output | integer | Total molecules placed |
| `packing_fraction` | output | float | Volume fraction occupied |

### make_packing_document

Factory function that builds a composite document dict wiring `CellPackStep` with a RAM emitter.

```python
from pbg_cellpack import make_packing_document

doc = make_packing_document(recipe=recipe, place_method='jitter', seed=42)
```

## Architecture

cellPACK is a stateless packing algorithm: given a recipe and seed, it produces a deterministic molecular configuration. The wrapper maps this as a process-bigraph **Step**:

```
recipe (config) + seed (input) → CellPackStep → positions, radii, names, colors (outputs)
```

Internally, the Step:
1. Writes the recipe dict to a temporary JSON file
2. Loads it through cellPACK's `RecipeLoader` and `ConfigLoader`
3. Creates an `Environment`, builds the spatial grid, and runs packing
4. Extracts packed object positions, radii, names, and colors from the result

## Demo

Generate an interactive HTML report showcasing three advanced cellPACK features:

```bash
python demo/demo_report.py
```

The three configurations demonstrate:
1. **Gradient-Biased Packing** — Exponential spatial gradients (X/Y) drive non-uniform molecular placement, mimicking morphogen fields
2. **Polydisperse Size Distributions** — Continuous (uniform 15-30 nm) and discrete (30/35/40/45/50 nm) radius distributions via `size_options`
3. **Receptor-Ligand Partner Binding** — `closePartner` mode with 70% binding probability places ligands near receptors

The report (`demo/report.html`) includes:
- Interactive 3D sphere viewers (Three.js InstancedMesh, orbit controls)
- Feature-specific Plotly charts: spatial position distributions, radius rank plots, ligand-receptor distance histograms, volume fraction pies
- Colored bigraph-viz architecture diagrams
- Collapsible PBG composite document trees

## Tests

```bash
pytest tests/ -v
```
