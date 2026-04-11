"""Demo: cellPACK multi-configuration packing report with 3D viewers.

Runs three distinct molecular packing simulations (crowded cytoplasm,
multi-size sphere packing, dense small-molecule packing), generates
interactive 3D sphere viewers with Three.js, Plotly charts, colored
bigraph-viz architecture diagrams, and navigatable PBG document trees
— all in a single self-contained HTML.
"""

import json
import math
import os
import base64
import subprocess
import tempfile
import time

import numpy as np
from process_bigraph import allocate_core
from pbg_cellpack.processes import CellPackStep
from pbg_cellpack.composites import make_packing_document


# ── Simulation Configs ──────────────────────────────────────────────

CONFIGS = [
    {
        'id': 'cytoplasm',
        'title': 'Crowded Cytoplasm',
        'subtitle': 'Multi-component molecular crowding in a cellular volume',
        'description': (
            'Four molecular species of different sizes (large complexes at '
            'r=40, medium enzymes at r=25, small metabolites at r=15, and tiny '
            'cofactors at r=8) are packed into a 600³ nm box simulating '
            'cytoplasmic crowding.  This demonstrates cellPACK\'s ability to '
            'handle polydisperse mixtures with realistic volume fractions.'
        ),
        'config': {
            'recipe': {
                'version': '1.0.0',
                'format_version': '2.0',
                'name': 'cytoplasm',
                'bounding_box': [[0, 0, 0], [600, 600, 600]],
                'objects': {
                    'base': {
                        'type': 'single_sphere',
                        'place_method': 'jitter',
                        'jitter_attempts': 20,
                        'packing_mode': 'random',
                        'max_jitter': [1, 1, 1],
                    },
                    'complex': {
                        'type': 'single_sphere', 'inherit': 'base',
                        'color': [0.25, 0.41, 0.88], 'radius': 40,
                    },
                    'enzyme': {
                        'type': 'single_sphere', 'inherit': 'base',
                        'color': [0.13, 0.73, 0.49], 'radius': 25,
                    },
                    'metabolite': {
                        'type': 'single_sphere', 'inherit': 'base',
                        'color': [0.96, 0.65, 0.14], 'radius': 15,
                    },
                    'cofactor': {
                        'type': 'single_sphere', 'inherit': 'base',
                        'color': [0.85, 0.24, 0.31], 'radius': 8,
                    },
                },
                'composition': {
                    'space': {'regions': {
                        'interior': ['A', 'B', 'C', 'D'],
                    }},
                    'A': {'object': 'complex', 'count': 15},
                    'B': {'object': 'enzyme', 'count': 40},
                    'C': {'object': 'metabolite', 'count': 80},
                    'D': {'object': 'cofactor', 'count': 200},
                },
            },
        },
        'seed': 42,
        'camera': [900, 600, 900],
        'color_scheme': 'indigo',
    },
    {
        'id': 'vesicle',
        'title': 'Synaptic Vesicle Lumen',
        'subtitle': 'Uniform neurotransmitter packing in a small volume',
        'description': (
            'Hundreds of small neurotransmitter-sized spheres (r=5) are packed '
            'tightly into a 200³ nm box representing the interior of a '
            'synaptic vesicle.  A small number of larger transporter proteins '
            '(r=20) are interspersed.  This showcases cellPACK\'s efficiency '
            'with large molecule counts and tight packing.'
        ),
        'config': {
            'recipe': {
                'version': '1.0.0',
                'format_version': '2.0',
                'name': 'vesicle',
                'bounding_box': [[0, 0, 0], [200, 200, 200]],
                'objects': {
                    'base': {
                        'type': 'single_sphere',
                        'place_method': 'jitter',
                        'jitter_attempts': 20,
                        'packing_mode': 'random',
                        'max_jitter': [1, 1, 1],
                    },
                    'transporter': {
                        'type': 'single_sphere', 'inherit': 'base',
                        'color': [0.56, 0.27, 0.80], 'radius': 20,
                    },
                    'neurotransmitter': {
                        'type': 'single_sphere', 'inherit': 'base',
                        'color': [0.20, 0.78, 0.60], 'radius': 5,
                    },
                },
                'composition': {
                    'space': {'regions': {
                        'interior': ['A', 'B'],
                    }},
                    'A': {'object': 'transporter', 'count': 8},
                    'B': {'object': 'neurotransmitter', 'count': 500},
                },
            },
        },
        'seed': 7,
        'camera': [300, 200, 300],
        'color_scheme': 'emerald',
    },
    {
        'id': 'binary',
        'title': 'Binary Sphere Packing',
        'subtitle': 'Two-component size-ratio packing study',
        'description': (
            'A binary mixture of large (r=50) and small (r=15) spheres is '
            'packed into a 500³ nm box.  The 3.3:1 size ratio allows small '
            'spheres to fill the interstices between large spheres, a classic '
            'granular-packing problem.  This configuration is useful for '
            'studying packing efficiency as a function of size ratio.'
        ),
        'config': {
            'recipe': {
                'version': '1.0.0',
                'format_version': '2.0',
                'name': 'binary_packing',
                'bounding_box': [[0, 0, 0], [500, 500, 500]],
                'objects': {
                    'base': {
                        'type': 'single_sphere',
                        'place_method': 'jitter',
                        'jitter_attempts': 20,
                        'packing_mode': 'random',
                        'max_jitter': [1, 1, 1],
                    },
                    'large': {
                        'type': 'single_sphere', 'inherit': 'base',
                        'color': [0.96, 0.26, 0.40], 'radius': 50,
                    },
                    'small': {
                        'type': 'single_sphere', 'inherit': 'base',
                        'color': [0.15, 0.53, 0.87], 'radius': 15,
                    },
                },
                'composition': {
                    'space': {'regions': {
                        'interior': ['A', 'B'],
                    }},
                    'A': {'object': 'large', 'count': 25},
                    'B': {'object': 'small', 'count': 300},
                },
            },
        },
        'seed': 99,
        'camera': [750, 500, 750],
        'color_scheme': 'rose',
    },
]


# ── Simulation ──────────────────────────────────────────────────────

def run_simulation(cfg_entry):
    """Run a single cellPACK packing, returning results and wall-clock time."""
    core = allocate_core()
    core.register_link('CellPackStep', CellPackStep)

    t0 = time.perf_counter()
    step = CellPackStep(config=cfg_entry['config'], core=core)
    result = step.update({'seed': cfg_entry['seed']})
    runtime = time.perf_counter() - t0

    return result, runtime


# ── Analysis helpers ────────────────────────────────────────────────

def compute_ingredient_counts(result):
    """Return {ingredient_name: count} dict."""
    counts = {}
    for name in result['ingredient_names']:
        counts[name] = counts.get(name, 0) + 1
    return counts


def compute_nearest_neighbor_distances(positions):
    """Compute nearest-neighbor distance for each packed sphere."""
    pts = np.array(positions)
    n = len(pts)
    if n < 2:
        return []
    nn_dists = []
    for i in range(n):
        diffs = pts - pts[i]
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        dists[i] = np.inf
        nn_dists.append(float(np.min(dists)))
    return nn_dists


def compute_radial_distribution(positions, bb):
    """Compute distance of each sphere from box center."""
    pts = np.array(positions)
    center = np.array([(bb[0][i] + bb[1][i]) / 2 for i in range(3)])
    dists = np.sqrt(np.sum((pts - center) ** 2, axis=1))
    return [float(d) for d in dists]


# ── Bigraph diagram ────────────────────────────────────────────────

def generate_bigraph_image(cfg_entry):
    """Generate a colored bigraph-viz PNG for the composite document."""
    from bigraph_viz import plot_bigraph

    recipe = cfg_entry['config']['recipe']
    ingredient_names = list(recipe['objects'].keys())
    ingredient_names = [n for n in ingredient_names if n != 'base']

    doc = {
        'packer': {
            '_type': 'step',
            'address': 'local:CellPackStep',
            'config': {'place_method': 'jitter'},
            'inputs': {
                'seed': ['stores', 'seed'],
            },
            'outputs': {
                'positions': ['stores', 'positions'],
                'radii': ['stores', 'radii'],
                'n_packed': ['stores', 'n_packed'],
                'packing_fraction': ['stores', 'packing_fraction'],
            },
        },
        'stores': {},
        'emitter': {
            '_type': 'step',
            'address': 'local:ram-emitter',
            'config': {'emit': {
                'n_packed': 'integer',
                'packing_fraction': 'float',
            }},
            'inputs': {
                'n_packed': ['stores', 'n_packed'],
                'packing_fraction': ['stores', 'packing_fraction'],
            },
        },
    }

    node_colors = {
        ('packer',): '#6366f1',
        ('emitter',): '#8b5cf6',
        ('stores',): '#e0e7ff',
    }

    outdir = tempfile.mkdtemp()
    plot_bigraph(
        state=doc,
        out_dir=outdir,
        filename='bigraph',
        file_format='png',
        remove_process_place_edges=True,
        rankdir='LR',
        node_fill_colors=node_colors,
        node_label_size='16pt',
        port_labels=False,
        dpi='150',
    )
    png_path = os.path.join(outdir, 'bigraph.png')
    with open(png_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'data:image/png;base64,{b64}'


# ── HTML Report ─────────────────────────────────────────────────────

COLOR_SCHEMES = {
    'indigo': {'primary': '#6366f1', 'light': '#e0e7ff', 'dark': '#4338ca',
               'bg': '#eef2ff', 'accent': '#818cf8', 'text': '#312e81'},
    'emerald': {'primary': '#10b981', 'light': '#d1fae5', 'dark': '#059669',
                'bg': '#ecfdf5', 'accent': '#34d399', 'text': '#064e3b'},
    'rose': {'primary': '#f43f5e', 'light': '#ffe4e6', 'dark': '#e11d48',
             'bg': '#fff1f2', 'accent': '#fb7185', 'text': '#881337'},
}


def generate_html(sim_results, output_path):
    """Generate comprehensive self-contained HTML report."""

    sections_html = []
    all_js_data = {}
    pbg_docs = {}

    for idx, (cfg, (result, runtime)) in enumerate(sim_results):
        sid = cfg['id']
        cs = COLOR_SCHEMES[cfg['color_scheme']]
        recipe = cfg['config']['recipe']
        bb = recipe['bounding_box']

        n_packed = result['n_packed']
        pf = result['packing_fraction']

        # Ingredient breakdown
        ingr_counts = compute_ingredient_counts(result)
        # Get unique ingredient info (name -> color, radius)
        ingr_info = {}
        for i, name in enumerate(result['ingredient_names']):
            if name not in ingr_info:
                ingr_info[name] = {
                    'color': result['ingredient_colors'][i],
                    'radius': result['radii'][i],
                }

        # Analysis data
        nn_dists = compute_nearest_neighbor_distances(result['positions'])
        radial_dists = compute_radial_distribution(result['positions'], bb)

        # JS data for Three.js viewer
        all_js_data[sid] = {
            'positions': result['positions'],
            'radii': result['radii'],
            'colors': result['ingredient_colors'],
            'names': result['ingredient_names'],
            'bb': bb,
            'camera': cfg['camera'],
            'ingr_counts': ingr_counts,
            'ingr_info': {k: {'color': v['color'], 'radius': v['radius']}
                          for k, v in ingr_info.items()},
            'nn_dists': nn_dists,
            'radial_dists': radial_dists,
        }

        # Bigraph diagram
        print(f'  Generating bigraph diagram for {sid}...')
        bigraph_img = generate_bigraph_image(cfg)

        # PBG document
        pbg_docs[sid] = make_packing_document(
            recipe=recipe,
            place_method='jitter',
            seed=cfg['seed'],
        )

        # Volume stats
        bb_vol = np.prod(np.array(bb[1]) - np.array(bb[0]))
        bb_size = [bb[1][i] - bb[0][i] for i in range(3)]

        # Ingredient count summary
        ingr_summary = ' &middot; '.join(
            f'{name}: <strong>{count}</strong>'
            for name, count in sorted(ingr_counts.items())
        )

        section = f"""
    <div class="sim-section" id="sim-{sid}">
      <div class="sim-header" style="border-left: 4px solid {cs['primary']};">
        <div class="sim-number" style="background:{cs['light']}; color:{cs['dark']};">{idx+1}</div>
        <div>
          <h2 class="sim-title">{cfg['title']}</h2>
          <p class="sim-subtitle">{cfg['subtitle']}</p>
        </div>
      </div>
      <p class="sim-description">{cfg['description']}</p>

      <div class="metrics-row">
        <div class="metric"><span class="metric-label">Packed</span><span class="metric-value">{n_packed:,}</span></div>
        <div class="metric"><span class="metric-label">Species</span><span class="metric-value">{len(ingr_counts)}</span></div>
        <div class="metric"><span class="metric-label">Packing &phi;</span><span class="metric-value">{pf:.3f}</span></div>
        <div class="metric"><span class="metric-label">Box</span><span class="metric-value">{bb_size[0]:.0f}&sup3;</span><span class="metric-sub">nm</span></div>
        <div class="metric"><span class="metric-label">Box Vol</span><span class="metric-value">{bb_vol:.0e}</span><span class="metric-sub">nm&sup3;</span></div>
        <div class="metric"><span class="metric-label">Runtime</span><span class="metric-value">{runtime:.1f}s</span></div>
      </div>
      <p class="ingr-summary">{ingr_summary}</p>

      <h3 class="subsection-title">3D Molecular Packing Viewer</h3>
      <div class="viewer-wrap">
        <canvas id="canvas-{sid}" class="mesh-canvas"></canvas>
        <div class="viewer-info">
          <strong>{n_packed}</strong> molecules &middot;
          <strong>{len(ingr_counts)}</strong> species<br>
          Drag to rotate &middot; Scroll to zoom
        </div>
        <div class="legend-box" id="legend-{sid}"></div>
      </div>

      <h3 class="subsection-title">Packing Analysis</h3>
      <div class="charts-row">
        <div class="chart-box"><div id="chart-counts-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-radial-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-nn-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-volume-{sid}" class="chart"></div></div>
      </div>

      <div class="pbg-row">
        <div class="pbg-col">
          <h3 class="subsection-title">Bigraph Architecture</h3>
          <div class="bigraph-img-wrap">
            <img src="{bigraph_img}" alt="Bigraph architecture diagram">
          </div>
        </div>
        <div class="pbg-col">
          <h3 class="subsection-title">Composite Document</h3>
          <div class="json-tree" id="json-{sid}"></div>
        </div>
      </div>
    </div>
"""
        sections_html.append(section)

    # Navigation
    nav_items = ''.join(
        f'<a href="#sim-{c["id"]}" class="nav-link" '
        f'style="border-color:{COLOR_SCHEMES[c["color_scheme"]]["primary"]};">'
        f'{c["title"]}</a>'
        for c in [r[0] for r in sim_results])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>cellPACK Molecular Packing Report</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
       background:#fff; color:#1e293b; line-height:1.6; }}
.page-header {{
  background:linear-gradient(135deg,#f8fafc 0%,#eef2ff 50%,#fdf2f8 100%);
  border-bottom:1px solid #e2e8f0; padding:3rem;
}}
.page-header h1 {{ font-size:2.2rem; font-weight:800; color:#0f172a; margin-bottom:.3rem; }}
.page-header p {{ color:#64748b; font-size:.95rem; max-width:700px; }}
.nav {{ display:flex; gap:.8rem; padding:1rem 3rem; background:#f8fafc;
        border-bottom:1px solid #e2e8f0; position:sticky; top:0; z-index:100; }}
.nav-link {{ padding:.4rem 1rem; border-radius:8px; border:1.5px solid;
             text-decoration:none; font-size:.85rem; font-weight:600;
             transition:all .15s; }}
.nav-link:hover {{ transform:translateY(-1px); box-shadow:0 2px 8px rgba(0,0,0,.08); }}
.sim-section {{ padding:2.5rem 3rem; border-bottom:1px solid #e2e8f0; }}
.sim-header {{ display:flex; align-items:center; gap:1rem; margin-bottom:.8rem;
               padding-left:1rem; }}
.sim-number {{ width:36px; height:36px; border-radius:10px; display:flex;
               align-items:center; justify-content:center; font-weight:800; font-size:1.1rem; }}
.sim-title {{ font-size:1.5rem; font-weight:700; color:#0f172a; }}
.sim-subtitle {{ font-size:.9rem; color:#64748b; }}
.sim-description {{ color:#475569; font-size:.9rem; margin-bottom:1.5rem; max-width:800px; }}
.ingr-summary {{ font-size:.85rem; color:#64748b; margin-bottom:1rem; }}
.subsection-title {{ font-size:1.05rem; font-weight:600; color:#334155;
                     margin:1.5rem 0 .8rem; }}
.metrics-row {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr));
                gap:.8rem; margin-bottom:.8rem; }}
.metric {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
           padding:.8rem; text-align:center; }}
.metric-label {{ display:block; font-size:.7rem; text-transform:uppercase;
                 letter-spacing:.06em; color:#94a3b8; margin-bottom:.2rem; }}
.metric-value {{ display:block; font-size:1.3rem; font-weight:700; color:#1e293b; }}
.metric-sub {{ display:block; font-size:.7rem; color:#94a3b8; }}
.viewer-wrap {{ position:relative; background:#f1f5f9; border:1px solid #e2e8f0;
                border-radius:14px; overflow:hidden; margin-bottom:1rem; }}
.mesh-canvas {{ width:100%; height:500px; display:block; cursor:grab; }}
.mesh-canvas:active {{ cursor:grabbing; }}
.viewer-info {{ position:absolute; top:.8rem; left:.8rem; background:rgba(255,255,255,.92);
                border:1px solid #e2e8f0; border-radius:8px; padding:.5rem .8rem;
                font-size:.75rem; color:#64748b; backdrop-filter:blur(4px); }}
.viewer-info strong {{ color:#1e293b; }}
.legend-box {{ position:absolute; top:.8rem; right:.8rem; background:rgba(255,255,255,.92);
               border:1px solid #e2e8f0; border-radius:8px; padding:.6rem .8rem;
               font-size:.72rem; color:#475569; backdrop-filter:blur(4px); }}
.legend-item {{ display:flex; align-items:center; gap:.4rem; margin-bottom:.25rem; }}
.legend-dot {{ width:12px; height:12px; border-radius:50%; display:inline-block; }}
.charts-row {{ display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-bottom:1rem; }}
.chart-box {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; overflow:hidden; }}
.chart {{ height:280px; }}
.pbg-row {{ display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; margin-top:1rem; }}
.pbg-col {{ min-width:0; }}
.bigraph-img-wrap {{ background:#fafafa; border:1px solid #e2e8f0; border-radius:10px;
                     padding:1.5rem; text-align:center; }}
.bigraph-img-wrap img {{ max-width:100%; height:auto; }}
.json-tree {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
              padding:1rem; max-height:500px; overflow-y:auto; font-family:'SF Mono',
              Menlo,Monaco,'Courier New',monospace; font-size:.78rem; line-height:1.5; }}
.jt-key {{ color:#7c3aed; font-weight:600; }}
.jt-str {{ color:#059669; }}
.jt-num {{ color:#2563eb; }}
.jt-bool {{ color:#d97706; }}
.jt-null {{ color:#94a3b8; }}
.jt-toggle {{ cursor:pointer; user-select:none; color:#94a3b8; margin-right:.3rem; }}
.jt-toggle:hover {{ color:#1e293b; }}
.jt-collapsed {{ display:none; }}
.jt-bracket {{ color:#64748b; }}
.footer {{ text-align:center; padding:2rem; color:#94a3b8; font-size:.8rem;
           border-top:1px solid #e2e8f0; }}
@media(max-width:900px) {{
  .charts-row,.pbg-row {{ grid-template-columns:1fr; }}
  .sim-section,.page-header {{ padding:1.5rem; }}
}}
</style>
</head>
<body>

<div class="page-header">
  <h1>cellPACK Molecular Packing Report</h1>
  <p>Three mesoscale packing simulations wrapped as <strong>process-bigraph</strong>
  Steps using cellPACK. Each configuration demonstrates a distinct packing
  scenario with interactive 3D visualization and statistical analysis.</p>
</div>

<div class="nav">{nav_items}</div>

{''.join(sections_html)}

<div class="footer">
  Generated by <strong>pbg-cellpack</strong> &mdash;
  cellPACK + process-bigraph &mdash;
  Mesoscale Molecular Packing
</div>

<script>
const DATA = {json.dumps(all_js_data)};
const DOCS = {json.dumps(pbg_docs, indent=2)};

// ─── JSON Tree Viewer ───
function renderJson(obj, depth) {{
  if (depth === undefined) depth = 0;
  if (obj === null) return '<span class="jt-null">null</span>';
  if (typeof obj === 'boolean') return '<span class="jt-bool">' + obj + '</span>';
  if (typeof obj === 'number') return '<span class="jt-num">' + obj + '</span>';
  if (typeof obj === 'string') return '<span class="jt-str">"' + obj.replace(/</g,'&lt;') + '"</span>';
  if (Array.isArray(obj)) {{
    if (obj.length === 0) return '<span class="jt-bracket">[]</span>';
    if (obj.length <= 5 && obj.every(x => typeof x !== 'object' || x === null)) {{
      const items = obj.map(x => renderJson(x, depth+1)).join(', ');
      return '<span class="jt-bracket">[</span>' + items + '<span class="jt-bracket">]</span>';
    }}
    const id = 'jt' + Math.random().toString(36).slice(2,9);
    let html = '<span class="jt-toggle" onclick="toggleJt(\\'' + id + '\\')">&blacktriangledown;</span>';
    html += '<span class="jt-bracket">[</span> <span style="color:#94a3b8;font-size:.7rem;">' + obj.length + ' items</span>';
    html += '<div id="' + id + '" style="margin-left:1.2rem;">';
    obj.forEach((v, i) => {{ html += '<div>' + renderJson(v, depth+1) + (i < obj.length-1 ? ',' : '') + '</div>'; }});
    html += '</div><span class="jt-bracket">]</span>';
    return html;
  }}
  if (typeof obj === 'object') {{
    const keys = Object.keys(obj);
    if (keys.length === 0) return '<span class="jt-bracket">{{}}</span>';
    const id = 'jt' + Math.random().toString(36).slice(2,9);
    const collapsed = depth >= 2;
    let html = '<span class="jt-toggle" onclick="toggleJt(\\'' + id + '\\')">' +
               (collapsed ? '&blacktriangleright;' : '&blacktriangledown;') + '</span>';
    html += '<span class="jt-bracket">{{</span>';
    html += '<div id="' + id + '"' + (collapsed ? ' class="jt-collapsed"' : '') + ' style="margin-left:1.2rem;">';
    keys.forEach((k, i) => {{
      html += '<div><span class="jt-key">' + k + '</span>: ' +
              renderJson(obj[k], depth+1) + (i < keys.length-1 ? ',' : '') + '</div>';
    }});
    html += '</div><span class="jt-bracket">}}</span>';
    return html;
  }}
  return String(obj);
}}
function toggleJt(id) {{
  const el = document.getElementById(id);
  if (el.classList.contains('jt-collapsed')) {{
    el.classList.remove('jt-collapsed');
    const prev = el.previousElementSibling;
    if (prev && prev.previousElementSibling && prev.previousElementSibling.classList.contains('jt-toggle'))
      prev.previousElementSibling.innerHTML = '&blacktriangledown;';
  }} else {{
    el.classList.add('jt-collapsed');
    const prev = el.previousElementSibling;
    if (prev && prev.previousElementSibling && prev.previousElementSibling.classList.contains('jt-toggle'))
      prev.previousElementSibling.innerHTML = '&blacktriangleright;';
  }}
}}
Object.keys(DOCS).forEach(sid => {{
  const el = document.getElementById('json-' + sid);
  if (el) el.innerHTML = renderJson(DOCS[sid], 0);
}});

// ─── Legend ───
Object.keys(DATA).forEach(sid => {{
  const d = DATA[sid];
  const el = document.getElementById('legend-' + sid);
  if (!el) return;
  let html = '<div style="font-weight:600;margin-bottom:.3rem;">Ingredients</div>';
  Object.keys(d.ingr_info).forEach(name => {{
    const info = d.ingr_info[name];
    const c = info.color;
    const r = Math.round(c[0]*255), g = Math.round(c[1]*255), b = Math.round(c[2]*255);
    const count = d.ingr_counts[name] || 0;
    html += '<div class="legend-item">' +
      '<span class="legend-dot" style="background:rgb('+r+','+g+','+b+')"></span>' +
      '<span>' + name + ' (r=' + info.radius + ') &times; ' + count + '</span></div>';
  }});
  el.innerHTML = html;
}});

// ─── Three.js Viewers ───
Object.keys(DATA).forEach(sid => {{
  const d = DATA[sid];
  const canvas = document.getElementById('canvas-' + sid);
  const W = canvas.parentElement.clientWidth;
  const H = 500;
  canvas.width = W * window.devicePixelRatio;
  canvas.height = H * window.devicePixelRatio;
  canvas.style.width = W + 'px';
  canvas.style.height = H + 'px';

  const renderer = new THREE.WebGLRenderer({{canvas, antialias:true}});
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(W, H);
  renderer.setClearColor(0xf1f5f9);

  const scene = new THREE.Scene();
  const cam = new THREE.PerspectiveCamera(45, W/H, 0.1, 10000);
  cam.position.set(...d.camera);

  const controls = new THREE.OrbitControls(cam, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.6;

  // Center the camera target on the bounding box center
  const bbCenter = [
    (d.bb[0][0] + d.bb[1][0]) / 2,
    (d.bb[0][1] + d.bb[1][1]) / 2,
    (d.bb[0][2] + d.bb[1][2]) / 2,
  ];
  controls.target.set(...bbCenter);

  scene.add(new THREE.AmbientLight(0xffffff, 0.45));
  const dl1 = new THREE.DirectionalLight(0xffffff, 0.7);
  dl1.position.set(500, 800, 600); scene.add(dl1);
  const dl2 = new THREE.DirectionalLight(0xcbd5e1, 0.35);
  dl2.position.set(-300, -200, -400); scene.add(dl2);

  // Bounding box wireframe
  const bbSize = [d.bb[1][0]-d.bb[0][0], d.bb[1][1]-d.bb[0][1], d.bb[1][2]-d.bb[0][2]];
  const boxGeom = new THREE.BoxGeometry(...bbSize);
  const boxEdges = new THREE.EdgesGeometry(boxGeom);
  const boxLine = new THREE.LineSegments(boxEdges,
    new THREE.LineBasicMaterial({{color:0x94a3b8, transparent:true, opacity:0.4}}));
  boxLine.position.set(...bbCenter);
  scene.add(boxLine);

  // Render spheres using InstancedMesh for performance
  // Group by radius for efficiency
  const byRadius = {{}};
  for (let i = 0; i < d.positions.length; i++) {{
    const r = d.radii[i];
    if (!byRadius[r]) byRadius[r] = [];
    byRadius[r].push(i);
  }}

  Object.keys(byRadius).forEach(rKey => {{
    const r = parseFloat(rKey);
    const indices = byRadius[rKey];
    const sphereGeom = new THREE.SphereGeometry(r, 16, 12);
    const mat = new THREE.MeshPhongMaterial({{
      vertexColors: false,
      shininess: 60,
      specular: 0x333333,
    }});
    const instMesh = new THREE.InstancedMesh(sphereGeom, mat, indices.length);

    const dummy = new THREE.Object3D();
    const color = new THREE.Color();
    for (let j = 0; j < indices.length; j++) {{
      const idx = indices[j];
      const pos = d.positions[idx];
      dummy.position.set(pos[0], pos[1], pos[2]);
      dummy.updateMatrix();
      instMesh.setMatrixAt(j, dummy.matrix);
      const c = d.colors[idx];
      color.setRGB(c[0], c[1], c[2]);
      instMesh.setColorAt(j, color);
    }}
    instMesh.instanceMatrix.needsUpdate = true;
    instMesh.instanceColor.needsUpdate = true;
    instMesh.material.vertexColors = false;
    // Enable per-instance colors
    instMesh.material = new THREE.MeshPhongMaterial({{
      shininess: 60, specular: 0x333333,
    }});
    // Re-set colors with correct material
    instMesh.material.dispose();
    const mat2 = new THREE.MeshPhongMaterial({{ shininess:60, specular:0x444444 }});
    const instMesh2 = new THREE.InstancedMesh(sphereGeom, mat2, indices.length);
    for (let j = 0; j < indices.length; j++) {{
      const idx = indices[j];
      const pos = d.positions[idx];
      dummy.position.set(pos[0], pos[1], pos[2]);
      dummy.updateMatrix();
      instMesh2.setMatrixAt(j, dummy.matrix);
      const c = d.colors[idx];
      color.setRGB(c[0], c[1], c[2]);
      instMesh2.setColorAt(j, color);
    }}
    instMesh2.instanceMatrix.needsUpdate = true;
    if (instMesh2.instanceColor) instMesh2.instanceColor.needsUpdate = true;
    scene.add(instMesh2);
  }});

  function animate() {{
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, cam);
  }}
  animate();
}});

// ─── Plotly Charts ───
const pLayout = {{
  paper_bgcolor:'#f8fafc', plot_bgcolor:'#f8fafc',
  font:{{ color:'#64748b', family:'-apple-system,sans-serif', size:11 }},
  margin:{{ l:50, r:15, t:35, b:40 }},
  xaxis:{{ gridcolor:'#e2e8f0', zerolinecolor:'#e2e8f0' }},
  yaxis:{{ gridcolor:'#e2e8f0', zerolinecolor:'#e2e8f0' }},
}};
const pCfg = {{ responsive:true, displayModeBar:false }};

Object.keys(DATA).forEach(sid => {{
  const d = DATA[sid];

  // 1. Ingredient Counts (bar chart)
  const names = Object.keys(d.ingr_counts);
  const counts = names.map(n => d.ingr_counts[n]);
  const barColors = names.map(n => {{
    const c = d.ingr_info[n].color;
    return 'rgb(' + Math.round(c[0]*255) + ',' + Math.round(c[1]*255) + ',' + Math.round(c[2]*255) + ')';
  }});
  Plotly.newPlot('chart-counts-'+sid, [{{
    x: names, y: counts, type: 'bar',
    marker: {{ color: barColors, line: {{ color:'rgba(0,0,0,0.1)', width:1 }} }},
    text: counts, textposition: 'auto',
  }}], {{...pLayout,
    title:{{ text:'Ingredient Counts', font:{{ size:12, color:'#334155' }} }},
    yaxis:{{...pLayout.yaxis, title:{{ text:'Count', font:{{ size:10 }} }} }},
  }}, pCfg);

  // 2. Radial Distribution (histogram)
  Plotly.newPlot('chart-radial-'+sid, [{{
    x: d.radial_dists, type: 'histogram',
    nbinsx: 25,
    marker: {{ color:'#6366f1', line: {{ color:'#4338ca', width:0.5 }} }},
  }}], {{...pLayout,
    title:{{ text:'Distance from Center', font:{{ size:12, color:'#334155' }} }},
    xaxis:{{...pLayout.xaxis, title:{{ text:'Distance (nm)', font:{{ size:10 }} }} }},
    yaxis:{{...pLayout.yaxis, title:{{ text:'Count', font:{{ size:10 }} }} }},
  }}, pCfg);

  // 3. Nearest Neighbor Distances (histogram)
  Plotly.newPlot('chart-nn-'+sid, [{{
    x: d.nn_dists, type: 'histogram',
    nbinsx: 25,
    marker: {{ color:'#10b981', line: {{ color:'#059669', width:0.5 }} }},
  }}], {{...pLayout,
    title:{{ text:'Nearest Neighbor Distance', font:{{ size:12, color:'#334155' }} }},
    xaxis:{{...pLayout.xaxis, title:{{ text:'Distance (nm)', font:{{ size:10 }} }} }},
    yaxis:{{...pLayout.yaxis, title:{{ text:'Count', font:{{ size:10 }} }} }},
  }}, pCfg);

  // 4. Volume Fractions (pie chart)
  const volNames = names.slice();
  const vols = volNames.map(n => {{
    const r = d.ingr_info[n].radius;
    return d.ingr_counts[n] * (4/3) * Math.PI * r*r*r;
  }});
  const pieColors = volNames.map(n => {{
    const c = d.ingr_info[n].color;
    return 'rgb(' + Math.round(c[0]*255) + ',' + Math.round(c[1]*255) + ',' + Math.round(c[2]*255) + ')';
  }});
  Plotly.newPlot('chart-volume-'+sid, [{{
    labels: volNames, values: vols, type: 'pie',
    marker: {{ colors: pieColors }},
    textinfo: 'label+percent',
    hole: 0.35,
  }}], {{...pLayout,
    title:{{ text:'Volume by Ingredient', font:{{ size:12, color:'#334155' }} }},
    showlegend: false,
    margin:{{ l:20, r:20, t:40, b:20 }},
  }}, pCfg);
}});

</script>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html)
    print(f'Report saved to {output_path}')


def run_demo():
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(demo_dir, 'report.html')

    sim_results = []
    for cfg in CONFIGS:
        print(f'Running: {cfg["title"]}...')
        result, runtime = run_simulation(cfg)
        sim_results.append((cfg, (result, runtime)))
        print(f'  {result["n_packed"]} molecules packed in {runtime:.2f}s')

    print('Generating HTML report...')
    generate_html(sim_results, output_path)

    # Open in Safari
    subprocess.run(['open', '-a', 'Safari', output_path])


if __name__ == '__main__':
    run_demo()
