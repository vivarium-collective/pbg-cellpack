"""Demo: cellPACK canonical packing scenarios with 3D viewers.

Five biologically-motivated packing simulations showcasing cellPACK's
core capabilities:
  1. Blood Plasma — 8 real protein species at physiological concentrations
  2. Vesicle with Surface & Interior Packing — nested compartments
  3. Gradient-Biased Organelle Distribution — spatial gradients
  4. Receptor-Ligand Assembly — partner binding with priority packing
  5. Membrane-Biased Peroxisomes — radial gradient biasing toward cortex

Generates an interactive self-contained HTML report with Three.js 3D
viewers (InstancedMesh), Plotly charts, bigraph-viz diagrams, and PBG
document trees.  Large container ingredients (vesicle shells, dense
cores) and biased zones are rendered as translucent volumes so the
confinement region of each packing is visible at a glance.
"""

import json
import os
import base64
import subprocess
import tempfile
import time

import numpy as np
from process_bigraph import allocate_core
from pbg_cellpack.processes import CellPackStep
from pbg_cellpack.composites import make_packing_document


MESH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'meshes')


def load_obj_mesh(path):
    """Load a triangulated .obj mesh into flat vertex/face arrays.

    Uses trimesh when available (handles quads, normals, etc.) and
    falls back to a tiny pure-Python parser for simple triangle soups.
    Returns a dict ready to serialize into the JS payload:
    ``{'vertices': [x,y,z,...], 'faces': [i,j,k,...], 'bounds': ...}``.
    """
    try:
        import trimesh
        mesh = trimesh.load(path, force='mesh')
        verts = np.asarray(mesh.vertices, dtype=np.float32).reshape(-1)
        faces = np.asarray(mesh.faces, dtype=np.uint32).reshape(-1)
        bounds = mesh.bounds.tolist()
    except Exception:
        vs, fs = [], []
        with open(path) as fh:
            for line in fh:
                if line.startswith('v '):
                    _, x, y, z = line.split()[:4]
                    vs.extend([float(x), float(y), float(z)])
                elif line.startswith('f '):
                    idx = [int(tok.split('/')[0]) - 1 for tok in line.split()[1:4]]
                    fs.extend(idx)
        verts = np.asarray(vs, dtype=np.float32)
        faces = np.asarray(fs, dtype=np.uint32)
        pts = verts.reshape(-1, 3)
        bounds = [pts.min(axis=0).tolist(), pts.max(axis=0).tolist()]
    return {
        'vertices': verts.tolist(),
        'faces': faces.tolist(),
        'bounds': bounds,
        'n_vertices': int(len(verts) // 3),
        'n_faces': int(len(faces) // 3),
    }


# ── Simulation Configs ──────────────────────────────────────────────

CONFIGS = [
    # ── 1. Blood Plasma ──────────────────────────────────────────
    {
        'id': 'plasma',
        'title': 'Blood Plasma',
        'subtitle': '8 protein species at physiological concentrations',
        'description': (
            'The canonical cellPACK demonstration: blood plasma proteins packed '
            'into a 600 nm cube at roughly physiological ratios.  Serum albumin '
            '(PDB 1e7i) dominates at ~75% of total count, followed by IgG '
            'antibodies, transferrin, alpha-1-antitrypsin, fibrinogen, '
            'ceruloplasmin, hemoglobin, and a single IgM pentamer.  Protein '
            'radii are set to their crystallographic encapsulating radii.  '
            'This model matches Figure 4 of the cellPACK Nature Methods paper '
            '(Johnson et al., 2015).'
        ),
        'config': {
            'recipe': {
                'version': '1.0.0',
                'format_version': '2.0',
                'name': 'blood_plasma',
                'bounding_box': [[0, 0, 0], [600, 600, 600]],
                'objects': {
                    'base': {
                        'type': 'single_sphere', 'place_method': 'jitter',
                        'jitter_attempts': 15, 'packing_mode': 'random',
                        'max_jitter': [1, 1, 1],
                    },
                    'SerumAlbumin': {
                        'type': 'single_sphere', 'inherit': 'base',
                        'color': [0.35, 0.55, 0.82], 'radius': 22,
                    },
                    'IgG_Antibody': {
                        'type': 'single_sphere', 'inherit': 'base',
                        'color': [0.80, 0.60, 0.20], 'radius': 28,
                    },
                    'Transferrin': {
                        'type': 'single_sphere', 'inherit': 'base',
                        'color': [0.17, 0.76, 0.01], 'radius': 26,
                    },
                    'Fibrinogen': {
                        'type': 'single_sphere', 'inherit': 'base',
                        'color': [0.85, 0.62, 0.20], 'radius': 25,
                    },
                    'AntiTrypsin': {
                        'type': 'single_sphere', 'inherit': 'base',
                        'color': [0.56, 0.40, 0.68], 'radius': 17,
                    },
                    'Hemoglobin': {
                        'type': 'single_sphere', 'inherit': 'base',
                        'color': [0.76, 0.18, 0.15], 'radius': 23,
                    },
                    'Ceruloplasmin': {
                        'type': 'single_sphere', 'inherit': 'base',
                        'color': [0.77, 0.71, 0.61], 'radius': 19,
                    },
                    'IgM_Antibody': {
                        'type': 'single_sphere', 'inherit': 'base',
                        'color': [0.90, 0.45, 0.10], 'radius': 37,
                    },
                },
                'composition': {
                    'space': {'regions': {
                        'interior': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
                    }},
                    'A': {'object': 'SerumAlbumin', 'count': 400},
                    'B': {'object': 'IgG_Antibody', 'count': 55},
                    'C': {'object': 'Transferrin', 'count': 20},
                    'D': {'object': 'Fibrinogen', 'count': 8},
                    'E': {'object': 'AntiTrypsin', 'count': 35},
                    'F': {'object': 'Hemoglobin', 'count': 1},
                    'G': {'object': 'Ceruloplasmin', 'count': 2},
                    'H': {'object': 'IgM_Antibody', 'count': 1},
                },
            },
        },
        'seed': 42,
        'camera': [900, 600, 900],
        'color_scheme': 'indigo',
    },

    # ── 2. Vesicle with Surface & Interior Packing ───────────────
    {
        'id': 'vesicle',
        'title': 'Synaptic Vesicle Model',
        'subtitle': 'Nested compartments with surface receptors and luminal cargo',
        'description': (
            'A vesicle-like compartment (r=400 nm) is packed with surface '
            'receptor proteins on the outer shell and neurotransmitter-sized '
            'cargo in the lumen.  An inner dense-core granule (r=80 nm) '
            'is embedded inside.  This uses cellPACK\'s hierarchical '
            'composition with both surface and interior packing regions, '
            'matching the synaptic vesicle models from the Nature Methods '
            'paper (Figure 5).'
        ),
        'config': {
            'recipe': {
                'version': '1.0.0',
                'format_version': '2.0',
                'name': 'vesicle_model',
                'bounding_box': [[-600, -600, -600], [600, 600, 600]],
                'objects': {
                    'base': {
                        'type': 'single_sphere', 'jitter_attempts': 10,
                        'packing_mode': 'random', 'max_jitter': [1, 1, 1],
                        'place_method': 'jitter',
                        'available_regions': {'interior': {}, 'surface': {}},
                    },
                    'vesicle_shell': {
                        'inherit': 'base',
                        'color': [0.75, 0.75, 0.85], 'radius': 400,
                    },
                    'dense_core': {
                        'inherit': 'base',
                        'color': [0.40, 0.40, 0.55], 'radius': 80,
                    },
                    'vATPase': {
                        'inherit': 'base',
                        'color': [0.85, 0.30, 0.25], 'radius': 30,
                    },
                    'SNARE_complex': {
                        'inherit': 'base',
                        'color': [0.20, 0.70, 0.40], 'radius': 20,
                    },
                    'neurotransmitter': {
                        'inherit': 'base',
                        'color': [0.95, 0.75, 0.15], 'radius': 12,
                    },
                    'luminal_protein': {
                        'inherit': 'base',
                        'color': [0.55, 0.35, 0.75], 'radius': 18,
                    },
                },
                'composition': {
                    'bounding_area': {
                        'regions': {'interior': ['vesicle']}
                    },
                    'vesicle': {
                        'object': 'vesicle_shell', 'count': 1,
                        'regions': {
                            'interior': [
                                'granule',
                                {'object': 'neurotransmitter', 'count': 60},
                                {'object': 'luminal_protein', 'count': 20},
                            ],
                            'surface': [
                                {'object': 'vATPase', 'count': 12},
                                {'object': 'SNARE_complex', 'count': 25},
                            ],
                        },
                    },
                    'granule': {
                        'object': 'dense_core',
                        'regions': {
                            'interior': [
                                {'object': 'luminal_protein', 'count': 10},
                            ],
                        },
                    },
                },
            },
        },
        'seed': 7,
        'camera': [700, 500, 700],
        'color_scheme': 'emerald',
        'container_ingredients': ['vesicle_shell', 'dense_core'],
    },

    # ── 3. Gradient-Biased Organelle Distribution ────────────────
    {
        'id': 'gradient',
        'title': 'Gradient-Biased Organelle Packing',
        'subtitle': 'Spatial gradients drive non-uniform organelle placement',
        'description': (
            'Mimics the canonical cellPACK peroxisome/endosome distribution '
            'models: organelle-sized puncta are placed under the influence of '
            'exponential spatial gradients.  Two populations are packed: '
            'peroxisomes (green) biased toward low-X by an X-gradient, and '
            'endosomes (gold) biased toward low-Y by a Y-gradient.  A uniform '
            'background of ribosomes fills the remaining volume.  This '
            'reproduces the gradient-packing pattern used in the hiPS cell '
            'organelle distribution studies.'
        ),
        'config': {
            'recipe': {
                'version': '1.0.0',
                'format_version': '2.0',
                'name': 'organelle_gradient',
                'bounding_box': [[0, 0, 0], [500, 500, 500]],
                'gradients': {
                    'X_gradient': {
                        'description': 'Exponential decay along X',
                        'mode': 'X', 'pick_mode': 'rnd',
                        'weight_mode': 'exponential',
                        'weight_mode_settings': {'decay_length': 0.12},
                    },
                    'Y_gradient': {
                        'description': 'Exponential decay along Y',
                        'mode': 'Y', 'pick_mode': 'rnd',
                        'weight_mode': 'exponential',
                        'weight_mode_settings': {'decay_length': 0.12},
                    },
                },
                'objects': {
                    'base': {
                        'type': 'single_sphere', 'place_method': 'jitter',
                        'jitter_attempts': 15, 'packing_mode': 'random',
                        'max_jitter': [1, 1, 1],
                    },
                    'peroxisome': {
                        'type': 'single_sphere', 'inherit': 'base',
                        'color': [0.20, 0.72, 0.10], 'radius': 15,
                        'packing_mode': 'gradient',
                        'gradient': ['X_gradient'],
                        'gradient_weights': [100],
                    },
                    'endosome': {
                        'type': 'single_sphere', 'inherit': 'base',
                        'color': [1.0, 0.84, 0.0], 'radius': 20,
                        'packing_mode': 'gradient',
                        'gradient': ['Y_gradient'],
                        'gradient_weights': [100],
                    },
                    'ribosome': {
                        'type': 'single_sphere', 'inherit': 'base',
                        'color': [0.60, 0.60, 0.65], 'radius': 8,
                    },
                },
                'composition': {
                    'space': {'regions': {
                        'interior': ['A', 'B', 'C'],
                    }},
                    'A': {'object': 'peroxisome', 'count': 120},
                    'B': {'object': 'endosome', 'count': 60},
                    'C': {'object': 'ribosome', 'count': 250},
                },
            },
        },
        'seed': 17,
        'camera': [750, 500, 750],
        'color_scheme': 'amber',
    },

    # ── 4. Receptor-Ligand Assembly ──────────────────────────────
    {
        'id': 'partner',
        'title': 'Receptor-Ligand Assembly',
        'subtitle': 'Priority packing with stochastic partner binding',
        'description': (
            'Models the formation of receptor-ligand complexes in a crowded '
            'environment.  Receptor proteins (blue, r=25 nm) are placed first '
            'using priority=-1.  Ligand molecules (orange, r=12 nm) then pack '
            'using cellPACK\'s closePartner mode with a 70% binding '
            'probability, preferentially placing each ligand near an existing '
            'receptor.  A dense background of small inert molecules (gray, '
            'r=8 nm) fills the remaining volume.  This pattern is used to '
            'study multivalent binding, immune synapse formation, and other '
            'proximity-driven assembly phenomena.'
        ),
        'config': {
            'recipe': {
                'version': '1.1',
                'format_version': '2.1',
                'name': 'receptor_ligand',
                'bounding_box': [[0, 0, 0], [500, 500, 500]],
                'objects': {
                    'receptor': {
                        'color': [0.11, 0.47, 0.69], 'jitter_attempts': 20,
                        'rotation_range': 6.2831, 'max_jitter': [1, 1, 1],
                        'packing_mode': 'random', 'type': 'single_sphere',
                        'rejection_threshold': 200, 'place_method': 'jitter',
                        'radius': 25,
                    },
                    'ligand': {
                        'color': [1.0, 0.41, 0.0], 'jitter_attempts': 20,
                        'partners': [
                            {'name': 'receptor', 'binding_probability': 0.7},
                        ],
                        'rotation_range': 6.2831, 'max_jitter': [1, 1, 1],
                        'packing_mode': 'closePartner',
                        'type': 'single_sphere',
                        'rejection_threshold': 500, 'place_method': 'jitter',
                        'radius': 12,
                    },
                    'crowder': {
                        'color': [0.65, 0.65, 0.65], 'jitter_attempts': 10,
                        'rotation_range': 6.2831, 'max_jitter': [1, 1, 1],
                        'packing_mode': 'random', 'type': 'single_sphere',
                        'place_method': 'jitter', 'radius': 8,
                    },
                },
                'composition': {
                    'space': {
                        'regions': {
                            'interior': [
                                {'object': 'receptor', 'count': 40, 'priority': -1},
                                {'object': 'ligand', 'count': 100, 'priority': 0},
                                {'object': 'crowder', 'count': 250, 'priority': 1},
                            ],
                        },
                    },
                },
            },
        },
        'seed': 99,
        'camera': [750, 500, 750],
        'color_scheme': 'rose',
    },

    # ── 5. Membrane-Biased Peroxisome Packing ────────────────────
    # Adapted from cellPACK's examples/recipes/v2/er_peroxisome.json
    # (mesoscope/cellpack).  The real recipe uses three triangulated
    # compartments from a segmented hiPS cell (plasma membrane,
    # nucleus, and endoplasmic reticulum) and biases peroxisomes by
    # distance to the ER surface.  We reuse the same bounding box,
    # peroxisome radius/count, and the three meshes for visualization,
    # but drive the packing with a lighter-weight radial gradient so
    # the demo runs in seconds without the mesh-based grid build.
    {
        'id': 'membrane',
        'title': 'Peroxisomes in a Segmented hiPS Cell',
        'subtitle': 'er_peroxisome recipe — real membrane, nucleus, and ER meshes',
        'description': (
            'Adapted from the cellPACK '
            '<a href="https://github.com/mesoscope/cellpack/blob/main/examples/recipes/v2/er_peroxisome.json" target="_blank">'
            '<code>er_peroxisome.json</code></a> recipe '
            '(mesoscope/cellpack).  The canonical recipe packs 121 '
            'peroxisomes (r&nbsp;=&nbsp;2.37&nbsp;nm) inside a triangulated '
            'plasma-membrane compartment of a segmented hiPS cell, '
            'excluding the nucleus and endoplasmic reticulum, and biases '
            'them by distance to the ER surface.  We reuse the real '
            'bounding box and the three meshes (membrane / nucleus / ER) '
            'as the confinement visualization, and drive the packing '
            'itself with a radial gradient centered on the cell so it '
            'runs in seconds without the mesh-surface grid build.  '
            'Green spheres are peroxisomes, gray spheres are smaller '
            'crowders.  Toggle the compartments in the legend.'
        ),
        'config': {
            'recipe': {
                'version': '1.0.0',
                'format_version': '2.1',
                'name': 'er_peroxisome_sim',
                'bounding_box': [
                    [33.775, 35.375, 7.125],
                    [274.225, 208.625, 106.875],
                ],
                'gradients': {
                    'cell_radial': {
                        'description': 'Radial, inverted so weight peaks near the plasma membrane',
                        'mode': 'radial',
                        'mode_settings': {
                            'center': [154.0, 122.0, 57.0],
                            'radius': 130.0,
                        },
                        'pick_mode': 'rnd',
                        'weight_mode': 'linear',
                        'invert': 'weight',
                    },
                },
                'objects': {
                    'base': {
                        'type': 'single_sphere', 'place_method': 'jitter',
                        'jitter_attempts': 20, 'packing_mode': 'random',
                        'max_jitter': [1, 1, 1],
                    },
                    'peroxisome': {
                        'type': 'single_sphere', 'inherit': 'base',
                        'color': [0.12, 0.80, 0.18], 'radius': 2.37,
                        'packing_mode': 'gradient',
                        'gradient': ['cell_radial'],
                        'gradient_weights': [100],
                    },
                    'crowder': {
                        'type': 'single_sphere', 'inherit': 'base',
                        'color': [0.70, 0.70, 0.74], 'radius': 1.4,
                    },
                },
                'composition': {
                    'space': {'regions': {
                        'interior': ['A', 'B'],
                    }},
                    'A': {'object': 'peroxisome', 'count': 121},
                    'B': {'object': 'crowder', 'count': 900},
                },
            },
        },
        'seed': 31,
        'camera': [500, 350, 500],
        'color_scheme': 'teal',
        # Named meshes to overlay as translucent compartments in the
        # 3D viewer.  Files live in demo/meshes/ and were downloaded
        # from cellpack-analysis-data.s3.us-west-2.amazonaws.com.
        'meshes': [
            {'id': 'membrane', 'file': 'mem_mesh_370574.obj',
             'label': 'Plasma membrane', 'color': '#94a3b8', 'opacity': 0.08},
            {'id': 'nucleus',  'file': 'nuc_mesh_370574.obj',
             'label': 'Nucleus', 'color': '#6366f1', 'opacity': 0.28},
            {'id': 'er',       'file': 'struct_mesh_370574.obj',
             'label': 'Endoplasmic reticulum', 'color': '#f59e0b', 'opacity': 0.35},
        ],
    },
]


# ── Simulation ──────────────────────────────────────────────────────

def run_simulation(cfg_entry):
    core = allocate_core()
    core.register_link('CellPackStep', CellPackStep)
    t0 = time.perf_counter()
    step = CellPackStep(config=cfg_entry['config'], core=core)
    result = step.update({'seed': cfg_entry['seed']})
    return result, time.perf_counter() - t0


# ── Analysis helpers ────────────────────────────────────────────────

def ingredient_counts(result):
    c = {}
    for n in result['ingredient_names']:
        c[n] = c.get(n, 0) + 1
    return c


def nearest_neighbor_dists(positions):
    pts = np.array(positions)
    n = len(pts)
    if n < 2:
        return []
    nn = []
    for i in range(n):
        d = np.sqrt(np.sum((pts - pts[i]) ** 2, axis=1))
        d[i] = np.inf
        nn.append(float(np.min(d)))
    return nn


def axis_positions(positions, axis):
    return [float(p[axis]) for p in positions]


def species_positions(result, keyword, axis):
    return [result['positions'][i][axis]
            for i, n in enumerate(result['ingredient_names'])
            if keyword in n.lower()]


def cross_species_dists(positions, names, kw_target, kw_source):
    pts = np.array(positions)
    idx_t = [i for i, n in enumerate(names) if kw_target in n.lower()]
    idx_s = [i for i, n in enumerate(names) if kw_source in n.lower()]
    if not idx_t or not idx_s:
        return []
    pa = pts[idx_t]
    return [float(np.min(np.sqrt(np.sum((pa - pts[i]) ** 2, axis=1))))
            for i in idx_s]


def radial_dists(positions, bb):
    pts = np.array(positions)
    center = np.array([(bb[0][i] + bb[1][i]) / 2 for i in range(3)])
    return [float(d) for d in np.sqrt(np.sum((pts - center) ** 2, axis=1))]


# ── Bigraph diagram ────────────────────────────────────────────────

def generate_bigraph_image(cfg_entry):
    from bigraph_viz import plot_bigraph
    doc = {
        'packer': {
            '_type': 'step', 'address': 'local:CellPackStep',
            'config': {'place_method': 'jitter'},
            'inputs': {'seed': ['stores', 'seed']},
            'outputs': {
                'positions': ['stores', 'positions'],
                'radii': ['stores', 'radii'],
                'n_packed': ['stores', 'n_packed'],
                'packing_fraction': ['stores', 'packing_fraction'],
            },
        },
        'stores': {},
        'emitter': {
            '_type': 'step', 'address': 'local:ram-emitter',
            'config': {'emit': {
                'n_packed': 'integer', 'packing_fraction': 'float'}},
            'inputs': {
                'n_packed': ['stores', 'n_packed'],
                'packing_fraction': ['stores', 'packing_fraction'],
            },
        },
    }
    node_colors = {
        ('packer',): '#6366f1', ('emitter',): '#8b5cf6', ('stores',): '#e0e7ff',
    }
    outdir = tempfile.mkdtemp()
    plot_bigraph(
        state=doc, out_dir=outdir, filename='bigraph',
        file_format='png', remove_process_place_edges=True,
        rankdir='LR', node_fill_colors=node_colors,
        node_label_size='16pt', port_labels=False, dpi='150',
    )
    with open(os.path.join(outdir, 'bigraph.png'), 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'data:image/png;base64,{b64}'


# ── HTML ────────────────────────────────────────────────────────────

COLOR_SCHEMES = {
    'indigo': {'primary': '#6366f1', 'light': '#e0e7ff', 'dark': '#4338ca',
               'bg': '#eef2ff', 'accent': '#818cf8', 'text': '#312e81'},
    'emerald': {'primary': '#10b981', 'light': '#d1fae5', 'dark': '#059669',
                'bg': '#ecfdf5', 'accent': '#34d399', 'text': '#064e3b'},
    'amber':   {'primary': '#f59e0b', 'light': '#fef3c7', 'dark': '#d97706',
                'bg': '#fffbeb', 'accent': '#fbbf24', 'text': '#78350f'},
    'rose':    {'primary': '#f43f5e', 'light': '#ffe4e6', 'dark': '#e11d48',
                'bg': '#fff1f2', 'accent': '#fb7185', 'text': '#881337'},
    'teal':    {'primary': '#0d9488', 'light': '#ccfbf1', 'dark': '#0f766e',
                'bg': '#f0fdfa', 'accent': '#2dd4bf', 'text': '#134e4a'},
}


def generate_html(sim_results, output_path):
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
        ic = ingredient_counts(result)

        ingr_info = {}
        for i, name in enumerate(result['ingredient_names']):
            if name not in ingr_info:
                ingr_info[name] = {
                    'color': result['ingredient_colors'][i],
                    'radius': result['radii'][i],
                }

        nn = nearest_neighbor_dists(result['positions'])

        # Per-config extra analysis
        extra = {}
        if sid == 'plasma':
            extra['radial'] = radial_dists(result['positions'], bb)
        elif sid == 'vesicle':
            extra['radial'] = radial_dists(result['positions'], bb)
        elif sid == 'gradient':
            extra['perox_x'] = species_positions(result, 'perox', 0)
            extra['endo_y'] = species_positions(result, 'endo', 1)
            extra['all_x'] = axis_positions(result['positions'], 0)
            extra['all_y'] = axis_positions(result['positions'], 1)
        elif sid == 'partner':
            extra['lig_rec'] = cross_species_dists(
                result['positions'], result['ingredient_names'],
                'receptor', 'ligand')
            extra['crowd_rec'] = cross_species_dists(
                result['positions'], result['ingredient_names'],
                'receptor', 'crowder')
        elif sid == 'membrane':
            # Radial distance from the cell center, split by species,
            # to visualize the membrane bias.
            center = np.array([(bb[0][i] + bb[1][i]) / 2 for i in range(3)])
            names = result['ingredient_names']
            pts = np.array(result['positions'])
            dists = np.sqrt(np.sum((pts - center) ** 2, axis=1))
            extra['perox_r'] = [float(dists[i]) for i, n in enumerate(names)
                                if 'perox' in n.lower()]
            extra['ribo_r'] = [float(dists[i]) for i, n in enumerate(names)
                               if 'crowd' in n.lower() or 'ribo' in n.lower()]
            # Approximate cell radius for the "membrane" annotation on
            # the chart: half the shorter in-plane box dimension.
            extra['half_width'] = float(min(
                bb[1][0] - bb[0][0], bb[1][1] - bb[0][1]) / 2)

        # Confinement-volume descriptor used by the 3D viewer to draw
        # the region objects are packed into (translucent overlay).
        confinement = {'kind': 'box'}
        if sid == 'membrane' and cfg.get('meshes'):
            mesh_payload = []
            for entry in cfg['meshes']:
                path = os.path.join(MESH_DIR, entry['file'])
                if not os.path.exists(path):
                    print(f'  missing mesh: {path} — skipping')
                    continue
                print(f'  loading mesh {entry["file"]}...')
                m = load_obj_mesh(path)
                m.update({
                    'id': entry['id'], 'label': entry['label'],
                    'color': entry['color'], 'opacity': entry['opacity'],
                })
                mesh_payload.append(m)
                print(
                    f'    {m["n_vertices"]} verts / {m["n_faces"]} faces')
            confinement = {'kind': 'mesh', 'meshes': mesh_payload}

        all_js_data[sid] = {
            'positions': result['positions'],
            'radii': result['radii'],
            'colors': result['ingredient_colors'],
            'names': result['ingredient_names'],
            'bb': bb, 'camera': cfg['camera'],
            'ingr_counts': ic,
            'ingr_info': {k: v for k, v in ingr_info.items()},
            'nn': nn, 'extra': extra,
            'container_ingredients': cfg.get('container_ingredients', []),
            'confinement': confinement,
        }

        print(f'  Generating bigraph diagram for {sid}...')
        bigraph_img = generate_bigraph_image(cfg)
        pbg_docs[sid] = make_packing_document(
            recipe=recipe, place_method='jitter', seed=cfg['seed'])

        bb_vol = float(np.prod(np.array(bb[1]) - np.array(bb[0])))
        bb_size = [bb[1][i] - bb[0][i] for i in range(3)]

        # Pretty-print the confinement volume: nm^3 up to 10^6, otherwise µm^3.
        if bb_vol >= 1e9:
            vol_value = f'{bb_vol / 1e9:.2f}'
            vol_units = '&micro;m&sup3;'
        else:
            vol_value = f'{bb_vol / 1e6:.2f}'
            vol_units = '10&#8310; nm&sup3;'

        confinement_caption = {
            'plasma': 'Confinement: cubic bounding box',
            'vesicle': 'Confinement: vesicle shell (r=400 nm) &middot; dense-core granule (r=80 nm)',
            'gradient': 'Confinement: cubic bounding box &middot; X/Y gradients',
            'partner': 'Confinement: cubic bounding box',
            'membrane': 'Confinement: hiPS-cell mesh &middot; plasma membrane / nucleus / ER (cellPACK er_peroxisome)',
        }.get(sid, 'Confinement: cubic bounding box')

        ingr_summary = ' &middot; '.join(
            f'{name}: <strong>{count}</strong>'
            for name, count in sorted(ic.items(), key=lambda x: -x[1]))

        section = f"""
    <div class="sim-section" id="sim-{sid}">
      <div class="sim-header" style="border-left:4px solid {cs['primary']};">
        <div class="sim-number" style="background:{cs['light']};color:{cs['dark']};">{idx+1}</div>
        <div>
          <h2 class="sim-title">{cfg['title']}</h2>
          <p class="sim-subtitle">{cfg['subtitle']}</p>
        </div>
      </div>
      <p class="sim-description">{cfg['description']}</p>
      <div class="metrics-row">
        <div class="metric"><span class="metric-label">Packed</span><span class="metric-value">{n_packed:,}</span></div>
        <div class="metric"><span class="metric-label">Species</span><span class="metric-value">{len(ic)}</span></div>
        <div class="metric"><span class="metric-label">Packing &phi;</span><span class="metric-value">{pf:.3f}</span></div>
        <div class="metric"><span class="metric-label">Box</span><span class="metric-value">{bb_size[0]:.0f}&sup3;</span><span class="metric-sub">nm</span></div>
        <div class="metric"><span class="metric-label">Volume</span><span class="metric-value">{vol_value}</span><span class="metric-sub">{vol_units}</span></div>
        <div class="metric"><span class="metric-label">Runtime</span><span class="metric-value">{runtime:.1f}s</span></div>
      </div>
      <p class="ingr-summary">{ingr_summary}</p>
      <h3 class="subsection-title">3D Molecular Packing</h3>
      <div class="viewer-wrap">
        <canvas id="canvas-{sid}" class="mesh-canvas"></canvas>
        <div class="viewer-info"><strong>{n_packed}</strong> molecules &middot; <strong>{len(ic)}</strong> species<br>{confinement_caption}<br>Drag to rotate &middot; Scroll to zoom</div>
        <div class="legend-box" id="legend-{sid}"></div>
      </div>
      <h3 class="subsection-title">Analysis</h3>
      <div class="charts-row">
        <div class="chart-box"><div id="chart-a-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-b-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-c-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-d-{sid}" class="chart"></div></div>
      </div>
      <div class="pbg-row">
        <div class="pbg-col"><h3 class="subsection-title">Bigraph Architecture</h3><div class="bigraph-img-wrap"><img src="{bigraph_img}" alt="Bigraph"></div></div>
        <div class="pbg-col"><h3 class="subsection-title">Composite Document</h3><div class="json-tree" id="json-{sid}"></div></div>
      </div>
    </div>"""
        sections_html.append(section)

    nav_items = ''.join(
        f'<a href="#sim-{c["id"]}" class="nav-link" '
        f'style="border-color:{COLOR_SCHEMES[c["color_scheme"]]["primary"]};">'
        f'{c["title"]}</a>'
        for c in [r[0] for r in sim_results])

    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>cellPACK Molecular Packing Report</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#fff;color:#1e293b;line-height:1.6}}
.page-header{{background:linear-gradient(135deg,#f8fafc 0%,#eef2ff 50%,#fdf2f8 100%);border-bottom:1px solid #e2e8f0;padding:3rem}}
.page-header h1{{font-size:2.2rem;font-weight:800;color:#0f172a;margin-bottom:.3rem}}
.page-header p{{color:#64748b;font-size:.95rem;max-width:720px}}
.nav{{display:flex;gap:.8rem;padding:1rem 3rem;background:#f8fafc;border-bottom:1px solid #e2e8f0;position:sticky;top:0;z-index:100}}
.nav-link{{padding:.4rem 1rem;border-radius:8px;border:1.5px solid;text-decoration:none;font-size:.85rem;font-weight:600;transition:all .15s}}
.nav-link:hover{{transform:translateY(-1px);box-shadow:0 2px 8px rgba(0,0,0,.08)}}
.sim-section{{padding:2.5rem 3rem;border-bottom:1px solid #e2e8f0}}
.sim-header{{display:flex;align-items:center;gap:1rem;margin-bottom:.8rem;padding-left:1rem}}
.sim-number{{width:36px;height:36px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:1.1rem}}
.sim-title{{font-size:1.5rem;font-weight:700;color:#0f172a}}
.sim-subtitle{{font-size:.9rem;color:#64748b}}
.sim-description{{color:#475569;font-size:.9rem;margin-bottom:1.2rem;max-width:800px}}
.ingr-summary{{font-size:.85rem;color:#64748b;margin-bottom:1rem}}
.subsection-title{{font-size:1.05rem;font-weight:600;color:#334155;margin:1.5rem 0 .8rem}}
.metrics-row{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:.8rem;margin-bottom:.8rem}}
.metric{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:.8rem;text-align:center}}
.metric-label{{display:block;font-size:.7rem;text-transform:uppercase;letter-spacing:.06em;color:#94a3b8;margin-bottom:.2rem}}
.metric-value{{display:block;font-size:1.3rem;font-weight:700;color:#1e293b}}
.metric-sub{{display:block;font-size:.7rem;color:#94a3b8}}
.viewer-wrap{{position:relative;background:#f1f5f9;border:1px solid #e2e8f0;border-radius:14px;overflow:hidden;margin-bottom:1rem}}
.mesh-canvas{{width:100%;height:500px;display:block;cursor:grab}}.mesh-canvas:active{{cursor:grabbing}}
.viewer-info{{position:absolute;top:.8rem;left:.8rem;background:rgba(255,255,255,.92);border:1px solid #e2e8f0;border-radius:8px;padding:.5rem .8rem;font-size:.75rem;color:#64748b;backdrop-filter:blur(4px)}}
.viewer-info strong{{color:#1e293b}}
.legend-box{{position:absolute;top:.8rem;right:.8rem;background:rgba(255,255,255,.92);border:1px solid #e2e8f0;border-radius:8px;padding:.6rem .8rem;font-size:.72rem;color:#475569;backdrop-filter:blur(4px)}}
.legend-item{{display:flex;align-items:center;gap:.4rem;margin-bottom:.25rem}}
.legend-dot{{width:12px;height:12px;border-radius:50%;display:inline-block}}
.charts-row{{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1rem}}
.chart-box{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;overflow:hidden}}
.chart{{height:280px}}
.pbg-row{{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;margin-top:1rem}}
.pbg-col{{min-width:0}}
.bigraph-img-wrap{{background:#fafafa;border:1px solid #e2e8f0;border-radius:10px;padding:1.5rem;text-align:center}}
.bigraph-img-wrap img{{max-width:100%;height:auto}}
.json-tree{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:1rem;max-height:500px;overflow-y:auto;font-family:'SF Mono',Menlo,Monaco,'Courier New',monospace;font-size:.78rem;line-height:1.5}}
.jt-key{{color:#7c3aed;font-weight:600}}.jt-str{{color:#059669}}.jt-num{{color:#2563eb}}.jt-bool{{color:#d97706}}.jt-null{{color:#94a3b8}}
.jt-toggle{{cursor:pointer;user-select:none;color:#94a3b8;margin-right:.3rem}}.jt-toggle:hover{{color:#1e293b}}
.jt-collapsed{{display:none}}.jt-bracket{{color:#64748b}}
.footer{{text-align:center;padding:2rem;color:#94a3b8;font-size:.8rem;border-top:1px solid #e2e8f0}}
@media(max-width:900px){{.charts-row,.pbg-row{{grid-template-columns:1fr}}.sim-section,.page-header{{padding:1.5rem}}}}
</style></head><body>
<div class="page-header">
  <h1>cellPACK Molecular Packing Report</h1>
  <p>Five canonical mesoscale packing scenarios wrapped as <strong>process-bigraph</strong> Steps using cellPACK.
  Each configuration demonstrates a core capability: multi-species plasma packing, compartment hierarchy
  with surface/interior regions, gradient-biased organelle placement, receptor-ligand partner binding,
  and membrane-biased peroxisome distribution.  Confinement volumes (bounding cubes, vesicle shells,
  membrane-bias zones) are rendered as translucent overlays in each viewer.</p>
</div>
<div class="nav">{nav_items}</div>
{''.join(sections_html)}
<div class="footer">Generated by <strong>pbg-cellpack</strong> &mdash; cellPACK + process-bigraph &mdash; Mesoscale Molecular Packing</div>
<script>
const DATA={json.dumps(all_js_data)};
const DOCS={json.dumps(pbg_docs,indent=2)};

// JSON tree
function renderJson(o,d){{if(d===undefined)d=0;if(o===null)return'<span class="jt-null">null</span>';if(typeof o==='boolean')return'<span class="jt-bool">'+o+'</span>';if(typeof o==='number')return'<span class="jt-num">'+o+'</span>';if(typeof o==='string')return'<span class="jt-str">"'+o.replace(/</g,'&lt;')+'"</span>';if(Array.isArray(o)){{if(!o.length)return'<span class="jt-bracket">[]</span>';if(o.length<=5&&o.every(x=>typeof x!=='object'||x===null))return'<span class="jt-bracket">[</span>'+o.map(x=>renderJson(x,d+1)).join(', ')+'<span class="jt-bracket">]</span>';const id='jt'+Math.random().toString(36).slice(2,9);let h='<span class="jt-toggle" onclick="toggleJt(\\''+id+'\\')">&#9660;</span><span class="jt-bracket">[</span> <span style="color:#94a3b8;font-size:.7rem">'+o.length+' items</span><div id="'+id+'" style="margin-left:1.2rem">';o.forEach((v,i)=>{{h+='<div>'+renderJson(v,d+1)+(i<o.length-1?',':'')+'</div>'}});return h+'</div><span class="jt-bracket">]</span>'}}if(typeof o==='object'){{const keys=Object.keys(o);if(!keys.length)return'<span class="jt-bracket">{{}}</span>';const id='jt'+Math.random().toString(36).slice(2,9);const col=d>=2;let h='<span class="jt-toggle" onclick="toggleJt(\\''+id+'\\')">'+(col?'&#9654;':'&#9660;')+'</span><span class="jt-bracket">{{</span><div id="'+id+'"'+(col?' class="jt-collapsed"':'')+' style="margin-left:1.2rem">';keys.forEach((k,i)=>{{h+='<div><span class="jt-key">'+k+'</span>: '+renderJson(o[k],d+1)+(i<keys.length-1?',':'')+'</div>'}});return h+'</div><span class="jt-bracket">}}</span>'}}return String(o)}}
function toggleJt(id){{const el=document.getElementById(id);if(el.classList.contains('jt-collapsed')){{el.classList.remove('jt-collapsed');const p=el.previousElementSibling;if(p&&p.previousElementSibling&&p.previousElementSibling.classList.contains('jt-toggle'))p.previousElementSibling.innerHTML='&#9660;'}}else{{el.classList.add('jt-collapsed');const p=el.previousElementSibling;if(p&&p.previousElementSibling&&p.previousElementSibling.classList.contains('jt-toggle'))p.previousElementSibling.innerHTML='&#9654;'}}}}
Object.keys(DOCS).forEach(s=>{{const el=document.getElementById('json-'+s);if(el)el.innerHTML=renderJson(DOCS[s],0)}});

// Legend
Object.keys(DATA).forEach(s=>{{const d=DATA[s],el=document.getElementById('legend-'+s);if(!el)return;let h='<div style="font-weight:600;margin-bottom:.3rem">Ingredients</div>';Object.keys(d.ingr_info).forEach(n=>{{const info=d.ingr_info[n],c=info.color,r=Math.round(c[0]*255),g=Math.round(c[1]*255),b=Math.round(c[2]*255);h+='<div class="legend-item"><span class="legend-dot" style="background:rgb('+r+','+g+','+b+')"></span><span>'+n+' (r='+info.radius.toFixed(1)+') &times; '+(d.ingr_counts[n]||0)+'</span></div>'}});if(d.confinement&&d.confinement.kind==='mesh'&&d.confinement.meshes){{h+='<div style="font-weight:600;margin-top:.5rem;margin-bottom:.25rem">Compartments</div>';d.confinement.meshes.forEach(m=>{{h+='<div class="legend-item"><span class="legend-dot" style="background:'+m.color+';opacity:.85;border:1px solid '+m.color+'"></span><span>'+m.label+' &mdash; '+m.n_faces+' faces</span></div>'}})}}el.innerHTML=h}});

// Three.js
Object.keys(DATA).forEach(sid=>{{const d=DATA[sid],canvas=document.getElementById('canvas-'+sid),W=canvas.parentElement.clientWidth,H=500;canvas.width=W*devicePixelRatio;canvas.height=H*devicePixelRatio;canvas.style.width=W+'px';canvas.style.height=H+'px';const renderer=new THREE.WebGLRenderer({{canvas,antialias:true,alpha:true}});renderer.setPixelRatio(devicePixelRatio);renderer.setSize(W,H);renderer.setClearColor(0xf1f5f9);const scene=new THREE.Scene(),cam=new THREE.PerspectiveCamera(45,W/H,0.1,10000);cam.position.set(...d.camera);const ctrl=new THREE.OrbitControls(cam,canvas);ctrl.enableDamping=true;ctrl.dampingFactor=0.08;ctrl.autoRotate=true;ctrl.autoRotateSpeed=0.6;const ctr=[(d.bb[0][0]+d.bb[1][0])/2,(d.bb[0][1]+d.bb[1][1])/2,(d.bb[0][2]+d.bb[1][2])/2];ctrl.target.set(...ctr);scene.add(new THREE.AmbientLight(0xffffff,0.55));const dl1=new THREE.DirectionalLight(0xffffff,0.7);dl1.position.set(500,800,600);scene.add(dl1);const dl2=new THREE.DirectionalLight(0xcbd5e1,0.35);dl2.position.set(-300,-200,-400);scene.add(dl2);
const bSz=[d.bb[1][0]-d.bb[0][0],d.bb[1][1]-d.bb[0][1],d.bb[1][2]-d.bb[0][2]];
// Confinement-volume overlay
const confKind=(d.confinement&&d.confinement.kind)||'box';
if(confKind==='box'){{
  const boxFill=new THREE.Mesh(new THREE.BoxGeometry(...bSz),new THREE.MeshBasicMaterial({{color:0x0ea5e9,transparent:true,opacity:0.05,depthWrite:false,side:THREE.BackSide}}));boxFill.position.set(...ctr);scene.add(boxFill);
  const bE=new THREE.EdgesGeometry(new THREE.BoxGeometry(...bSz));const bL=new THREE.LineSegments(bE,new THREE.LineBasicMaterial({{color:0x64748b,transparent:true,opacity:0.55}}));bL.position.set(...ctr);scene.add(bL);
}}
if(confKind==='mesh'){{
  // Render each named compartment (plasma membrane, nucleus, ER) as a
  // translucent surface mesh from the cellPACK er_peroxisome recipe.
  const meshes=(d.confinement.meshes)||[];
  meshes.forEach(m=>{{
    const g=new THREE.BufferGeometry();
    g.setAttribute('position',new THREE.BufferAttribute(new Float32Array(m.vertices),3));
    g.setIndex(new THREE.BufferAttribute(new Uint32Array(m.faces),1));
    g.computeVertexNormals();
    const col=new THREE.Color(m.color);
    const fill=new THREE.Mesh(g,new THREE.MeshPhongMaterial({{color:col,transparent:true,opacity:m.opacity,depthWrite:false,side:THREE.DoubleSide,shininess:20}}));
    fill.renderOrder=2;scene.add(fill);
    // faint silhouette lines so the mesh is legible through the fill
    const wire=new THREE.LineSegments(new THREE.EdgesGeometry(g,25),new THREE.LineBasicMaterial({{color:col,transparent:true,opacity:Math.min(0.55,m.opacity*2.0),linewidth:1}}));
    wire.renderOrder=3;scene.add(wire);
  }});
  const bE=new THREE.EdgesGeometry(new THREE.BoxGeometry(...bSz));
  const bL=new THREE.LineSegments(bE,new THREE.LineBasicMaterial({{color:0x94a3b8,transparent:true,opacity:0.25}}));
  bL.position.set(...ctr);scene.add(bL);
}}
// Group packed positions by (ingredient-name, radius). Container
// ingredients (e.g. vesicle_shell, dense_core) render as translucent
// shells so the packed interior is visible.
const containers=new Set(d.container_ingredients||[]);
const groups={{}};
for(let i=0;i<d.positions.length;i++){{const nm=d.names?d.names[i]:'';const key=(containers.has(nm)?'C:':'N:')+d.radii[i].toFixed(2);if(!groups[key])groups[key]={{isContainer:containers.has(nm),radius:d.radii[i],indices:[]}};groups[key].indices.push(i)}}
const dm=new THREE.Object3D(),tc=new THREE.Color();
Object.keys(groups).forEach(k=>{{const g=groups[k],r=g.radius,idx=g.indices,sg=new THREE.SphereGeometry(r,g.isContainer?32:16,g.isContainer?24:12);const mt=g.isContainer?new THREE.MeshPhongMaterial({{shininess:30,specular:0x222222,transparent:true,opacity:0.18,depthWrite:false,side:THREE.DoubleSide}}):new THREE.MeshPhongMaterial({{shininess:60,specular:0x444444}});const im=new THREE.InstancedMesh(sg,mt,idx.length);im.renderOrder=g.isContainer?1:0;for(let j=0;j<idx.length;j++){{const i=idx[j];dm.position.set(d.positions[i][0],d.positions[i][1],d.positions[i][2]);dm.updateMatrix();im.setMatrixAt(j,dm.matrix);tc.setRGB(d.colors[i][0],d.colors[i][1],d.colors[i][2]);im.setColorAt(j,tc)}}im.instanceMatrix.needsUpdate=true;if(im.instanceColor)im.instanceColor.needsUpdate=true;scene.add(im);
  // Draw a thin wireframe over container shells so the confinement
  // boundary stays visible through the translucent fill.
  if(g.isContainer){{const wg=new THREE.SphereGeometry(r,28,18);const wm=new THREE.MeshBasicMaterial({{color:0x1e293b,wireframe:true,transparent:true,opacity:0.12}});const wim=new THREE.InstancedMesh(wg,wm,idx.length);for(let j=0;j<idx.length;j++){{const i=idx[j];dm.position.set(d.positions[i][0],d.positions[i][1],d.positions[i][2]);dm.updateMatrix();wim.setMatrixAt(j,dm.matrix)}}wim.instanceMatrix.needsUpdate=true;scene.add(wim)}}
}});
(function a(){{requestAnimationFrame(a);ctrl.update();renderer.render(scene,cam)}})()}});

// Plotly
const pL={{paper_bgcolor:'#f8fafc',plot_bgcolor:'#f8fafc',font:{{color:'#64748b',family:'-apple-system,sans-serif',size:11}},margin:{{l:55,r:15,t:35,b:45}},xaxis:{{gridcolor:'#e2e8f0',zerolinecolor:'#e2e8f0'}},yaxis:{{gridcolor:'#e2e8f0',zerolinecolor:'#e2e8f0'}}}};
const pC={{responsive:true,displayModeBar:false}};
function rgb(c){{return'rgb('+Math.round(c[0]*255)+','+Math.round(c[1]*255)+','+Math.round(c[2]*255)+')'}}
function barChart(el,d){{const n=Object.keys(d.ingr_counts),v=n.map(x=>d.ingr_counts[x]),c=n.map(x=>rgb(d.ingr_info[x].color));Plotly.newPlot(el,[{{x:n,y:v,type:'bar',marker:{{color:c}},text:v,textposition:'auto'}}],{{...pL,title:{{text:'Ingredient Counts',font:{{size:12,color:'#334155'}}}},yaxis:{{...pL.yaxis,title:{{text:'Count',font:{{size:10}}}}}}}},pC)}}
function nnChart(el,nn){{Plotly.newPlot(el,[{{x:nn,type:'histogram',nbinsx:25,marker:{{color:'#10b981',line:{{width:.5,color:'#059669'}}}}}}],{{...pL,title:{{text:'Nearest Neighbor Distance',font:{{size:12,color:'#334155'}}}},xaxis:{{...pL.xaxis,title:{{text:'Distance (nm)',font:{{size:10}}}}}},yaxis:{{...pL.yaxis,title:{{text:'Count',font:{{size:10}}}}}}}},pC)}}
function pieChart(el,d){{const n=Object.keys(d.ingr_counts),v=n.map(x=>{{const r=d.ingr_info[x].radius;return d.ingr_counts[x]*(4/3)*Math.PI*r*r*r}}),c=n.map(x=>rgb(d.ingr_info[x].color));Plotly.newPlot(el,[{{labels:n,values:v,type:'pie',marker:{{colors:c}},textinfo:'label+percent',hole:.35}}],{{...pL,title:{{text:'Volume by Ingredient',font:{{size:12,color:'#334155'}}}},showlegend:false,margin:{{l:20,r:20,t:40,b:20}}}},pC)}}

// ── Plasma ──
(function(){{const d=DATA.plasma;if(!d)return;
barChart('chart-a-plasma',d);
Plotly.newPlot('chart-b-plasma',[{{x:d.extra.radial,type:'histogram',nbinsx:30,marker:{{color:'#6366f1',line:{{width:.5,color:'#4338ca'}}}}}}],{{...pL,title:{{text:'Distance from Box Center',font:{{size:12,color:'#334155'}}}},xaxis:{{...pL.xaxis,title:{{text:'Distance (nm)',font:{{size:10}}}}}},yaxis:{{...pL.yaxis,title:{{text:'Count',font:{{size:10}}}}}}}},pC);
nnChart('chart-c-plasma',d.nn);
pieChart('chart-d-plasma',d)}})();

// ── Vesicle ──
(function(){{const d=DATA.vesicle;if(!d)return;
barChart('chart-a-vesicle',d);
Plotly.newPlot('chart-b-vesicle',[{{x:d.extra.radial,type:'histogram',nbinsx:30,marker:{{color:'#10b981',line:{{width:.5,color:'#059669'}}}}}}],{{...pL,title:{{text:'Distance from Origin (Vesicle Center)',font:{{size:12,color:'#334155'}}}},xaxis:{{...pL.xaxis,title:{{text:'Distance (nm)',font:{{size:10}}}}}},yaxis:{{...pL.yaxis,title:{{text:'Count',font:{{size:10}}}}}}}},pC);
nnChart('chart-c-vesicle',d.nn);
pieChart('chart-d-vesicle',d)}})();

// ── Gradient ──
(function(){{const d=DATA.gradient;if(!d)return;
Plotly.newPlot('chart-a-gradient',[
{{x:d.extra.perox_x,type:'histogram',nbinsx:25,name:'Peroxisome (X-grad)',marker:{{color:'rgba(51,184,26,0.55)',line:{{width:.5,color:'#15803d'}}}},opacity:.7}},
{{x:d.extra.all_x,type:'histogram',nbinsx:25,name:'All molecules',marker:{{color:'rgba(100,116,139,0.3)',line:{{width:.5,color:'#64748b'}}}},opacity:.5}},
],{{...pL,barmode:'overlay',title:{{text:'X-Position: Peroxisomes vs All',font:{{size:12,color:'#334155'}}}},xaxis:{{...pL.xaxis,title:{{text:'X (nm)',font:{{size:10}}}}}},yaxis:{{...pL.yaxis,title:{{text:'Count',font:{{size:10}}}}}},legend:{{font:{{size:9}},bgcolor:'rgba(0,0,0,0)',x:.5,y:.95}}}},pC);
Plotly.newPlot('chart-b-gradient',[
{{x:d.extra.endo_y,type:'histogram',nbinsx:25,name:'Endosome (Y-grad)',marker:{{color:'rgba(255,214,0,0.6)',line:{{width:.5,color:'#a16207'}}}},opacity:.7}},
{{x:d.extra.all_y,type:'histogram',nbinsx:25,name:'All molecules',marker:{{color:'rgba(100,116,139,0.3)',line:{{width:.5,color:'#64748b'}}}},opacity:.5}},
],{{...pL,barmode:'overlay',title:{{text:'Y-Position: Endosomes vs All',font:{{size:12,color:'#334155'}}}},xaxis:{{...pL.xaxis,title:{{text:'Y (nm)',font:{{size:10}}}}}},yaxis:{{...pL.yaxis,title:{{text:'Count',font:{{size:10}}}}}},legend:{{font:{{size:9}},bgcolor:'rgba(0,0,0,0)',x:.5,y:.95}}}},pC);
nnChart('chart-c-gradient',d.nn);
barChart('chart-d-gradient',d)}})();

// ── Partner ──
(function(){{const d=DATA.partner;if(!d)return;
Plotly.newPlot('chart-a-partner',[
{{x:d.extra.lig_rec,type:'histogram',nbinsx:25,name:'Ligand \u2192 Receptor',marker:{{color:'rgba(255,105,0,0.6)',line:{{width:.5,color:'#c65000'}}}},opacity:.7}},
{{x:d.extra.crowd_rec,type:'histogram',nbinsx:25,name:'Crowder \u2192 Receptor',marker:{{color:'rgba(160,160,160,0.5)',line:{{width:.5,color:'#777'}}}},opacity:.6}},
],{{...pL,barmode:'overlay',title:{{text:'Distance to Nearest Receptor',font:{{size:12,color:'#334155'}}}},xaxis:{{...pL.xaxis,title:{{text:'Distance (nm)',font:{{size:10}}}}}},yaxis:{{...pL.yaxis,title:{{text:'Count',font:{{size:10}}}}}},legend:{{font:{{size:9}},bgcolor:'rgba(0,0,0,0)',x:.35,y:.95}}}},pC);
barChart('chart-b-partner',d);
nnChart('chart-c-partner',d.nn);
pieChart('chart-d-partner',d)}})();

// ── Membrane-biased peroxisomes ──
(function(){{const d=DATA.membrane;if(!d)return;
Plotly.newPlot('chart-a-membrane',[
{{x:d.extra.perox_r,type:'histogram',nbinsx:28,name:'Peroxisomes',marker:{{color:'rgba(51,184,26,0.65)',line:{{width:.5,color:'#15803d'}}}},opacity:.8}},
{{x:d.extra.ribo_r,type:'histogram',nbinsx:28,name:'Ribosomes',marker:{{color:'rgba(100,116,139,0.45)',line:{{width:.5,color:'#64748b'}}}},opacity:.7}},
],{{...pL,barmode:'overlay',title:{{text:'Distance from Cell Center (radial bias)',font:{{size:12,color:'#334155'}}}},xaxis:{{...pL.xaxis,title:{{text:'Radial distance (nm)',font:{{size:10}}}}}},yaxis:{{...pL.yaxis,title:{{text:'Count',font:{{size:10}}}}}},legend:{{font:{{size:9}},bgcolor:'rgba(0,0,0,0)',x:.05,y:.95}},shapes:[{{type:'line',x0:d.extra.half_width,x1:d.extra.half_width,yref:'paper',y0:0,y1:1,line:{{color:'#0d9488',width:1.5,dash:'dash'}}}}],annotations:[{{x:d.extra.half_width,yref:'paper',y:1,text:'membrane',showarrow:false,xanchor:'right',font:{{size:9,color:'#0d9488'}}}}]}},pC);
barChart('chart-b-membrane',d);
nnChart('chart-c-membrane',d.nn);
pieChart('chart-d-membrane',d)}})();
</script></body></html>"""

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
    subprocess.run(['open', '-a', 'Safari', output_path])


if __name__ == '__main__':
    run_demo()
