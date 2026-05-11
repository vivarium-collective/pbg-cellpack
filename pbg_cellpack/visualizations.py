"""Visualization Step subclasses for pbg-cellpack.

Visualizations follow the pbg-superpowers convention (v0.4.15+): each
subclass overrides `update()` to consume per-step state via wires (like
an Emitter), accumulates history internally, and returns
``{'html': '<rendered figure>'}`` each step. The composite spec wires
the input ports to store paths.

See pbg_superpowers.visualization for the base-class contract.
"""
from __future__ import annotations
import json

from pbg_superpowers.visualization import Visualization


class PackingPlots(Visualization):
    """Interactive Plotly figure of a cellPACK packing result.

    Consumes the core cellPACK outputs (positions, radii, ingredient names
    + colors, packing summary) at each step, accumulates a small history
    of summary scalars (n_packed, packing_fraction) across calls, and
    emits a self-contained Plotly HTML figure on every update.

    The rendered HTML shows:
      * a 2D scatter of the packed sphere centers (XY-projection, sized
        by radius, colored by ingredient color)
      * an ingredient-count bar chart
      * a small time-series of packing fraction across update calls
    """

    config_schema = {
        'title': {'_type': 'string', '_default': 'cellPACK packing'},
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.times: list[float] = []
        self.n_packed_history: list[int] = []
        self.packing_fraction_history: list[float] = []

    def inputs(self):
        return {
            'positions': 'list',
            'radii': 'list',
            'ingredient_names': 'list',
            'ingredient_colors': 'list',
            'n_packed': 'integer',
            'packing_fraction': 'float',
            'time': 'float',
        }

    def update(self, state, interval=1.0):
        positions = state.get('positions') or []
        radii = state.get('radii') or []
        names = state.get('ingredient_names') or []
        colors = state.get('ingredient_colors') or []
        n_packed = int(state.get('n_packed') or 0)
        pf = float(state.get('packing_fraction') or 0.0)

        self.times.append(float(state.get('time', len(self.times) * (interval or 1.0))))
        self.n_packed_history.append(n_packed)
        self.packing_fraction_history.append(pf)

        title = (self.config or {}).get('title', 'cellPACK packing')

        # 2D XY scatter: one point per sphere, colored & sized.
        xs = [float(p[0]) for p in positions]
        ys = [float(p[1]) for p in positions]
        marker_sizes = [max(4.0, float(r) * 0.4) for r in radii]
        marker_colors = [
            'rgb({},{},{})'.format(
                int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))
            for c in colors
        ]
        hover = [
            f'{names[i]}<br>r={radii[i]:.1f}' if i < len(names) else ''
            for i in range(len(positions))
        ]
        scatter_trace = {
            'x': xs, 'y': ys,
            'mode': 'markers',
            'type': 'scatter',
            'marker': {
                'size': marker_sizes,
                'color': marker_colors,
                'line': {'width': 0.5, 'color': '#1e293b'},
            },
            'text': hover,
            'hoverinfo': 'text',
            'name': 'spheres',
        }

        # Ingredient bar chart.
        counts: dict[str, int] = {}
        rep_colors: dict[str, str] = {}
        for i, nm in enumerate(names):
            counts[nm] = counts.get(nm, 0) + 1
            if nm not in rep_colors and i < len(colors):
                c = colors[i]
                rep_colors[nm] = 'rgb({},{},{})'.format(
                    int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))
        bar_keys = list(counts.keys())
        bar_trace = {
            'x': bar_keys,
            'y': [counts[k] for k in bar_keys],
            'type': 'bar',
            'marker': {'color': [rep_colors.get(k, '#94a3b8') for k in bar_keys]},
            'xaxis': 'x2',
            'yaxis': 'y2',
            'name': 'ingredients',
        }

        # Packing-fraction time-series (history across update() calls).
        pf_trace = {
            'x': self.times,
            'y': self.packing_fraction_history,
            'type': 'scatter',
            'mode': 'lines+markers',
            'line': {'color': '#6366f1'},
            'xaxis': 'x3',
            'yaxis': 'y3',
            'name': 'packing fraction',
        }

        layout = {
            'title': title,
            'margin': {'l': 55, 'r': 15, 't': 45, 'b': 40},
            'grid': {'rows': 1, 'columns': 3, 'pattern': 'independent'},
            'xaxis':  {'title': 'x (nm)', 'domain': [0.00, 0.33]},
            'yaxis':  {'title': 'y (nm)', 'scaleanchor': 'x', 'scaleratio': 1},
            'xaxis2': {'title': 'ingredient', 'domain': [0.40, 0.66]},
            'yaxis2': {'title': 'count', 'anchor': 'x2'},
            'xaxis3': {'title': 'time', 'domain': [0.73, 1.00]},
            'yaxis3': {'title': 'packing fraction', 'anchor': 'x3'},
            'showlegend': False,
        }

        data_js = json.dumps([scatter_trace, bar_trace, pf_trace])
        layout_js = json.dumps(layout)

        html = (
            f'<div id="pp" style="height:420px"></div>'
            f'<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'
            f'<script>Plotly.newPlot("pp",{data_js},{layout_js},'
            f'{{responsive:true,displayModeBar:false}});</script>'
        )
        return {'html': html}
