"""CellPACK process-bigraph wrapper.

Wraps the cellPACK mesoscale molecular packing tool as a process-bigraph
Step.  Given a recipe (objects + composition + bounding box) and packing
parameters, produces a packed 3-D molecular configuration.
"""

import json
import logging
import tempfile
import os

import numpy as np
from process_bigraph import Step

log = logging.getLogger(__name__)


class CellPackStep(Step):
    """One-shot molecular packing via cellPACK.

    The recipe is specified in the ``recipe`` config key using cellPACK v2
    format.  On each ``update`` call a fresh packing is generated using the
    seed drawn from the input port, allowing stochastic resampling.
    """

    config_schema = {
        'place_method': {'_type': 'string', '_default': 'jitter'},
        'inner_grid_method': {'_type': 'string', '_default': 'trimesh'},
        'use_periodicity': {'_type': 'boolean', '_default': False},
        'show_progress_bar': {'_type': 'boolean', '_default': False},
    }

    def __init__(self, config=None, core=None):
        # Stash the recipe before super().__init__ validates the schema;
        # the recipe dict is too deeply nested for bigraph-schema's type
        # system, so we keep it as a plain Python dict.
        config = dict(config) if config else {}
        self._recipe = config.pop('recipe', {})
        super().__init__(config=config, core=core)
        self._env = None

    # -- ports ----------------------------------------------------------------

    def inputs(self):
        return {'seed': 'integer'}

    def outputs(self):
        return {
            'positions': 'list',
            'radii': 'list',
            'ingredient_names': 'list',
            'ingredient_colors': 'list',
            'n_packed': 'integer',
            'packing_fraction': 'float',
        }

    # -- internals ------------------------------------------------------------

    def _run_packing(self, seed=0):
        """Execute cellPACK packing and return extracted results."""
        from cellpack.autopack.loaders.config_loader import ConfigLoader
        from cellpack.autopack.loaders.recipe_loader import RecipeLoader
        from cellpack.autopack.Environment import Environment
        from cellpack.autopack import upy

        recipe_dict = dict(self._recipe)

        # Write recipe to temporary file for the loader
        recipe_fd, recipe_path = tempfile.mkstemp(suffix='.json')
        try:
            with os.fdopen(recipe_fd, 'w') as f:
                json.dump(recipe_dict, f)

            # Build packing config
            outdir = tempfile.mkdtemp(prefix='pbg_cellpack_')
            config_dict = {
                'name': 'pbg_cellpack',
                'place_method': self.config['place_method'],
                'overwrite_place_method': True,
                'inner_grid_method': self.config['inner_grid_method'],
                'use_periodicity': self.config['use_periodicity'],
                'show_progress_bar': self.config['show_progress_bar'],
                'out': outdir,
                'randomness_seed': int(seed),
            }
            config_fd, config_path = tempfile.mkstemp(suffix='.json')
            with os.fdopen(config_fd, 'w') as f:
                json.dump(config_dict, f)

            config_loader = ConfigLoader(config_path)
            recipe_loader = RecipeLoader(recipe_path, save_converted_recipe=False)

            helper = upy.getHelperClass()()
            env = Environment(
                config=config_loader.config,
                recipe=recipe_loader.recipe_data,
            )
            env.helper = helper

            env.buildGrid()
            try:
                env.pack_grid(seedNum=int(seed))
            except Exception as exc:
                # cellPACK may error during the post-pack save step; the
                # packing results are still available on env.packed_objects.
                log.debug('Ignoring post-pack error: %s', exc)

            os.unlink(config_path)
        finally:
            if os.path.exists(recipe_path):
                os.unlink(recipe_path)

        return self._extract_results(env)

    @staticmethod
    def _extract_results(env):
        """Pull positions, radii, names, colors from a packed Environment."""
        po = env.packed_objects
        raw_positions = po.get_positions()
        raw_radii = po.get_radii()
        ingredients = po.get_ingredients()

        positions = [[float(p[0]), float(p[1]), float(p[2])]
                     for p in raw_positions]
        radii = [float(r) for r in raw_radii]
        names = [str(obj.name) for obj in ingredients]
        colors = [
            [float(c) for c in obj.color]
            if obj.color is not None else [0.5, 0.5, 0.5]
            for obj in ingredients
        ]

        n_packed = len(positions)

        # Compute packing fraction
        bb = np.array(env.boundingBox)
        bb_vol = float(np.prod(bb[1] - bb[0]))
        sphere_vol = float(sum(4.0 / 3.0 * np.pi * r ** 3 for r in radii))
        packing_fraction = sphere_vol / bb_vol if bb_vol > 0 else 0.0

        return {
            'positions': positions,
            'radii': radii,
            'ingredient_names': names,
            'ingredient_colors': colors,
            'n_packed': n_packed,
            'packing_fraction': packing_fraction,
        }

    # -- Step interface -------------------------------------------------------

    def update(self, state):
        seed = state.get('seed', 0)
        return self._run_packing(seed=seed)
