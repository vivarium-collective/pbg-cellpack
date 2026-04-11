"""Pre-built composite document factories for cellPACK packings."""


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
