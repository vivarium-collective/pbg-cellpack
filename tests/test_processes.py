"""Unit tests for CellPackStep."""

import pytest
from process_bigraph import allocate_core
from pbg_cellpack.processes import CellPackStep


SIMPLE_RECIPE = {
    'version': '1.0.0',
    'format_version': '2.0',
    'name': 'test_recipe',
    'bounding_box': [[0, 0, 0], [300, 300, 300]],
    'objects': {
        'base': {
            'type': 'single_sphere',
            'place_method': 'jitter',
            'jitter_attempts': 10,
            'packing_mode': 'random',
            'max_jitter': [1, 1, 1],
        },
        'sphere_a': {
            'type': 'single_sphere',
            'inherit': 'base',
            'color': [0.3, 0.6, 0.9],
            'radius': 30,
        },
        'sphere_b': {
            'type': 'single_sphere',
            'inherit': 'base',
            'color': [0.9, 0.3, 0.2],
            'radius': 20,
        },
    },
    'composition': {
        'space': {'regions': {'interior': ['A', 'B']}},
        'A': {'object': 'sphere_a', 'count': 5},
        'B': {'object': 'sphere_b', 'count': 10},
    },
}


@pytest.fixture
def core():
    c = allocate_core()
    c.register_link('CellPackStep', CellPackStep)
    return c


@pytest.fixture
def step(core):
    return CellPackStep(config={'recipe': SIMPLE_RECIPE}, core=core)


def test_instantiation(step):
    """CellPackStep can be instantiated with a recipe config."""
    assert step._recipe['name'] == 'test_recipe'
    assert step.config['place_method'] == 'jitter'


def test_ports(step):
    """Inputs and outputs declare expected ports."""
    assert 'seed' in step.inputs()
    outs = step.outputs()
    for key in ('positions', 'radii', 'ingredient_names',
                'ingredient_colors', 'n_packed', 'packing_fraction'):
        assert key in outs, f'missing output port: {key}'


def test_update_returns_results(step):
    """A single update() call returns packed objects."""
    result = step.update({'seed': 42})
    assert isinstance(result, dict)
    assert result['n_packed'] > 0
    assert len(result['positions']) == result['n_packed']
    assert len(result['radii']) == result['n_packed']
    assert len(result['ingredient_names']) == result['n_packed']
    assert len(result['ingredient_colors']) == result['n_packed']


def test_positions_are_3d(step):
    """Each position is a 3-element list of floats."""
    result = step.update({'seed': 1})
    for pos in result['positions']:
        assert len(pos) == 3
        assert all(isinstance(v, float) for v in pos)


def test_packing_fraction_positive(step):
    """Packing fraction is positive when objects are placed."""
    result = step.update({'seed': 7})
    assert result['packing_fraction'] > 0.0
    assert result['packing_fraction'] < 1.0


def test_config_defaults(core):
    """Unspecified config fields use defaults."""
    step = CellPackStep(config={'recipe': SIMPLE_RECIPE}, core=core)
    assert step.config['place_method'] == 'jitter'
    assert step.config['inner_grid_method'] == 'trimesh'
    assert step.config['use_periodicity'] is False
