"""Integration tests for composite assembly."""

import pytest
from process_bigraph import Composite, allocate_core
from process_bigraph.emitter import RAMEmitter
from pbg_cellpack.processes import CellPackStep
from pbg_cellpack.composites import make_packing_document


SIMPLE_RECIPE = {
    'version': '1.0.0',
    'format_version': '2.0',
    'name': 'test_composite',
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
            'color': [0.2, 0.7, 0.4],
            'radius': 25,
        },
    },
    'composition': {
        'space': {'regions': {'interior': ['A']}},
        'A': {'object': 'sphere_a', 'count': 8},
    },
}


@pytest.fixture
def core():
    c = allocate_core()
    c.register_link('CellPackStep', CellPackStep)
    c.register_link('ram-emitter', RAMEmitter)
    return c


def test_document_factory():
    """make_packing_document returns a well-formed document."""
    doc = make_packing_document(recipe=SIMPLE_RECIPE, seed=42)
    assert 'packer' in doc
    assert 'stores' in doc
    assert 'emitter' in doc
    assert doc['stores']['seed'] == 42


def test_document_factory_params():
    """Custom parameters propagate into the document."""
    doc = make_packing_document(
        recipe=SIMPLE_RECIPE,
        place_method='jitter',
        seed=99,
    )
    assert doc['packer']['config']['place_method'] == 'jitter'
    assert doc['stores']['seed'] == 99


def test_composite_assembly(core):
    """Composite can be assembled from the document."""
    doc = make_packing_document(recipe=SIMPLE_RECIPE, seed=42)
    sim = Composite({'state': doc}, core=core)
    assert sim is not None
