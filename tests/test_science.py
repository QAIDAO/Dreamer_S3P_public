from pathlib import Path
import numpy as np
import pytest
from s3p_repro.quantum import simulate, validate_probabilities, FeatureBackend
from s3p_repro.classification import Classifier
from s3p_repro.common import read_csv
from s3p_repro.chemistry import describe_split

ROOT = Path(__file__).resolve().parents[1]

def test_gate_order_and_endianness():
    assert np.argmax(simulate([np.pi, 0, 0])) == 7
    assert np.argmax(simulate([0, 0, np.pi])) == 4
    assert np.argmax(simulate([0, np.pi, 0])) == 6

def test_feature_dot_product_matches_closed_form():
    rng = np.random.default_rng(42)
    for _ in range(10):
        a, b = rng.uniform(0, np.pi, (2, 3))
        expected = np.prod((1 + np.cos(a) * np.cos(b)) / 2)
        assert simulate(a).sum() == pytest.approx(1)
        assert simulate(a) @ simulate(b) == pytest.approx(expected)

@pytest.mark.parametrize("bad", [[0] * 8, [1] * 7, [-1] + [1] * 7, [float("nan")] * 8, [float("inf")] * 8])
def test_bad_quantum_output_is_rejected(bad):
    with pytest.raises(ValueError):
        validate_probabilities(bad)

def test_fixed_scaffold_split_and_no_preprocessor_leakage(tmp_path):
    train = read_csv(ROOT / "data/classification/train_648.csv")
    test = read_csv(ROOT / "data/classification/test_162.csv")
    audit = describe_split(train, test)
    assert audit["shared_murcko_scaffolds"] == audit["shared_canonical_structures"] == 0
    model = Classifier(FeatureBackend(tmp_path)).fit(train)
    mean, components, minimum = model.standard.mean_.copy(), model.pca.components_.copy(), model.angles.data_min_.copy()
    result, predictions = model.evaluate(test)
    assert len(predictions) == 3 * 162
    for value in result.values():
        assert value["TP"] + value["TN"] + value["FP"] + value["FN"] == 162
    np.testing.assert_equal(model.standard.mean_, mean)
    np.testing.assert_equal(model.pca.components_, components)
    np.testing.assert_equal(model.angles.data_min_, minimum)
