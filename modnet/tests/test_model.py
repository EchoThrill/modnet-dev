#!/usr/bin/env python
import pytest
import numpy as np


def test_train_small_model_single_target(subset_moddata, tf_session):
    """Tests the single target training."""
    from modnet.models import MODNetModel

    data = subset_moddata
    # set 'optimal' features manually
    data.optimal_features = [
        col for col in data.df_featurized.columns if col.startswith("ElementProperty")
    ]

    model = MODNetModel(
        [[["eform"]]],
        weights={"eform": 1},
        num_neurons=[[16], [8], [8], [4]],
        n_feat=10,
    )

    model.fit(data, epochs=2)
    model.predict(data)
    assert not np.isnan(model.evaluate(data))


def test_train_small_model_single_target_classif(subset_moddata, tf_session):
    """Tests the single target training."""
    from modnet.models import MODNetModel

    data = subset_moddata
    # set 'optimal' features manually
    data.optimal_features = [
        col for col in data.df_featurized.columns if col.startswith("ElementProperty")
    ]

    def is_metal(egap):
        if egap == 0:
            return 1
        else:
            return 0

    data.df_targets["is_metal"] = data.df_targets["egap"].apply(is_metal)
    model = MODNetModel(
        [[["is_metal"]]],
        weights={"is_metal": 1},
        num_neurons=[[16], [8], [8], [4]],
        num_classes={"is_metal": 2},
        n_feat=10,
    )

    model.fit(data, epochs=2)
    assert not np.isnan(model.evaluate(data))


def test_train_small_model_multi_target(subset_moddata, tf_session):
    """Tests the multi-target training."""
    from modnet.models import MODNetModel

    data = subset_moddata
    # set 'optimal' features manually
    data.optimal_features = [
        col for col in data.df_featurized.columns if col.startswith("ElementProperty")
    ]

    model = MODNetModel(
        [[["eform", "egap"]]],
        weights={"eform": 1, "egap": 1},
        num_neurons=[[16], [8], [8], [4]],
        n_feat=10,
    )

    model.fit(data, epochs=2)
    model.predict(data)
    assert not np.isnan(model.evaluate(data))


def test_train_small_model_presets(subset_moddata, tf_session):
    """Tests the `fit_preset()` method."""
    from modnet.model_presets import gen_presets
    from modnet.models import MODNetModel

    modified_presets = gen_presets(100, 100)[:2]

    for ind, preset in enumerate(modified_presets):
        modified_presets[ind]["epochs"] = 2

    data = subset_moddata
    # set 'optimal' features manually
    data.optimal_features = [
        col for col in data.df_featurized.columns if col.startswith("ElementProperty")
    ]

    model = MODNetModel(
        [[["eform", "egap"]]],
        weights={"eform": 1, "egap": 1},
        num_neurons=[[16], [8], [8], [4]],
        n_feat=10,
    )

    # nested=0/False -> no inner loop, so only 1 model
    # nested=1/True -> inner loop, but default n_folds so 5
    for num_nested, nested_option in zip([5, 1, 5], [5, 0, 1]):
        results = model.fit_preset(
            data,
            presets=modified_presets,
            nested=nested_option,
            val_fraction=0.2,
            n_jobs=1,
            refit=True,
        )
        models = results[0]
        assert len(models) == len(modified_presets)
        assert len(models[0]) == num_nested
    assert not np.isnan(model.evaluate(data))


def test_model_integration(subset_moddata, tf_session):
    """Tests training, saving, loading and predictions."""
    from modnet.models import MODNetModel

    data = subset_moddata
    # set 'optimal' features manually
    data.optimal_features = [
        col for col in data.df_featurized.columns if col.startswith("ElementProperty")
    ][:10]

    model = MODNetModel(
        [[["eform", "egap"]]],
        weights={"eform": 1, "egap": 1},
        num_neurons=[[16], [8], [8], [4]],
        n_feat=10,
    )

    model.fit(data, epochs=2)

    model.save("test")
    loaded_model = MODNetModel.load("test")

    assert model.predict(data).equals(loaded_model.predict(data))
    assert not np.isnan(model.evaluate(data))


@pytest.mark.deprecated
def test_train_small_bayesian_single_target(subset_moddata, tf_session):
    """Tests the single target training."""
    from modnet.models import BayesianMODNetModel

    data = subset_moddata
    # set 'optimal' features manually
    data.optimal_features = [
        col for col in data.df_featurized.columns if col.startswith("ElementProperty")
    ]

    model = BayesianMODNetModel(
        [[["eform"]]],
        weights={"eform": 1},
        num_neurons=[[16], [8], [8], [4]],
        n_feat=10,
    )

    model.fit(data, epochs=2)
    model.predict(data)
    model.predict(data, return_unc=True)
    assert not np.isnan(model.evaluate(data))


@pytest.mark.deprecated
def test_train_small_bayesian_single_target_classif(subset_moddata, tf_session):
    """Tests the single target training."""
    from modnet.models import BayesianMODNetModel

    data = subset_moddata
    # set 'optimal' features manually
    data.optimal_features = [
        col for col in data.df_featurized.columns if col.startswith("ElementProperty")
    ]

    def is_metal(egap):
        if egap == 0:
            return 1
        else:
            return 0

    data.df_targets["is_metal"] = data.df_targets["egap"].apply(is_metal)
    model = BayesianMODNetModel(
        [[["is_metal"]]],
        weights={"is_metal": 1},
        num_neurons=[[16], [8], [8], [4]],
        num_classes={"is_metal": 2},
        n_feat=10,
    )

    model.fit(data, epochs=2)
    model.predict(data)
    model.predict(data, return_unc=True)
    assert not np.isnan(model.evaluate(data))


@pytest.mark.deprecated
def test_train_small_bayesian_multi_target(subset_moddata, tf_session):
    """Tests the multi-target training."""
    from modnet.models import BayesianMODNetModel

    data = subset_moddata
    # set 'optimal' features manually
    data.optimal_features = [
        col for col in data.df_featurized.columns if col.startswith("ElementProperty")
    ]

    model = BayesianMODNetModel(
        [[["eform", "egap"]]],
        weights={"eform": 1, "egap": 1},
        num_neurons=[[16], [8], [8], [4]],
        n_feat=10,
    )

    model.fit(data, epochs=2)
    model.predict(data)
    model.predict(data, return_unc=True)
    assert not np.isnan(model.evaluate(data))


def test_train_small_bootstrap_single_target(subset_moddata, tf_session):
    """Tests the single target training."""
    from modnet.models import EnsembleMODNetModel

    data = subset_moddata
    # set 'optimal' features manually
    data.optimal_features = [
        col for col in data.df_featurized.columns if col.startswith("ElementProperty")
    ]

    model = EnsembleMODNetModel(
        [[["eform"]]],
        weights={"eform": 1},
        num_neurons=[[16], [8], [8], [4]],
        n_feat=10,
        n_models=3,
        bootstrap=True,
    )

    model.fit(data, epochs=2)
    model.predict(data)
    model.predict(data, return_unc=True)
    assert not np.isnan(model.evaluate(data))


def test_train_small_bootstrap_single_target_classif(small_moddata, tf_session):
    """Tests the single target training."""
    from modnet.models import EnsembleMODNetModel

    data = small_moddata
    # set 'optimal' features manually
    data.optimal_features = [
        col for col in data.df_featurized.columns if col.startswith("ElementProperty")
    ]

    def is_metal(egap):
        if egap == 0:
            return 1
        else:
            return 0

    data.df_targets["is_metal"] = data.df_targets["egap"].apply(is_metal)
    model = EnsembleMODNetModel(
        [[["is_metal"]]],
        weights={"is_metal": 1},
        num_neurons=[[16], [8], [8], [4]],
        num_classes={"is_metal": 2},
        n_feat=10,
        n_models=3,
        bootstrap=True,
    )

    model.fit(data, epochs=2)
    model.predict(data)
    model.predict(data, return_prob=False, voting_type="soft", return_unc=True)
    model.predict(data, return_prob=False, voting_type="hard", return_unc=True)
    model.predict(data, return_prob=True, return_unc=True)
    assert not np.isnan(model.evaluate(data))


def test_train_small_bootstrap_multi_target_classif(small_moddata, tf_session):
    """Tests the multi target classification training."""
    from modnet.models import EnsembleMODNetModel

    data = small_moddata
    # set 'optimal' features manually
    data.optimal_features = [
        col for col in data.df_featurized.columns if col.startswith("ElementProperty")
    ]

    def is_metal(egap):
        if egap == 0:
            return 1
        else:
            return 0

    def eform_cl(eform):
        if eform > 0:
            return 1
        else:
            return 0

    data.df_targets["is_metal"] = data.df_targets["egap"].apply(is_metal)
    data.df_targets["eform_cl"] = data.df_targets["eform"].apply(eform_cl)
    model = EnsembleMODNetModel(
        [[["eform_cl"], ["is_metal"]]],
        weights={
            "eform_cl": 1,
            "is_metal": 1,
        },
        num_neurons=[[16], [8], [8], [4]],
        num_classes={"eform_cl": 2, "is_metal": 2},
        n_feat=10,
        n_models=3,
        bootstrap=True,
    )

    model.fit(data, epochs=2)
    model.predict(data, return_prob=True, return_unc=True)
    model.predict(data, return_prob=False, voting_type="soft", return_unc=True)
    model.predict(data, return_prob=False, voting_type="hard", return_unc=True)
    assert not np.isnan(model.evaluate(data))


def test_train_small_bootstrap_multi_target(small_moddata, tf_session):
    """Tests the multi-target training."""
    from modnet.models import EnsembleMODNetModel

    data = small_moddata
    # set 'optimal' features manually
    data.optimal_features = [
        col for col in data.df_featurized.columns if col.startswith("ElementProperty")
    ]

    model = EnsembleMODNetModel(
        [[["eform", "egap"]]],
        weights={"eform": 1, "egap": 1},
        num_neurons=[[16], [8], [8], [4]],
        n_feat=10,
        n_models=3,
        bootstrap=True,
    )

    model.fit(data, epochs=2)
    model.predict(data, return_unc=True)


def _partially_label(data, target, n_blank, seed=0):
    """Replace `n_blank` of `target`'s labels with NaN, leaving the rest intact.

    Returns the blanked `data` and the labels that survived.
    """
    rng = np.random.default_rng(seed)
    blanked = rng.choice(len(data.df_targets), size=n_blank, replace=False)
    data.df_targets.loc[data.df_targets.index[blanked], target] = np.nan
    data.optimal_features = [
        col for col in data.df_featurized.columns if col.startswith("ElementProperty")
    ]
    return data, data.df_targets[target].dropna()


def _single_target_model(model_cls=None, **kwargs):
    from modnet.models import MODNetModel

    return (model_cls or MODNetModel)(
        [[["eform"]]],
        weights={"eform": 1},
        num_neurons=[[16], [8], [8], [4]],
        n_feat=10,
        **kwargs,
    )


def test_fit_rejects_nan_targets_under_non_nan_aware_loss(subset_moddata, tf_session):
    """A non-NaN-aware loss must refuse NaN targets up front."""
    data, _ = _partially_label(subset_moddata, "eform", 40)

    with pytest.raises(ValueError, match="propagates them to its value"):
        _single_target_model().fit(data, epochs=2)


def test_train_small_model_partial_labels(subset_moddata, tf_session):
    """`loss='maen'` trains a model end-to-end on a partially-labelled target."""
    data, _ = _partially_label(subset_moddata, "eform", 40)
    assert data.df_targets["eform"].isna().sum() == 40, "fixture should be partial"

    model = _single_target_model()
    model.fit(data, epochs=5, loss="maen")

    assert all(np.isfinite(w).all() for w in model.model.get_weights())

    predictions = model.predict(data)
    assert list(predictions.columns) == ["eform"]
    assert list(predictions.index) == list(data.df_targets.index)
    assert predictions.shape == (len(data.df_targets), 1)
    assert np.isfinite(predictions.values.astype(float)).all()

    assert np.isfinite(model.evaluate(data, loss="maen"))


def test_prediction_bounds_ignore_nan_targets(subset_moddata, tf_session):
    """Prediction clamping bounds come from the observed labels only.

    `min`/`max` over a target holding NaN propagates NaN into the bounds,
    which silently disables the out-of-bounds remapping in `predict`.
    """
    data, observed = _partially_label(subset_moddata, "eform", 40)

    model = _single_target_model()
    model.fit(data, epochs=2, loss="maen")

    assert float(model.min_y[0][0]) == pytest.approx(observed.min())
    assert float(model.max_y[0][0]) == pytest.approx(observed.max())


def test_train_small_bootstrap_partial_labels(subset_moddata, tf_session):
    """Ensembles train on partial labels and forward `loss` down to each member."""
    from modnet.models import EnsembleMODNetModel

    data, _ = _partially_label(subset_moddata, "eform", 40)

    model = _single_target_model(EnsembleMODNetModel, n_models=2, bootstrap=True)
    model.fit(data, epochs=2, loss="maen")

    predictions, stds = model.predict(data, return_unc=True)
    assert list(predictions.columns) == ["eform"]
    assert np.isfinite(predictions.values.astype(float)).all()
    assert stds.shape == predictions.shape
    assert (stds.values.astype(float) >= 0).all()

    assert np.isfinite(model.evaluate(data, loss="maen"))
    with pytest.raises(RuntimeError, match="not recognized"):
        model.evaluate(data, loss="not_a_loss")


@pytest.mark.slow
def test_train_small_bootstrap_presets(small_moddata, tf_session):
    """Tests the `fit_preset()` method."""
    import time

    from modnet.model_presets import gen_presets
    from modnet.models import EnsembleMODNetModel

    start = time.time()
    modified_presets = gen_presets(100, 100)[:2]

    for ind, preset in enumerate(modified_presets):
        modified_presets[ind]["epochs"] = 2

    data = small_moddata
    # set 'optimal' features manually
    data.optimal_features = [
        col for col in data.df_featurized.columns if col.startswith("ElementProperty")
    ]

    model = EnsembleMODNetModel(
        [[["eform", "egap"]]],
        weights={"eform": 1, "egap": 1},
        num_neurons=[[4], [2], [2], [2]],
        n_feat=3,
        n_models=2,
        bootstrap=True,
    )
    print(f"{time.time() - start} elapsed after model creation.")

    # nested=0/False -> no inner loop, so only 1 model
    # nested=1/True -> inner loop, but default n_folds so 5
    for num_nested, nested_option in zip([2, 1], [2, 0]):
        results = model.fit_preset(
            data,
            presets=modified_presets,
            nested=nested_option,
            val_fraction=0.2,
            n_jobs=2,
        )
        print(f"{time.time() - start} elapsed nested {num_nested}, {nested_option}")
        models = results[0]
        assert len(models) == len(modified_presets)
        assert len(models[0]) == num_nested

        assert not np.isnan(model.evaluate(data))
