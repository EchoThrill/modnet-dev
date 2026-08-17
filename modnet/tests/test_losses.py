#!/usr/bin/env python
import numpy as np
import pytest
import tensorflow as tf

from modnet.losses import MAENanLoss, MAENanMetric, check_nan_targets, resolve_loss


def _gradient(loss, y_true, y_pred):
    """Returns d(loss) / d(y_pred) as a numpy array."""
    prediction = tf.Variable(y_pred)
    with tf.GradientTape() as tape:
        value = loss(y_true, prediction)
    return tape.gradient(value, prediction).numpy()


def test_mae_nan_loss_masking():
    """Masking an entry gives the same loss as deleting it."""
    np.random.seed(42)
    y_true = np.random.normal(0, 5, (20, 3)).astype(np.float32)
    y_pred = np.random.normal(0, 5, (20, 3)).astype(np.float32)
    mask = np.random.rand(20, 3) < 0.4

    masked = MAENanLoss()(tf.constant(np.where(mask, np.nan, y_true)), y_pred)
    deleted = tf.keras.losses.MeanAbsoluteError()(y_true[~mask], y_pred[~mask])

    assert float(masked) == pytest.approx(float(deleted), abs=1e-5)


def test_mae_nan_loss_gradient():
    """Masked entries pull nothing; the observed ones pull what deleting would."""
    np.random.seed(42)
    y_true = np.random.normal(0, 5, (20, 3)).astype(np.float32)
    y_pred = np.random.normal(0, 5, (20, 3)).astype(np.float32)
    mask = np.random.rand(20, 3) < 0.4

    masked = _gradient(
        MAENanLoss(), tf.constant(np.where(mask, np.nan, y_true)), y_pred
    )
    deleted = _gradient(
        tf.keras.losses.MeanAbsoluteError(), y_true[~mask], y_pred[~mask]
    )

    assert (masked[mask] == 0).all()
    np.testing.assert_allclose(masked[~mask], deleted, atol=1e-6)


def test_mae_nan_loss_training():
    """Training on partial labels reaches the same weights as dropping the rows."""
    np.random.seed(42)
    x = np.random.rand(40, 5).astype(np.float32)
    y = (2 * x[:, 0] - x[:, 1]).astype(np.float32).reshape(-1, 1)
    observed = np.random.rand(40) > 0.35

    def create_model(loss, initial_weights=None):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(8, activation="relu", input_shape=(5,)),
                tf.keras.layers.Dense(1),
            ]
        )
        if initial_weights is not None:
            model.set_weights(initial_weights)
        model.compile(optimizer=tf.keras.optimizers.SGD(0.05), loss=loss)
        return model

    masked_model = create_model(MAENanLoss())
    deleted_model = create_model("mae", initial_weights=masked_model.get_weights())

    fit_kwargs = dict(epochs=30, shuffle=False, verbose=0)
    masked_history = masked_model.fit(
        x, np.where(observed[:, None], y, np.nan), batch_size=40, **fit_kwargs
    ).history["loss"]
    deleted_history = deleted_model.fit(
        x[observed], y[observed], batch_size=int(observed.sum()), **fit_kwargs
    ).history["loss"]

    np.testing.assert_allclose(masked_history, deleted_history, atol=1e-5)
    np.testing.assert_allclose(
        np.concatenate([w.ravel() for w in masked_model.get_weights()]),
        np.concatenate([w.ravel() for w in deleted_model.get_weights()]),
        atol=1e-6,
    )


def test_mae_nan_handling():
    """Observed errors are [1, 1, 2, 2], so the loss is their mean of 1.5."""
    y_true = tf.constant([1.0, 2.0, np.nan, 4.0, 5.0])
    y_pred = tf.constant([2.0, 3.0, 9.0, 6.0, 7.0])

    assert float(MAENanLoss()(y_true, y_pred)) == pytest.approx(1.5)
    # the built-in loss must fail here, or the example proves nothing
    assert np.isnan(tf.keras.losses.MeanAbsoluteError()(y_true, y_pred))


def test_mae_nan_loss_all_nan():
    """A batch holding no labels at all yields no loss and no gradient."""
    loss = MAENanLoss()
    y_true = tf.constant([[np.nan], [np.nan]])
    y_pred = tf.constant([[1.0], [2.0]])

    assert float(loss(y_true, y_pred)) == 0.0
    np.testing.assert_array_equal(_gradient(loss, y_true, y_pred), 0.0)


def test_mae_nan_loss_per_sample():
    """call() leaves the reduction to Keras, so sample_weight still applies."""
    loss = MAENanLoss()
    y_true = tf.constant([[1.0], [np.nan], [3.0]])
    y_pred = tf.constant([[0.5], [9.9], [2.5]])

    assert tuple(loss.call(y_true, y_pred).shape) == (3,)

    unweighted = float(loss(y_true, y_pred))
    weighted = float(loss(y_true, y_pred, sample_weight=tf.constant([1.0, 1.0, 0.0])))

    assert unweighted != pytest.approx(weighted)


def test_mae_nan_metric_matches_loss():
    """The metric masks independently of the loss, so the two must agree."""
    np.random.seed(42)
    y_true = np.random.normal(0, 5, (20, 3)).astype(np.float32)
    y_pred = np.random.normal(0, 5, (20, 3)).astype(np.float32)
    y_true[np.random.rand(20, 3) < 0.4] = np.nan

    metric = MAENanMetric()
    metric.update_state(tf.constant(y_true), y_pred)

    expected = float(MAENanLoss()(tf.constant(y_true), y_pred))
    assert float(metric.result()) == pytest.approx(expected, abs=1e-5)


def test_mae_nan_metric_name():
    """The metric is named 'mae', so callbacks watching val_mae keep working."""
    assert MAENanMetric().name == "mae"

    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(2,))])
    model.compile(optimizer="adam", loss=MAENanLoss(), metrics=[MAENanMetric()])
    history = model.fit(
        np.zeros((4, 2), dtype=np.float32),
        np.array([[1.0], [np.nan], [3.0], [4.0]], dtype=np.float32),
        epochs=1,
        verbose=0,
    )

    assert "mae" in history.history
    assert np.isfinite(history.history["mae"]).all()


def test_mae_nan_metric_multi_batch():
    """State accumulates over batches, counting only the observed entries."""
    metric = MAENanMetric()
    metric.update_state(tf.constant([1.0, 2.0]), tf.constant([2.0, 3.0]))
    metric.update_state(tf.constant([np.nan, 4.0]), tf.constant([9.0, 6.0]))

    assert float(metric.result()) == pytest.approx(4.0 / 3.0)


def test_mae_nan_metric_reset():
    """reset_state zeroes the accumulated state."""
    metric = MAENanMetric()
    metric.update_state(tf.constant([1.0, 2.0]), tf.constant([2.0, 3.0]))
    metric.reset_state()

    assert float(metric.result()) == 0.0


def test_resolve_loss_aliases():
    """resolve_loss returns a MAENanLoss for the alias, whatever its case."""
    for alias in ("maen", "MAEN", "Maen"):
        assert isinstance(resolve_loss(alias), MAENanLoss), f"failed on '{alias}'"


def test_resolve_loss_passthrough():
    """Non-alias strings, existing instances and None come back unchanged."""
    for value in ("mae", "mse", "mae_nan", None):
        assert resolve_loss(value) is value

    instance = MAENanLoss()
    assert resolve_loss(instance) is instance


def test_mae_nan_loss_config_round_trip():
    """MAENanLoss can be rebuilt from its own config, as Keras requires."""
    original = MAENanLoss()
    restored = MAENanLoss.from_config(original.get_config())

    assert isinstance(restored, MAENanLoss)
    assert restored.name == original.name == "maen"

    y_true = tf.constant([[1.0], [np.nan], [3.0]])
    y_pred = tf.constant([[0.5], [9.9], [2.5]])

    assert float(restored(y_true, y_pred)) == pytest.approx(
        float(original(y_true, y_pred))
    )


def test_check_nan_targets():
    """The guard fires only for NaN targets under a loss that propagates them."""
    with pytest.raises(ValueError, match="propagates them to its value"):
        check_nan_targets("mse", [np.array([1.0, np.nan, 3.0])])

    # clean targets, NaN-aware losses and non-float targets are all fine
    check_nan_targets("mse", [np.array([1.0, 2.0])])
    check_nan_targets(MAENanLoss(), [np.array([1.0, np.nan])])
    check_nan_targets("mse", [np.array([1, 2, 3])])


def test_check_nan_targets_gradient():
    """A finite value with a NaN gradient is caught, which a name check misses."""

    def leaky_masked_mae(y_true, y_pred):
        mask = tf.math.is_nan(y_true)
        return tf.reduce_mean(
            tf.where(mask, tf.zeros_like(y_pred), tf.abs(y_true - y_pred))
        )

    y_true = tf.constant([[1.0], [np.nan]])
    assert np.isfinite(leaky_masked_mae(y_true, tf.zeros_like(y_true)))

    with pytest.raises(ValueError, match="propagates them to its gradient"):
        check_nan_targets(leaky_masked_mae, [np.array([1.0, np.nan])])
