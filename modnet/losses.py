"""Loss utilities for MODNet models."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Union

import numpy as np
import tensorflow as tf

__all__ = ("MAENanLoss", "MAENanMetric", "resolve_loss", "check_nan_targets")


class MAENanLoss(tf.keras.losses.Loss):
    """Mean absolute error that ignores NaN targets.

    NaN entries in `y_true` are masked, so only observed values contribute to
    the loss and its gradient. This is what allows training on partially
    labelled targets.
    """

    def __init__(self, name: str = "maen", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        mask = tf.math.is_nan(y_true)
        safe_y_true = tf.where(mask, y_pred, y_true)
        err = tf.where(mask, tf.zeros_like(y_pred), tf.abs(safe_y_true - y_pred))
        per_sample = tf.reduce_sum(err, axis=-1)
        n_obs = tf.maximum(
            tf.reduce_sum(tf.cast(tf.logical_not(mask), tf.float32)), 1.0
        )
        return per_sample * tf.cast(tf.size(per_sample), tf.float32) / n_obs


class MAENanMetric(tf.keras.metrics.Metric):
    """Mean absolute error metric that ignores NaN targets.

    Drop-in replacement for the built-in 'mae' metric; works correctly
    when targets contain NaN values (partial labels).
    Named 'mae' by default so Keras history keys are unchanged.
    """

    def __init__(self, name: str = "mae", **kwargs):
        super().__init__(name=name, **kwargs)
        self._sum = self.add_weight(name="sum", initializer="zeros")
        self._count = self.add_weight(name="count", initializer="zeros")

    def update_state(
        self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None
    ) -> None:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        mask = tf.math.is_nan(y_true)
        safe_y_true = tf.where(mask, y_pred, y_true)
        abs_err = tf.where(mask, tf.zeros_like(y_pred), tf.abs(safe_y_true - y_pred))
        self._sum.assign_add(tf.reduce_sum(abs_err))
        self._count.assign_add(tf.reduce_sum(tf.cast(tf.logical_not(mask), tf.float32)))

    def result(self) -> tf.Tensor:
        return tf.math.divide_no_nan(self._sum, self._count)

    def reset_state(self) -> None:
        self._sum.assign(0.0)
        self._count.assign(0.0)


LOSS_ALIASES: Dict[str, Callable[[], tf.keras.losses.Loss]] = {
    "maen": MAENanLoss,
}


def resolve_loss(loss: Union[str, tf.keras.losses.Loss, None]):
    """Return a loss object from a string alias, if provided.

    Anything that is not a known alias is returned unchanged, so built-in
    Keras names such as "mse" still reach `compile()` untouched.
    """

    if isinstance(loss, str):
        normalized = loss.lower()
        alias = LOSS_ALIASES.get(normalized)
        if alias is not None:
            return alias()
    return loss


def check_nan_targets(loss: Any, targets: Iterable) -> None:
    """Raises if `targets` hold NaN values that `loss` cannot handle.

    The loss is run on a NaN sample rather than judged by its name, so custom
    callables are covered too. The gradient is checked as well as the value:
    masking with `tf.where` over a difference that is NaN in the discarded
    branch gives a finite loss whose gradient is still NaN.

    Requires eager execution, and assumes the loss accepts a single-column
    target.

    Args:
        loss: The loss about to be handed to `compile(...)`, as a string alias,
            a `tf.keras.losses.Loss` instance or any callable.
        targets (Iterable): The per-output target arrays about to be fitted.

    Raises:
        ValueError: If a target holds NaN and `loss` propagates it to its value
            or to its gradient.

    """
    if not any(
        np.asarray(target).dtype.kind == "f" and np.isnan(target).any()
        for target in targets
    ):
        return

    y_true = tf.constant([[1.0], [np.nan]])
    y_pred = tf.zeros_like(y_true)
    with tf.GradientTape() as tape:
        tape.watch(y_pred)
        value = tf.keras.losses.get(loss)(y_true, y_pred)
    gradient = tape.gradient(value, y_pred)

    if not tf.reduce_all(tf.math.is_finite(value)):
        culprit = "value"
    elif gradient is None or not tf.reduce_all(tf.math.is_finite(gradient)):
        culprit = "gradient"
    else:
        return

    name = loss if isinstance(loss, str) else getattr(loss, "name", repr(loss))
    raise ValueError(
        f"Target values contain NaN and the loss {name!r} propagates them to "
        f"its {culprit}, so training would silently produce NaN weights. "
        "Pass a NaN-aware loss to train on partially-labelled targets."
    )
