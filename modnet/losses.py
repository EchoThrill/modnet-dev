import tensorflow as tf

class MAENanLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        mask = tf.math.is_nan(y_true)
        y_true = tf.where(mask, y_pred, y_true)
        diff = tf.abs(y_true - y_pred)
        loss = tf.reduce_sum(diff) / tf.reduce_sum(1 - tf.cast(mask, tf.float32))
        return loss
