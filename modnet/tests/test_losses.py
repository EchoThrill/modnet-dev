import numpy as np
import tensorflow as tf

from modnet.losses import MAENanLoss


def test_mae_nan_loss_basic():
    """Test the basic functionality of MAENanLoss by comparing it with the Keras MeanAbsoluteError loss."""
    mae_loss = tf.keras.losses.MeanAbsoluteError()
    mae_nan_loss = MAENanLoss()
    
    y_true = tf.random.uniform((10,), minval=0, maxval=1)
    y_pred = tf.random.uniform((10,), minval=0, maxval=1)
    
    loss_mae = mae_loss(y_true, y_pred)
    loss_mae_nan = mae_nan_loss(y_true, y_pred)
    
    tf.debugging.assert_near(loss_mae, loss_mae_nan, atol=1e-6)
    print("Loss values match:", loss_mae.numpy(), loss_mae_nan.numpy())


def test_mae_nan_loss_gradients():
    """Test the gradient computation of MAENanLoss by comparing it with the Keras MeanAbsoluteError loss."""
    mae_loss = tf.keras.losses.MeanAbsoluteError()
    mae_nan_loss = MAENanLoss()
    
    y_true = tf.random.uniform((10,), minval=0, maxval=1)
    y_pred = tf.Variable(tf.random.uniform((10,), minval=0, maxval=1))
    
    with tf.GradientTape() as tape1:
        loss_mae = mae_loss(y_true, y_pred)
    grad_keras = tape1.gradient(loss_mae, y_pred)
    
    with tf.GradientTape() as tape2:
        loss_mae_nan = mae_nan_loss(y_true, y_pred)
    grad_custom = tape2.gradient(loss_mae_nan, y_pred)
    
    tf.debugging.assert_near(grad_keras, grad_custom, atol=1e-6)
    print("Gradients match:", grad_keras.numpy(), grad_custom.numpy())


def test_mae_nan_loss_gradients_complex():
    """Test the gradient computation of MAENanLoss with more complex input data by comparing it with the Keras MeanAbsoluteError loss."""
    mae_loss = tf.keras.losses.MeanAbsoluteError()
    mae_nan_loss = MAENanLoss()
    
    y_true = tf.math.sin(tf.random.uniform((10,), minval=0, maxval=1) * 2 * np.pi)
    y_pred = tf.Variable(tf.math.cos(tf.random.uniform((10,), minval=0, maxval=1) * 2 * np.pi))
    
    with tf.GradientTape() as tape1:
        loss_mae = mae_loss(y_true, y_pred)
    grad_keras = tape1.gradient(loss_mae, y_pred)
    
    with tf.GradientTape() as tape2:
        loss_mae_nan = mae_nan_loss(y_true, y_pred)
    grad_custom = tape2.gradient(loss_mae_nan, y_pred)
    
    tf.debugging.assert_near(grad_keras, grad_custom, atol=1e-6)
    print("Gradients match:", grad_keras.numpy(), grad_custom.numpy())


def test_mae_nan_loss_equivalence():
    """Test the equivalence of MAENanLoss and Keras MeanAbsoluteError loss when there are no NaN values in the target."""
    x_train = np.random.rand(100, 10)
    y_train = 2 * x_train[:, 0] + 3 * x_train[:, 1] + np.random.normal(0, 0.01, (100, 1))

    def create_model(loss):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss=loss)
        return model

    model_mae_nan_loss = create_model(MAENanLoss())
    model_mae = create_model('mae')
    
    history_mae_nan_loss = model_mae_nan_loss.fit(x_train, y_train, epochs=300, verbose=0)
    history_mae = model_mae.fit(x_train, y_train, epochs=300, verbose=0)
    
    final_loss_mae_nan_loss = history_mae_nan_loss.history['loss'][-1]
    final_loss_mae = history_mae.history['loss'][-1]
    
    assert np.isclose(final_loss_mae_nan_loss, final_loss_mae, atol=1e-3), \
        f"Losses are not equivalent: {final_loss_mae_nan_loss} vs {final_loss_mae}"


def test_mae_nan_edge_cases():
    """Test MAENanLoss with various edge cases and compare it with the Keras MeanAbsoluteError loss."""
    mae_loss = tf.keras.losses.MeanAbsoluteError()
    mae_nan_loss = MAENanLoss()
    
    edge_cases = [
        (tf.zeros(10), tf.zeros(10)),
        (tf.ones(10), tf.ones(10)),
        (tf.ones(10), tf.zeros(10)),
        (tf.ones(10) * 1e12, tf.ones(10) * 1e12),
    ]
    
    for y_true, y_pred in edge_cases:
        loss_mae = mae_loss(y_true, y_pred)
        loss_mae_nan = mae_nan_loss(y_true, y_pred)
        
        tf.debugging.assert_near(loss_mae, loss_mae_nan, atol=1e-6)
        print(f"Edge case passed: {loss_mae.numpy()} vs {loss_mae_nan.numpy()}")

def test_mae_nan_loss_various_shapes():
    """Test MAENanLoss with various shapes in the target tensor and compare it with the Keras MeanAbsoluteError loss."""
    mae_loss = tf.keras.losses.MeanAbsoluteError()
    mae_nan_loss = MAENanLoss()
    
    shapes = [
        (10,),
        (10, 1),
        (10, 10),
        (10, 10, 10),
        (100, 100, 100)
    ]
    
    for shape in shapes:
        y_true = tf.random.uniform(shape, minval=0, maxval=1)
        y_pred = tf.random.uniform(shape, minval=0, maxval=1)
        
        loss_mae = mae_loss(y_true, y_pred)
        loss_mae_nan = mae_nan_loss(y_true, y_pred)
        
        tf.debugging.assert_near(loss_mae, loss_mae_nan, atol=1e-6)
        print(f"Shape {shape} passed: {loss_mae.numpy()} vs {loss_mae_nan.numpy()}")


def test_mae_nan_handling():
    """Test the handling of NaN values in the target by MAENanLoss and compare it with the Keras MeanAbsoluteError loss."""
    mae_loss = tf.keras.losses.MeanAbsoluteError()
    mae_nan_loss = MAENanLoss()
    
    y_true = tf.constant([1.0, 2.0, np.nan, 4.0, 5.0])
    y_pred = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
    
    loss_mae = mae_loss(y_true, y_pred)
    loss_mae_nan = mae_nan_loss(y_true, y_pred)
    
    assert tf.math.is_nan(loss_mae), f"Keras MAE loss is not NaN: {loss_mae.numpy()}"
    assert tf.math.equal(loss_mae_nan, 0), f"Custom MAE loss is not 0: {loss_mae_nan.numpy()}"