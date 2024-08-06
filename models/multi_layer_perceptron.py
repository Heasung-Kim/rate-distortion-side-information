import tensorflow as tf


def multi_layer_percepton(input_shape, output_shape, units, activations, double_input, name='mlp', dtype=None):
    assert len(units) == len(activations)

    if double_input is True:
        x1 = tf.keras.layers.Input(shape=input_shape[0])
        input1 = tf.keras.layers.Flatten()(x1)
        x2 = tf.keras.layers.Input(shape=input_shape[1])
        input2 = tf.keras.layers.Flatten()(x2)
        output = tf.keras.layers.Concatenate(axis=-1)([input1, input2])
        for unit, activation in zip(units, activations):
            output = tf.keras.layers.Dense(units=unit, activation=activation, use_bias=True)(output)
        model = tf.keras.models.Model(inputs=[x1, x2], outputs=output)

    else:
        input = tf.keras.layers.Input(shape=input_shape)
        output = tf.keras.layers.Flatten()(input)
        for unit, activation in zip(units, activations):
            output = tf.keras.layers.Dense(units=unit, activation=activation, use_bias=True)(output)
        model = tf.keras.models.Model(inputs=input, outputs=output)

    return model


