import tensorflow as tf
from keras import layers

def preprocess(x, remaining_delay=4):
    #encoder_inputs = keras.Input(shape=config['batch_shape'], dtype=tf.complex64)
    x = tf.squeeze(x, axis=[1,3])
    x = x[:,:,:,-1,:] # last ofdm symbol
    x = tf.signal.ifft2d(x) # inner-most dimension Inverse FFT
    x = x[:,:,:,:remaining_delay]
    x = tf.stack([tf.math.real(x), tf.math.imag(x)], axis=-1)
    #tf.print(x.shape)
    return x #keras.Model(encoder_inputs, x, name="preprocessor")

def postprocess(x, fft_size=72, remaining_delay=4):
    x_shape = x.shape.as_list()
    x = tf.complex(x[:,:,:,:,0], x[:,:,:,:,1])
    x = tf.concat([x, tf.zeros(shape=(200,x_shape[1],x_shape[2], fft_size-remaining_delay), dtype=tf.complex64)], axis=-1)
    x = tf.signal.fft2d(x)
    x = tf.expand_dims(x, 1)
    x = tf.expand_dims(x, 3)
    x = tf.expand_dims(x, 5)
    x = tf.repeat(x, repeats=14, axis=-2)
    return x


def single_symbol_to_multi_symbols(x):
    n_symbols = 1
    x_shape = x.shape.as_list()
    x = tf.expand_dims(x, 1)
    x = tf.expand_dims(x, 3)
    x = tf.expand_dims(x, 5)
    x = tf.repeat(x, repeats=n_symbols, axis=-2)
    return x


def preprocess_SF_to_cropped_AD(x):
    # encoder_inputs = keras.Input(shape=config['batch_shape'], dtype=tf.complex64)
    x = tf.signal.ifft2d(x)  # inner-most dimension Inverse FFT
    x = x[:, :, :, :32]
    x = tf.stack([tf.math.real(x), tf.math.imag(x)], axis=-1)
    x = tf.squeeze(x)
    # tf.print(x.shape)

    return x




def postprocess_cropped_AD_to_SF(x):
    x_shape = x.shape.as_list()
    x = tf.complex(x[:,:,:,0], x[:,:,:,1]) # Channel First. (Batch size, channel, Tx antenna, angular)
    x = tf.concat([x, tf.zeros(shape=(100,x_shape[1], 667-32), dtype=tf.complex64)], axis=-1)
    x = tf.signal.fft(x)
    x = tf.expand_dims(x, 1)
    #x = tf.expand_dims(x, 3)
    #x = tf.expand_dims(x, 5)
    #x = tf.repeat(x, repeats=14, axis=-2)
    #x = tf.math.tanh(x)
    return x

class PriorWeights(tf.keras.Model):
    def __init__(self, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim

    def build(self, inputs):
        self.loc = tf.zeros((self.latent_dim, 1), dtype=self.dtype)
        self.log_scale = tf.ones((self.latent_dim, 1), dtype=self.dtype)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        log_scale_batch = tf.tile(self.log_scale, [1, batch_size])
        loc_batch = tf.tile(self.loc, [1, batch_size])

        log_scale_batch = tf.reshape(log_scale_batch, [batch_size, self.latent_dim])
        loc_batch = tf.reshape(loc_batch, [batch_size, self.latent_dim])
        combined = tf.concat([loc_batch, log_scale_batch], axis=1)

        return combined


def get_encoder_model(input_shape, output_shape, model_type, side_information, task):
    assert side_information =="none" or side_information == "D" or side_information == "ED"

    #if task == "rd_estimation":
    if model_type == "mlp":
        from models.multi_layer_perceptron import multi_layer_percepton
        model = multi_layer_percepton(input_shape, output_shape, units=[output_shape], activations=["linear"],
                                      double_input=True if side_information=="ED" else False, name='mlp', dtype=None)
    else:
        raise NotImplementedError

    return model

def get_decoder_model(input_shape, output_shape, model_type, side_information, task):
    assert side_information =="none" or side_information == "D" or side_information == "ED"
    #if task == "rd_estimation":
    if model_type == "mlp":
        from models.multi_layer_perceptron import multi_layer_percepton
        #model = multi_layer_percepton(input_shape, output_shape, units=[output_shape, output_shape], activations=["leaky_relu", "linear"],
        model = multi_layer_percepton(input_shape, output_shape, units=[output_shape, output_shape], activations=["leaky_relu","linear"],
                                      double_input=False if side_information == "none" else True, name='mlp', dtype=None)
    else:
        raise NotImplementedError

    return model

def get_prior_model(input_shape, output_shape, model_type, side_information):
    assert side_information =="none" or side_information == "D" or side_information == "ED"
    if model_type == "mlp":
        from models.multi_layer_perceptron import multi_layer_percepton
        model = multi_layer_percepton(input_shape, output_shape, units=[output_shape],
                                      activations=["linear"],
                                      double_input=False,
                                      dtype=None)
    else:
        raise NotImplementedError

    return model



def preprocessing(dataset_type):
    if dataset_type == "CSI_SF":
        return preprocess_SF_to_cropped_AD
    elif dataset_type == "CSI_AD":
        return lambda x: x

    else:
        return lambda x: x

def postprocessing(dataset_type):
    if dataset_type == "CSI_SF":
        return postprocess_cropped_AD_to_SF
    elif dataset_type == "CSI_AD":
        return lambda x: x
    else:
        return lambda x: x


if __name__ == "__main__":
    x = tf.random.uniform(shape=(200,2,8,4,2))

    y= postprocess(x)

    print(y.shape)
