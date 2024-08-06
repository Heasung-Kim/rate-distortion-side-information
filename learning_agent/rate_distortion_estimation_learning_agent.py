import math
import tensorflow as tf
import tensorflow_probability as tfp
from models.utils import preprocessing, postprocessing, get_encoder_model, get_decoder_model, get_prior_model
import numpy as np


class RateDistortionEstimationLearningAgent(tf.keras.Model):
    """


        Case 1: If side information is not available,
            posterior (encoder) = q_U|X
            prior = q_U

        Case 2: If side information is available only at decoder,
            posterior (encoder) = q_U|X
            prior = q_U|Y

        Case 3: If side information is available at encoder and decoder,
            posterior (encoder) = q_U|X,Y
            prior = q_U|Y
    """
    def __init__(self,
                 side_information,
                 distortion_metric,
                 lmbda,
                 data_shape,
                 model_type,
                 dataset,
                 task,
                 data_generator=None,
                 use_real_time_data=False):
        super().__init__()
        self.side_information = side_information
        self.distortion_metric = distortion_metric
        self.lmbda = lmbda
        self.data_shape = data_shape
        self.model_type = model_type
        self.dataset = dataset
        self.task = task
        self.data_generator = data_generator
        self.USE_REAL_TIME_DATA = use_real_time_data

        self.flatten_data_shape = self.data_shape if isinstance(self.data_shape, int) else math.prod(self.data_shape)
        self.latent_dim = int(self.flatten_data_shape)

        if self.side_information == "none" or self.side_information == "D":
            self.posterior_input_shape = self.data_shape
        if self.side_information == "ED":
            self.posterior_input_shape = [self.data_shape, self.data_shape]

        self.prior_encoder_output_dim = self.latent_dim * 2
        self.posterior_output_dim = self.latent_dim * 2

        if self.side_information == "none":
            self.decoder_input_shape = [self.latent_dim]
        if self.side_information == "ED" or self.side_information == "D":
            self.decoder_input_shape = [self.latent_dim, self.data_shape]   # latent and codeword

        self.decoder_output_shape = self.data_shape

        self.posterior = get_encoder_model(input_shape=self.posterior_input_shape,
                                           output_shape=self.posterior_output_dim,
                                           model_type=self.model_type,
                                           side_information=self.side_information,
                                           task=self.task)

        self.decoder = get_decoder_model(input_shape=self.decoder_input_shape,
                                         output_shape=self.decoder_output_shape,
                                         model_type=self.model_type,
                                         side_information=self.side_information,
                                         task=self.task)

        if self.side_information == "ED" or self.side_information == "D":
            self.prior = get_prior_model(input_shape=self.data_shape,
                                         output_shape=self.prior_encoder_output_dim,
                                         model_type=self.model_type,
                                         side_information=self.side_information)
        elif self.side_information == "none":
            from models.utils import PriorWeights
            self.prior = PriorWeights(latent_dim=self.latent_dim)

    def forward_propagation(self, x):
        """Given a batch of inputs, perform a full inference -> generative pass through the model."""

        const = np.log(np.expm1(1.))
        input_source, target_output, side_info = tf.split(x, num_or_size_splits=3, axis=-1)
        input_source = tf.squeeze(input_source, axis=-1)
        side_info = tf.squeeze(side_info, axis=-1)

        input_source = preprocessing(dataset_type=self.dataset)(input_source)
        side_info = preprocessing(dataset_type=self.dataset)(side_info)

        if self.side_information == "ED":
            encoder_input = [input_source, side_info]
        else:
            encoder_input = input_source

        encoder_res = self.posterior(encoder_input)
        posterior_mean = encoder_res[..., :self.latent_dim]
        posterior_var = tf.nn.softplus(encoder_res[..., self.latent_dim:] + const)
        posterior_distribution = tfp.distributions.MultivariateNormalDiag(loc=posterior_mean, scale_diag=posterior_var, name="posterior")
        codeword_sample = posterior_distribution.sample()

        posterior_log_probability = posterior_distribution.log_prob(codeword_sample)  
        
        if self.side_information == "ED" or self.side_information == "D":
            prior_mean_var = self.prior(side_info)
        elif self.side_information == "none":
            prior_mean_var = self.prior(input_source)

        prior_mean = prior_mean_var[..., :self.latent_dim]
        prior_var = tf.nn.softplus(prior_mean_var[...,  self.latent_dim:2 * self.latent_dim] + const)
        prior_distribution = tfp.distributions.MultivariateNormalDiag(loc=prior_mean, scale_diag=prior_var, name="prior")
        prior_log_probability = prior_distribution.log_prob(codeword_sample)

        kl_divergence = posterior_log_probability - prior_log_probability   # Rate term

        codeword_sample = tf.cast(codeword_sample, tf.float32)
        if self.side_information == "ED" or self.side_information == "D":
            decoder_input= [codeword_sample, side_info]
        else:
            decoder_input = codeword_sample
        decoder_output = self.decoder(decoder_input)
        decoder_output = postprocessing(dataset_type=self.dataset)(decoder_output)
        return posterior_distribution, decoder_output, kl_divergence
    
    def compute_loss(self, x):
        _, pred, rates = self.forward_propagation(x)
        input_source, target_output, side_info = tf.split(x, num_or_size_splits=3, axis=-1)
        input_source = tf.squeeze(input_source, axis=-1)
        target_output = tf.squeeze(target_output, axis=-1)

        if self.distortion_metric == "MSE":
            distortion = tf.reduce_mean(tf.math.square(tf.abs(target_output-pred)))
        elif self.distortion_metric == "NMSE":
            distortion = tf.reduce_mean(tf.math.square(tf.abs(target_output-pred)) / tf.reduce_mean(tf.math.square(tf.abs(target_output))))
        elif self.distortion_metric == "hamming_distance":
            distortion = tf.reduce_mean(target_output*(1-pred)+(1-target_output)*(pred)) # Hamming Loss
        elif self.distortion_metric == "ABS":
            distortion = tf.keras.losses.MeanAbsoluteError()(target_output, pred)
        elif self.distortion_metric == "BER":
            distortion = self.ofdm_e2e_ber.get_BCE_distortion(ground_truth_channel=target_output, estimated_channel=pred)  # BCE
        else:
            raise NotImplementedError
        
        rate = tf.reduce_mean(rates) / float(self.flatten_data_shape)
        loss = rate + self.lmbda * distortion

        codeword_sample_binary = tf.cast(tf.greater(pred, 0.5), tf.int32)
        hamming_loss = tf.reduce_mean(tf.abs(input_source - tf.cast(codeword_sample_binary,tf.float32)))

        return loss, rate, distortion, hamming_loss, pred

    def call(self, x):
        if self.USE_REAL_TIME_DATA:
            x = self.data_generator.get_batch()
        return self.compute_loss(x)

    def train_step(self, x):
        if self.USE_REAL_TIME_DATA:
            x = self.data_generator.get_batch()

        with tf.GradientTape(persistent=False) as tape:
            loss, rate, distortion, hamming_loss, pred = self.compute_loss(x)
        """
            if self.side_information == "D":
                self.prior.trainable = False
                self.posterior.trainable = True
                self.decoder.trainable = True
        """
        trainable_vars = [var for var in self.trainable_variables if var.trainable]
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        """
        if self.side_information == "D":
            with tf.GradientTape(persistent=False) as tape:
               loss, rate, distortion, hamming_loss, pred = self.compute_loss(x)
            self.prior.trainable = True
            self.posterior.trainable = False
            self.decoder.trainable = False
            trainable_vars = [var for var in self.trainable_variables if var.trainable]
            gradients = tape.gradient(rate, trainable_vars)
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        """

        for metric, value in zip([self.loss, self.rate, self.distortion, self.hamming_loss, self.lr],
                                 [loss, rate, distortion, hamming_loss, self.optimizer._decayed_lr('float32')]):
            metric.update_state(value)

        return {m.name: m.result() for m in [self.loss, self.rate, self.distortion,  self.hamming_loss, self.lr]}

    def predict_step(self, x):
        if self.USE_REAL_TIME_DATA:
            x = self.data_generator.get_batch()
        loss, rate, distortion, hamming_loss, pred = self.compute_loss(x)
        input_source, target_output, side_info = tf.split(x, num_or_size_splits=3, axis=-1)
        input_source = tf.squeeze(input_source, axis=-1)
        target_output = tf.squeeze(target_output, axis=-1)
        side_info = tf.squeeze(side_info, axis=-1)
        return input_source, target_output, side_info, pred, rate, distortion

    def test_step(self, x):
        if self.USE_REAL_TIME_DATA:
            x = self.data_generator.get_batch()
        loss, rate, distortion, hamming_loss, pred = self(x, training=False)
        for metric, value in zip([self.loss, self.rate, self.distortion, self.hamming_loss],
                                 [loss, rate, distortion, hamming_loss]):
            metric.update_state(value)
        return {m.name: m.result() for m in [self.loss, self.rate, self.distortion, self.hamming_loss]}

    def compile(self, **kwargs):
        super().compile(loss=None, metrics=None, loss_weights=None, weighted_metrics=None, **kwargs)
        metric_names = ["loss", "rate", "distortion", "hamming_loss", "lr"]
        self.metrics_dict = {name: tf.keras.metrics.Mean(name=name) for name in metric_names}
        for name, metric in self.metrics_dict.items():
            setattr(self, name, metric)

