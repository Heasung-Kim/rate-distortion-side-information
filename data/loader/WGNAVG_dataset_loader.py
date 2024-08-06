import tensorflow as tf
import tensorflow_probability as tfp
import re

def generate_2wgnavg_tf(P, rho, data_dim, batch_size, dtype):
    # Mean vector [E(X), E(Y)] - both are 0
    mean = [0.0, 0.0]
    covariance = [[P, rho * P],
                  [rho * P, P]]

    # Multivariate Normal Distribution
    mvn = tfp.distributions.MultivariateNormalFullCovariance(
        loc=mean,
        covariance_matrix=covariance
    )

    def map_sample_fun(non_used_input):
        # ... the raw_feature is preprocessed as per the use-case
        # Generate indices based on probabilities
        data = []
        for i in range(batch_size):
            # Generate samples
            samples = mvn.sample(data_dim)
            average = tf.reduce_mean(samples, axis=1, keepdims=True)
            #zeros = tf.zeros_like(average)
            expanded_tensor = tf.concat([samples, average], axis=1)
            #samples = tf.concat([expanded_tensor[:, 0:1], expanded_tensor[:, 2:], expanded_tensor[:, 1:2]], axis=1)
            samples = tf.concat([expanded_tensor[:, 0:1], expanded_tensor[:, 2:], expanded_tensor[:, 1:2]], axis=1)

            data.append(samples)
        batch = tf.cast(data, dtype=dtype)
        return batch
    #test = map_sample_fun(2)
    dataset = tf.data.Dataset.from_tensors([])
    dataset = dataset.repeat()
    dataset = dataset.map(map_sample_fun)

    return dataset


def get_2WGNAVG_dataset(config):
    """

    :param config:
    :return:
    """
    batch_size = config["batch_size"]
    dtype = config["dtype"]

    P = 1

    matches = re.search(r'(\d+\.\d+).*?(\d+)', config["dataset"])
    if matches:
        rho = float(matches.group(1))
        data_dim = int(matches.group(2))
    else:
        raise NotImplementedError

    train_dataset = generate_2wgnavg_tf(P=P, rho=rho, data_dim=data_dim, batch_size=batch_size, dtype=dtype)
    validation_dataset = generate_2wgnavg_tf(P=P, rho=rho, data_dim=data_dim, batch_size=batch_size, dtype=dtype)
    validation_dataset = validation_dataset.take(10)  # keras crashes without this (would be using an infinite validation set)

    test_dataset = generate_2wgnavg_tf(P=P, rho=rho, data_dim=data_dim, batch_size=batch_size, dtype=dtype)
    test_dataset = test_dataset.take(100)  # keras crashes without this (would be using an infinite validation set)

    dataset_variance = 1.0
    config["input_shape"] = (data_dim, 2)  # data dimension and source/side info
    config["data_shape"] = (data_dim)
    config["flatten_data_shape"] = data_dim * 1

    return train_dataset, validation_dataset, test_dataset, dataset_variance, None
