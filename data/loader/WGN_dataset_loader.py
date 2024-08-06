import tensorflow as tf
import tensorflow_probability as tfp
import re



def generate_2wgn_tf(P, N, data_dim, batch_size, dtype):
    # Mean vector [E(X), E(Y)] - both are 0
    mean = [0.0, 0.0]
    covariance = [[P,  P],
                  [P, P+N]]

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
            samples = tf.concat([samples[:, 0:1], samples[:, 0:1], samples[:, 1:2]], axis=1)

            data.append(samples)
        batch = tf.cast(data, dtype=dtype)
        return batch
    # test = map_sample_fun(2)
    dataset = tf.data.Dataset.from_tensors([])
    dataset = dataset.repeat()
    dataset = dataset.map(map_sample_fun)

    return dataset


def get_2WGN_dataset(config):
    """

    :param config:
    :return:
    """
    batch_size = config["batch_size"]
    dtype = config["dtype"]

    pattern = r'P(\d+\.\d+)|N(\d+\.\d+)|(\d+)dim'
    matches = re.findall(pattern, config["dataset"])
    values = {}
    for p, n, dim in matches:
        if p:
            P = float(p)
        if n:
            N = float(n)
        if dim:
            data_dim = int(dim)

    train_dataset = generate_2wgn_tf(P=P, N=N, data_dim=data_dim, batch_size=batch_size, dtype=dtype)
    validation_dataset = generate_2wgn_tf(P=P, N=N, data_dim=data_dim, batch_size=batch_size, dtype=dtype)
    validation_dataset = validation_dataset.take(10)  # keras crashes without this (would be using an infinite validation set)

    test_dataset = generate_2wgn_tf(P=P, N=N, data_dim=data_dim, batch_size=batch_size, dtype=dtype)
    test_dataset = test_dataset.take(100)  # keras crashes without this (would be using an infinite validation set)

    dataset_variance = 1.0
    config["input_shape"] = (data_dim, 2)  # data dimension and source/side info
    config["data_shape"] = (data_dim)
    config["flatten_data_shape"] = data_dim * 1

    return train_dataset, validation_dataset, test_dataset, dataset_variance, None
