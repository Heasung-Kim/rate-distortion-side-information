import argparse
from global_config import base_config, global_logger, ROOT_DIRECTORY, logging
from pathlib import Path
import numpy as np
import pickle
from utils import get_simulation_name
import os

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=3)
# Task and Dataset
parser.add_argument("--task",  type=str, default='rd_estimation', choices=['rd_estimation'])
parser.add_argument("--side_information", type=str, default="D", choices=["ED", "D", "none"])
parser.add_argument("--distortion_metric", type=str, default="MSE", choices=["MSE", "NMSE", "BER", "hamming_distance", "ABS"])
parser.add_argument("--lmbda", type=float, default=20., help="slope (Lagrange,s)")
parser.add_argument("--dataset",  type=str, default='2WGNAVG0.4rho10dim')
parser.add_argument("--USE_REAL_TIME_DATA",  type=bool, default=False)
parser.add_argument("--model_type",  type=str, default='mlp', choices=['mlp'])  # neural network selection
parser.add_argument("--random_seed", type=int, default=42)

# For fixed-rate compression
parser.add_argument("--n_codeword_bits", type=int, default=64)

# Training configuration
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--learning_rate", type=float, default=5e-3)
parser.add_argument("--steps_per_epoch", type=int, default=100)
parser.add_argument("--epochs", type=int, default=500)

# Results
parser.add_argument("--results_path_name", type=str, default="results")
parser.add_argument("--save", type=bool, default=True)
parser.add_argument("--evaluation_interval", type=int, default=2)

#
parser.add_argument("--load", type=bool, default=False)
parser.add_argument("--train", type=bool, default=True)
parser.add_argument("--test", type=bool, default=True)

args = parser.parse_args()

gpu_num = int(args.gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"

config = base_config
config.update(vars(args))
get_simulation_name(config)

import random
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
from tensorflow import keras

random_seed = config["random_seed"]
# tf.config.run_functions_eagerly(True)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['PYTHONHASHSEED'] = str(random_seed)
random.seed(random_seed)
tf.random.set_seed(random_seed)
np.random.seed(random_seed)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

results_path = os.path.join(ROOT_DIRECTORY, "data", config["results_path_name"], config["name"])
Path(results_path).mkdir(parents=True, exist_ok=True)
global_logger.addHandler(logging.FileHandler(os.path.join(results_path, "log.txt"), 'a'))

# Dataset
from data.utils import get_dataset
train_dataset, validation_dataset, test_dataset, data_variance, data_generator = get_dataset(config=config)

# Model
checkpoint_path = os.path.join(results_path,"cp.ckpt")
checkpoint_dir = os.path.dirname(checkpoint_path)

best_val_model_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                             save_weights_only=True, save_best_only=True,
                                                             monitor="val_loss" if config["task"] == "rd_estimation"
                                                             else "val_distortion",
                                                             mode='min', verbose=1)

if config["task"] == "rd_estimation":
    from learning_agent.rate_distortion_estimation_learning_agent import RateDistortionEstimationLearningAgent
    learning_agent = RateDistortionEstimationLearningAgent(side_information=config["side_information"],
                                                           distortion_metric=config["distortion_metric"],
                                                           lmbda=config["lmbda"],
                                                           data_shape=config["data_shape"],
                                                           model_type=config["model_type"],
                                                           dataset=config["dataset"],
                                                           task=config["task"],
                                                           data_generator=data_generator)
else:
    raise NotImplementedError

if config["load"] is True:
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    global_logger.info("ckpt dir:" + checkpoint_dir)
    learning_agent.load_weights(latest)

if config["train"] is True:
    lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=config["learning_rate"],
        decay_steps=config["epochs"] * config["steps_per_epoch"],
        alpha=0.2)
    learning_agent.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=lr_scheduler))
    learning_history = learning_agent.fit(
        train_dataset,
        epochs=config["epochs"],
        steps_per_epoch=config["steps_per_epoch"],
        validation_data=validation_dataset.take(1000),
        batch_size=config["batch_size"],
        verbose=2,
        callbacks=[best_val_model_callback]
    )
    with open(os.path.join(results_path,'trainHistory'), 'wb') as handle:
        pickle.dump(learning_history.history, handle)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    global_logger.info("ckpt dir:" + checkpoint_dir)
    learning_agent.load_weights(latest)

if config["test"] is True:
    input_source, label, side_info, decoder_output, rate, distortion = learning_agent.predict(test_dataset)
    global_logger.info("(distortion,rate): ({},{})".format(np.mean(distortion), np.mean(rate)))
