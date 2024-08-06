This repository contains the implementation for the paper [Estimation of Rate-Distortion Function for Computing with Decoder Side Information, IEEE ISIT 2024](https://openreview.net/forum?id=xDa9Dxoww0).


## Quick Start Guide

Please follow the instructions below to quickly check the results of the codebase.

This codebase estimates the rate-distortion function with side information from data points. 

Consider a scenario featuring a 2-component White Gaussian Noise, 2-WGN $(P, \rho)$ source, where $(X, Y)$ forms pairs of i.i.d. jointly Gaussian random variables. Each pair in the sequence $$(X_1, Y_1), (X_2, Y_2), \ldots, (X_n, Y_n)$$ has zero mean ($\mathbb{E}[X] = \mathbb{E}[Y] = 0$), equal variance ($\mathbb{E}[X^2] = \mathbb{E}[Y^2] = P$), and a correlation coefficient $\rho = \mathbb{E}[XY]/P$. With a squared error distortion measure $d$, unit variance $P=1$, and a function $g(X,Y)=(X+Y)/2,$ the rate distortion for computing with decoder side information $R_{\text{D,C}}$ is given by

$$R_{\text{D,C}}(D) = \max \lbrace \frac{1}{2}\log \big( \frac{(1-\rho^2)}{4D} \big), 0\rbrace.$$

The goal of the algorithm is to estimate point(s) on the true rate distortion function $R_{\text{D,C}}(D)$.

### Generating the Scenario

You can generate a specific scenario by explicitly setting the value of $\rho$ through the dataset name. For example, to use a scenario with $\rho=0.8$ and input source dimension 10, the dataset name should be:
  
    2WGNAVG0.8rho10dim

### Running the Main Script

To run the main file with the necessary arguments, you need to set the following parameters:

- GPU number: The GPU to use for computations.
- Task name: A name to identify the task.
- Side information setup: Use 'D' to indicate that side information is available at the decoder.
- Distortion metric: The distortion measure to use.
- lmbda (-s value): The Lagrangian multiplier or the slope.
- Dataset name: The name of the dataset (e.g., 2WGNAVG0.8rho10dim).
- Neural network architecture: The architecture of the neural network.
- Seed number: The random seed.
- Batch size: The batch size for training.
- Initial learning rate: The initial learning rate for the optimizer.
- Steps per epoch: The number of steps per epoch.
- Number of epochs: The total number of training epochs.
- Results path name: The directory to store results.
- Save model: Whether to save the model (True/False).
- Evaluation interval: The interval at which to evaluate the model.
- Train/Test: Specify whether to train or test the model.

Here is an example of how to run the main script with these arguments:

    python main.py --gpu 0 \
    --task rd_estimation \
    --side_information D \
    --distortion_metric MSE \
    --lmbda 20.0 \
    --dataset 2WGNAVG0.8rho10dim \
    --model_type mlp \
    --random_seed 42 \
    --batch_size 100  \
    --learning_rate 5e-3  \
    --steps_per_epoch 100  \
    --epochs 500 \
    --results_path_name results  \
    --save True  \
    --evaluation_interval 2  \
    --train True  \
    --test True 


Replace the placeholder values with your specific parameters.


### Plotting

To visualize the results, you may use the provided Python script `data/plot/2WGNAVG_plot.py`. Before running the plotting script, you should first obtain the rate-distortion estimation results by executing `main.py` with various setups.

For example, to reproduce the plot shown in the paper:

<img src="https://github.com/Heasung-Kim/rate-distortion-side-information/blob/main/imgs/rd_plot_2wgn.png?raw=true" height="400" />

you need to execute main.py for a total of 16 different setups with ($\rho \in \lbrace 0.2, 0.4, 0.6, 0.8 \rbrace$, lmbda $\in \lbrace 5, 10, 20, 40 \rbrace$).

After running these commands and obtaining the results, you can use the plotting script to generate the desired plots.

    python data/plot/2WGNAVG_plot.py

### Citing Our Work
If you find this repository helpful, please cite our work:

    @inproceedings{kim2024estimation,
    title={Estimation of Rate-Distortion Function for Computing with Decoder Side Information},
    author={Kim, Heasung and Kim, Hyeji and De Veciana, Gustavo},
    booktitle={First 'Learn to Compress' Workshop @ ISIT 2024}
    }


### References

A part of this project is inspired by the following papers and codebases:

1. Yibo Yang and Stephan Mandt, "Towards Empirical Sandwich Bounds on the Rate-Distortion Function", ICLR 2022, https://github.com/mandt-lab/RD-sandwich

2. Lu, Zhilin, Jintao Wang, and Jian Song. "Multi-resolution CSI feedback with deep learning in massive MIMO system." ICC 2020-2020 IEEE international conference on communications (ICC). IEEE, 2020.  https://github.com/Kylin9511/CRNet

3. https://github.com/keras-team/keras-io/blob/master/examples/generative/vq_vae.py
Title: Vector-Quantized Variational Autoencoders
Author: [Sayak Paul](https://twitter.com/RisingSayak)
