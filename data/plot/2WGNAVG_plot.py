import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from data.utils import parse_last_distortion_rate
import os
from global_config import ROOT_DIRECTORY


def rate_distortion_decoder_sideinfo_2wgnavg(D, P, rho):
    return np.clip(0.5 * np.log((1 - rho ** 2) * P / (4 * D)), 0, a_max=1e9)


rhos = [0.2, 0.4, 0.6, 0.8]
s_list = [5.0, 10.0, 20.0, 40.0]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7, 7))

markers = ['s', 'd', 'o', '*']
for idx, rho in enumerate(rhos):
    ax = axes[idx // 2, idx % 2]
    base_path = os.path.join(ROOT_DIRECTORY, "data/results/rd_estimation_D_MSE_2WGNAVG" + str(rho) + "rho10dim_mlp_")

    # Define the range of x values
    D = np.linspace(0.01, 0.3, 100)

    # Define values for P and N
    P = 1

    # Calculate y values for the curve
    y_rd_si_ed = rate_distortion_decoder_sideinfo_2wgnavg(D, P=P, rho=rho)
    y_rd = rate_distortion_decoder_sideinfo_2wgnavg(D, P=P, rho=0)

    # plt.figure(figsize=(4, 4))
    ax.plot(D, y_rd_si_ed, label='$R_{D,C}(D)$', color='b', marker=3, markevery=10)
    ax.plot(D, y_rd, label='$R_{D,C}(D), \\rho=0$', color='r')  # special case for rho=0 for comparison

    distortion_points = np.zeros(shape=len(s_list))
    estimated_rate_points = np.zeros(shape=len(s_list))
    ground_truth_rate_points = np.zeros(shape=len(s_list))

    for s_idx, s in enumerate(s_list):
        with open(base_path + str(s) + "/trainHistory", 'rb') as handle:  # loading old history
            history = pickle.load(handle)
        ax.plot(history['distortion'], history['rate'], linestyle='--', marker=markers[s_idx],
                markevery=len(history['distortion']) - 1, markersize=7, markerfacecolor='white',
                label='$\hat{R}_{D,C}(D)$ (s=-' + str(s) + ')', color='k')

        test_log = os.path.join(base_path + str(s), "log.txt")
        distortion, rate = parse_last_distortion_rate(test_log)

        distortion_points[s_idx] = history['distortion'][-1]  # distortion #
        estimated_rate_points[s_idx] = history['rate'][-1]  # rate #
        ground_truth_rate_points[s_idx] = rate_distortion_decoder_sideinfo_2wgnavg(D=history['distortion'][-1], P=P, rho=rho)

    ax.axis(xmin=0, xmax=0.25)
    ax.axis(ymin=0, ymax=1.5)
    ax.set_title('$\\rho={}$'.format(rho))
    ax.grid()

fig.supxlabel('Distortion')
fig.supylabel('Rate')

plt.legend()
plt.tight_layout()

dir_path = os.path.dirname(os.path.realpath(__file__))
plt.savefig(os.path.join(dir_path, "wgn_computing_rd.pdf"))
plt.show()
