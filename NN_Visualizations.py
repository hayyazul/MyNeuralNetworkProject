import random

from scipy.ndimage import gaussian_filter1d
import numpy as np
import matplotlib.pyplot as plt

def plot_smoothed(values, sigma=1.0, offset=0, show=False):
    """
    Literally copy and pasted from a chatbot
    :param values:
    :param sigma:
    :param offset: Offset of graph in the y-axis
    :param show: Whether to show the graph
    :return:
    """

    smoothed = gaussian_filter1d(np.array(values) + offset, sigma)
    plt.plot(smoothed)

    if show:
        plt.show()


random.seed(0)
OFFSET = 10

if __name__ == "__main__":

    high = 100
    a = [random.random() * high for _ in range(10)]
    for i in range(10):
        high *= 0.9
        a.extend([random.random() * high for _ in range(10)])

    for i in range(10):
        plot_smoothed(np.array(a), i / 10 + 1, offset=OFFSET * i)

    plt.show()