import random

from scipy.ndimage import gaussian_filter1d
import numpy as np
import matplotlib.pyplot as plt

def plot_smoothed(values, sigma=1.0):
    """
    Literally copy and pasted from a chatbot
    :param values:
    :param sigma:
    :return:
    """
    smoothed = gaussian_filter1d(values, sigma)
    plt.plot(smoothed)


random.seed(0)
OFFSET = 10

if __name__ == "__main__":

    high = 100
    a = [random.random() * high for _ in range(10)]
    for i in range(10):
        high *= 0.9
        a.extend([random.random() * high for _ in range(10)])

    for i in range(10):
        plot_smoothed(a, i / 10 + 1)
        a = [x + OFFSET for x in a]

    plt.show()