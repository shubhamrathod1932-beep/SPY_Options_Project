import numpy as np
from scipy.stats import norm


def call_delta(S, K, T, r, sigma):

    d1 = (
        np.log(S / K)
        + (r + sigma**2 / 2) * T
    ) / (sigma * np.sqrt(T))

    return norm.cdf(d1)


def put_delta(S, K, T, r, sigma):

    d1 = (
        np.log(S / K)
        + (r + sigma**2 / 2) * T
    ) / (sigma * np.sqrt(T))

    return norm.cdf(d1) - 1


def gamma(S, K, T, r, sigma):

    d1 = (
        np.log(S / K)
        + (r + sigma**2 / 2) * T
    ) / (sigma * np.sqrt(T))

    return norm.pdf(d1) / (
        S * sigma * np.sqrt(T)
    )