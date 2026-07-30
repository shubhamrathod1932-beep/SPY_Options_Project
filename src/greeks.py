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

def vega(S, K, T, r, sigma):

    d1 = (
        np.log(S / K)
        + (r + sigma**2 / 2) * T
    ) / (sigma * np.sqrt(T))

    return (
        S
        * norm.pdf(d1)
        * np.sqrt(T)
    )

def call_theta(S, K, T, r, sigma):

    d1 = (
        np.log(S / K)
        + (r + sigma**2 / 2) * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    theta = (
        -(S * norm.pdf(d1) * sigma)
        / (2 * np.sqrt(T))
        - r * K * np.exp(-r * T) * norm.cdf(d2)
    )

    # Return per-day Theta
    return theta / 365

def put_theta(S, K, T, r, sigma):

    d1 = (
        np.log(S / K)
        + (r + sigma**2 / 2) * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    theta = (
        -(S * norm.pdf(d1) * sigma)
        / (2 * np.sqrt(T))
        + r * K * np.exp(-r * T) * norm.cdf(-d2)
    )

    # Return per-day Theta
    return theta / 365
