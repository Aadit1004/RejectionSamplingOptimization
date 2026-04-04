import numpy as np

# default is standard normal
def gaussian_pdf(x, mu=0.0, sigma=1.0):
    return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )

def bimodal_pdf(x):
    return (
        0.4 * gaussian_pdf(x, mu=-2.0, sigma=0.7) +
        0.6 * gaussian_pdf(x, mu=2.0, sigma=1.0)
    )

def complex_multimodal_pdf(x):
    return (
        0.12 * gaussian_pdf(x, mu=-4.7, sigma=0.35) +
        0.18 * gaussian_pdf(x, mu=-3.1, sigma=0.90) +
        0.26 * gaussian_pdf(x, mu=-0.4, sigma=0.50) +
        0.31 * gaussian_pdf(x, mu=1.1, sigma=0.65) +
        0.13 * gaussian_pdf(x, mu=3.9, sigma=0.40)
    )

def gaussian_pdf_cupy(x, mu=0.0, sigma=1.0):
    import cupy as cp
    return (1.0 / (cp.sqrt(2 * cp.pi) * sigma)) * cp.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )

def bimodal_pdf_cupy(x):
    import cupy as cp
    return (
        0.4 * gaussian_pdf_cupy(x, mu=-2.0, sigma=0.7) +
        0.6 * gaussian_pdf_cupy(x, mu=2.0, sigma=1.0)
    )

def complex_multimodal_pdf_cupy(x):
    import cupy as cp
    return (
        0.12 * gaussian_pdf_cupy(x, mu=-4.7, sigma=0.35) +
        0.18 * gaussian_pdf_cupy(x, mu=-3.1, sigma=0.90) +
        0.26 * gaussian_pdf_cupy(x, mu=-0.4, sigma=0.50) +
        0.31 * gaussian_pdf_cupy(x, mu=1.1, sigma=0.65) +
        0.13 * gaussian_pdf_cupy(x, mu=3.9, sigma=0.40)
    )

def correlated_gaussian_pdf(X, mu, cov): # multivariate gaussian
    d = len(mu)
    cov_inv = np.linalg.inv(cov)
    cov_det = np.linalg.det(cov)

    diff = X - mu
    exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)

    norm_const = 1.0 / np.sqrt(((2 * np.pi) ** d) * cov_det)
    return norm_const * np.exp(exponent)

def whiten_samples(X, mu, cov):
    L = np.linalg.cholesky(cov)
    return np.linalg.solve(L, (X - mu).T).T