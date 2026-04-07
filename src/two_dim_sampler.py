import numpy as np

class GaussianSampler2D:
    def __init__(self, mu, scale=2.5):
        self.mu = np.asarray(mu)
        self.scale = scale
        self.M = None

    def g_pdf(self, X):
        diff = X - self.mu
        squared_norm = np.sum(diff**2, axis=1)
        norm_const = 1.0 / (2 * np.pi * self.scale**2)
        return norm_const * np.exp(-0.5 * squared_norm / (self.scale**2))

    def sample_proposal(self, n_samples):
        return np.random.normal(loc=self.mu, scale=self.scale, size=(n_samples, 2))

    def find_M(self, X_grid, target_pdf):
        ratios = target_pdf(X_grid) / self.g_pdf(X_grid)
        self.M = 1.05 * np.max(ratios)
        return self.M

    def vectorized_rejection_sample(self, n_samples, target_pdf):
        X = self.sample_proposal(n_samples)
        u = np.random.rand(n_samples)

        accept_prob = target_pdf(X) / (self.M * self.g_pdf(X))
        accepted_X = X[u <= accept_prob]

        acceptance_count = len(accepted_X)
        acceptance_rate = acceptance_count / n_samples
        return accepted_X, acceptance_count, acceptance_rate