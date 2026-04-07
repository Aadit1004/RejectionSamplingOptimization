import numpy as np

class WhitenedGaussianSampler2D:
    def __init__(self, mu, cov, scale=1.5):
        self.mu = np.asarray(mu)
        self.cov = np.asarray(cov)
        self.scale = scale

        self.L = np.linalg.cholesky(self.cov)
        self.M = None

    # transforms samples from original x-space into whitened y-space
    # removes correlation and makes the target more isotropic
    def whiten(self, X):
        return np.linalg.solve(self.L, (X - self.mu).T).T

    # maps samples from whitened y-space back to original x-space
    # should be used after rejection sampling to recover samples in the original target coordinates
    def unwhiten(self, Y):
        return (self.L @ Y.T).T + self.mu

    # estimate the whitened target density in y-space
    # after whitening, the correlated Gaussian becomes standard 2D normal
    def target_pdf_y(self, Y):
        squared_norm = np.sum(Y**2, axis=1)
        return (1.0 / (2 * np.pi)) * np.exp(-0.5 * squared_norm)

    # estimate the gaussian proposal density used for rejection sampling in y-space
    # the proposal is isotropic with variance done by self.scale
    def g_pdf(self, Y):
        d = 2
        sigma = self.scale
        squared_norm = np.sum(Y**2, axis=1)
        norm_const = 1.0 / (((2 * np.pi) ** (d / 2)) * (sigma ** d))
        return norm_const * np.exp(-0.5 * squared_norm / (sigma ** 2))

    # gets samples in whitened space from an isotropic gaussian
    def sample_proposal(self, n_samples):
        return np.random.normal(loc=0.0, scale=self.scale, size=(n_samples, 2))

    def find_M(self, Y_grid):
        ratios = self.target_pdf_y(Y_grid) / self.g_pdf(Y_grid)
        self.M = 1.05 * np.max(ratios)
        return self.M

    # runs vectorized rejection sampling in whitened space,
    # accepted samples are returned in both y-space and mapped back x-space
    def vectorized_rejection_sample(self, n_samples):
        Y = self.sample_proposal(n_samples)
        u = np.random.rand(n_samples)

        accept_prob = self.target_pdf_y(Y) / (self.M * self.g_pdf(Y))
        accepted_Y = Y[u <= accept_prob]
        accepted_X = self.unwhiten(accepted_Y)

        acceptance_count = len(accepted_Y)
        acceptance_rate = acceptance_count / n_samples

        return accepted_Y, accepted_X, acceptance_count, acceptance_rate