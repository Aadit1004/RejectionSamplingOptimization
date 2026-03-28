import numpy as np

# single gaussian as baseline
class BaselineSampler:
    def __init__(self, f_mu, f_sigma, scale=1.5):
        self.f_mu = f_mu 
        self.f_sigma = f_sigma
        self.scale = scale

        self.g_mu = None
        self.g_sigma = None
        self.M = None

    def g_pdf(self, x): 
        # gaussian pdf
        return (1.0 / (np.sqrt(2 * np.pi) * self.g_sigma)) * np.exp(
            -0.5 * ((x - self.g_mu) / self.g_sigma) ** 2
        )

    def set_proposal(self):
        # generating gaussian proposal
        self.g_mu = self.f_
        self.g_sigma = self.f_sigma * self.scale
        return self.g_mu, self.g_sigma

    def find_M(self, f_pdf, x_grid):
        ratios = f_pdf(x_grid) / self.g_pdf(x_grid)
        safety_scale = 1.05
        self.M = safety_scale * np.max(ratios)
        return self.M

    def rejection_sample(self, n_samples, f_pdf):
        # f_pdf is passed in as a function
        accepted_samples = []
        for i in range(n_samples):
            x = np.random.normal(loc=self.g_mu, scale=self.g_sigma)
            u = np.random.rand()

            accept_prob = f_pdf(x) / (self.M * self.g_pdf(x))
            if u <= accept_prob:
                accepted_samples.append(x);

        acceptance_count = len(accepted_samples)
        acceptance_rate = acceptance_count / n_samples
        return np.array(accepted_samples), acceptance_count, acceptance_rate