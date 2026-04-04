import numpy as np
from baseline_sampler import BaselineSampler


class MixtureProposal(BaselineSampler):
    def __init__(self, f_mu, f_sigma, n_components=3, scale=1.5):
        super().__init__(f_mu, f_sigma, scale)
        self.n_components = n_components

        self.g_mus = None
        self.g_sigmas = None
        self.weights = None

    def g_pdf(self, x):
        if self.g_mus is None:
            self.set_proposal()

        density = np.zeros_like(x, dtype=float)
        for i in range(self.n_components):
            gaussian = (1.0 / (np.sqrt(2 * np.pi) * self.g_sigmas[i])) * np.exp(
                -0.5 * ((x - self.g_mus[i]) / self.g_sigmas[i]) ** 2
            )
            density += self.weights[i] * gaussian

        return density

    def set_proposal(self, target_mus=None, target_weights=None):
        if target_mus is not None:
            self.g_mus = np.asarray(target_mus)
            self.n_components = len(self.g_mus)
        else:
            offset_range = self.f_sigma * self.scale
            self.g_mus = np.linspace(
                self.f_mu - offset_range, self.f_mu + offset_range, self.n_components
            )

        self.g_sigmas = np.ones(self.n_components) * self.f_sigma * self.scale
        if target_weights is not None:
            self.weights = np.asarray(target_weights)
        else:
            self.weights = np.ones(self.n_components) / self.n_components

        return self.g_mus, self.g_sigmas, self.weights

    def find_M(self, f_pdf, x_grid):
        g_vals = self.g_pdf(x_grid)
        f_vals = f_pdf(x_grid)

        valid_idx = g_vals > 1e-10
        ratios = np.zeros_like(g_vals)
        ratios[valid_idx] = f_vals[valid_idx] / g_vals[valid_idx]

        safety_scale = 1.05
        self.M = safety_scale * np.max(ratios)
        return self.M

    def rejection_sample(self, n_samples, f_pdf):
        if self.g_mus is None:
            self.set_proposal()

        accepted_samples = []
        for _ in range(n_samples):
            component_idx = np.random.choice(self.n_components, p=self.weights)
            x = np.random.normal(
                loc=self.g_mus[component_idx], scale=self.g_sigmas[component_idx]
            )
            u = np.random.rand()
            accept_prob = f_pdf(x) / (self.M * self.g_pdf(x))
            if u <= accept_prob:
                accepted_samples.append(x)

        acceptance_count = len(accepted_samples)
        acceptance_rate = acceptance_count / n_samples
        return np.array(accepted_samples), acceptance_count, acceptance_rate
