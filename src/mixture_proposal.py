import numpy as np
from scipy.signal import find_peaks


class MixtureProposal:
    def __init__(self, f_pdf, scale=1.5):
        self.f_pdf = f_pdf
        self.scale = scale

        self.g_mus = None
        self.g_sigmas = None
        self.weights = None
        self.M = None

    def g_pdf(self, x):
        density = np.zeros_like(x, dtype=float)
        for i in range(len(self.g_mus)):
            gaussian = (1.0 / (np.sqrt(2 * np.pi) * self.g_sigmas[i])) * np.exp(
                -0.5 * ((x - self.g_mus[i]) / self.g_sigmas[i]) ** 2
            )
            density += self.weights[i] * gaussian

        return density

    def set_proposal(self, x_range, height_threshold=0.01):
        peak_mus, peak_sigmas, peak_weights = self._estimate_components_from_pdf(
            x_range=x_range, height_threshold=height_threshold
        )
        self.g_mus = peak_mus
        self.g_sigmas = peak_sigmas
        self.weights = peak_weights

    def _estimate_components_from_pdf(
        self, x_range, n_points=1000, height_threshold=0.01
    ):
        x_grid = np.linspace(x_range[0], x_range[1], n_points)
        y = self.f_pdf(x_grid)

        # Find peaks
        peaks, properties = find_peaks(y, height=height_threshold)

        peak_mus = x_grid[peaks]
        peak_heights = y[peaks]
        peak_weights = peak_heights / np.sum(peak_heights)

        # Estimate sigmas from inter-peak spacing
        if len(peak_mus) > 1:
            mean_spacing = np.mean(np.diff(np.sort(peak_mus)))
            peak_sigmas = np.ones(len(peak_mus)) * mean_spacing * 0.4
        else:
            peak_sigmas = np.ones(len(peak_mus))

        return peak_mus, peak_sigmas, peak_weights

    def find_M(self, x_grid):
        g_vals = self.g_pdf(x_grid)
        f_vals = self.f_pdf(x_grid)

        valid_idx = g_vals > 1e-8
        ratios = np.zeros_like(g_vals)
        ratios[valid_idx] = f_vals[valid_idx] / g_vals[valid_idx]

        safety_scale = 1.05
        self.M = safety_scale * np.max(ratios)
        return self.M

    def rejection_sample(self, n_samples):
        accepted_samples = []
        for _ in range(n_samples):
            component_idx = np.random.choice(len(self.g_mus), p=self.weights)
            x = np.random.normal(
                loc=self.g_mus[component_idx], scale=self.g_sigmas[component_idx]
            )
            u = np.random.rand()
            accept_prob = self.f_pdf(x) / (self.M * self.g_pdf(x))
            if u <= accept_prob:
                accepted_samples.append(x)

        acceptance_count = len(accepted_samples)
        acceptance_rate = acceptance_count / n_samples
        return np.array(accepted_samples), acceptance_count, acceptance_rate
