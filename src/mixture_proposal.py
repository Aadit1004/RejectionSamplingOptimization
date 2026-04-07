import numpy as np
from scipy.signal import find_peaks


class MixtureProposal:
    def __init__(self, f_pdf, scale=2):
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

    def set_proposal(self, x_range):
        peak_mus, peak_sigmas, peak_weights = self._estimate_peaks(x_range)
        self.g_mus = peak_mus
        self.g_sigmas = peak_sigmas * self.scale
        self.weights = peak_weights

    def _estimate_peaks(self, x_range, n_points=1000):
        x_grid = np.linspace(x_range[0], x_range[1], n_points)
        y = self.f_pdf(x_grid)

        # Find peaks
        peaks, _ = find_peaks(y, prominence=1e-4)

        if len(peaks) == 0:
            peaks = np.array([np.argmax(y)])

        peak_mus = x_grid[peaks]
        peak_heights = y[peaks]
        peak_weights = peak_heights / np.sum(peak_heights)

        # Estimate sigmas from inter-peak spacing
        if len(peak_mus) > 1:
            mean_spacing = np.mean(np.diff(np.sort(peak_mus)))
            peak_sigmas = np.ones(len(peak_mus)) * mean_spacing * 0.3
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
        # safety_ceiling = np.sqrt(2 * np.pi * np.max(self.g_sigmas) ** 2)
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
            accept_prob = min(1.0, float(accept_prob))
            if u <= accept_prob:
                accepted_samples.append(x)

        acceptance_count = len(accepted_samples)
        acceptance_rate = acceptance_count / n_samples
        return np.array(accepted_samples), acceptance_count, acceptance_rate

    def vectorized_rejection_sample(self, n_samples):
        component_indices = np.random.choice(len(self.g_mus), size=n_samples, p=self.weights)
        x = np.random.normal(
            loc=self.g_mus[component_indices],
            scale=self.g_sigmas[component_indices]
        )
        u = np.random.rand(n_samples)

        accept_prob = self.f_pdf(x) / (self.M * self.g_pdf(x))
        accept_prob = np.minimum(accept_prob, 1.0)

        accepted_samples = x[u <= accept_prob]

        acceptance_count = len(accepted_samples)
        acceptance_rate = acceptance_count / n_samples
        return accepted_samples, acceptance_count, acceptance_rate
    
    def gpu_rejection_sample(self, n_samples):
        import cupy as cp

        if self.f_pdf_cupy is None:
            raise ValueError("gpu_rejection_sample requires a cupy compatible f_pdf_cupy")

        g_mus = cp.asarray(self.g_mus)
        g_sigmas = cp.asarray(self.g_sigmas)
        weights = cp.asarray(self.weights)

        component_indices = cp.random.choice(len(self.g_mus), size=n_samples, p=weights)
        x = cp.random.normal(
            loc=g_mus[component_indices],
            scale=g_sigmas[component_indices]
        )
        u = cp.random.rand(n_samples)

        density = cp.zeros_like(x, dtype=cp.float64)
        for i in range(len(self.g_mus)):
            gaussian = (1.0 / (cp.sqrt(2 * cp.pi) * g_sigmas[i])) * cp.exp(
                -0.5 * ((x - g_mus[i]) / g_sigmas[i]) ** 2
            )
            density += weights[i] * gaussian

        accept_prob = self.f_pdf_cupy(x) / (self.M * density)
        accept_prob = cp.minimum(accept_prob, 1.0)

        accepted_samples = x[u <= accept_prob]

        acceptance_count = int(accepted_samples.size)
        acceptance_rate = acceptance_count / n_samples
        return cp.asnumpy(accepted_samples), acceptance_count, acceptance_rate