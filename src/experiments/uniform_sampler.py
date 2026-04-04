import numpy as np

class UniformSampler:
    def __init__(self, lower=-5.0, upper=5.0):
        self.lower = lower
        self.upper = upper
        self.M = None

    def g_pdf(self, x):
        '''
        x = np.asarray(x)
        pdf = np.zeros_like(x, dtype=float)
        inside = (x >= self.lower) & (x <= self.upper)
        pdf[inside] = 1.0 / (self.upper - self.lower)
        return pdf
        '''
        x = np.asarray(x)
        return np.where(
            (x >= self.lower) & (x <= self.upper),
            1.0 / (self.upper - self.lower),
            0.0
        )
    
    def set_proposal(self):
        return self.lower, self.upper

    def find_M(self, f_pdf, x_grid):
        ratios = f_pdf(x_grid) / self.g_pdf(x_grid)
        finite_ratios = ratios[np.isfinite(ratios)] # this is to avoid div by 0
        safety_scale = 1.05
        self.M = safety_scale * np.max(finite_ratios)
        return self.M
    
    def rejection_sample(self, n_samples, f_pdf):
        accepted_samples = []

        for i in range(n_samples):
            x = np.random.uniform(low=self.lower, high=self.upper)
            u = np.random.rand()

            accept_prob = f_pdf(x) / (self.M * self.g_pdf(x))
            if u <= accept_prob:
                accepted_samples.append(x)

        acceptance_count = len(accepted_samples)
        acceptance_rate = acceptance_count / n_samples
        return np.array(accepted_samples), acceptance_count, acceptance_rate
    
    def vectorized_rejection_sample(self, n_samples, f_pdf):
        x = np.random.uniform(low=self.lower, high=self.upper, size=n_samples)
        u = np.random.rand(n_samples)

        accept_prob = f_pdf(x) / (self.M * self.g_pdf(x))
        accepted = x[u <= accept_prob]

        acceptance_count = len(accepted)
        acceptance_rate = acceptance_count / n_samples
        return accepted, acceptance_count, acceptance_rate
    
    def gpu_rejection_sample(self, n_samples, f_pdf):
        import cupy as cp

        x = cp.random.uniform(low=self.lower, high=self.upper, size=n_samples)
        u = cp.random.rand(n_samples)

        g_pdf = cp.full(n_samples, 1.0 / (self.upper - self.lower))

        accept_prob = f_pdf(x) / (self.M * g_pdf)
        accepted_samples = x[u <= accept_prob]

        acceptance_count = int(accepted_samples.size)
        acceptance_rate = acceptance_count / n_samples

        return cp.asnumpy(accepted_samples), acceptance_count, acceptance_rate