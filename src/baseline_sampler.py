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
        self.g_mu = self.f_mu
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

    def vectorized_rejection_sample(self, n_samples, f_pdf):
        # numpy batch version
        x = np.random.normal(loc=self.g_mu, scale=self.g_sigma, size=n_samples)
        u = np.random.rand(n_samples)

        accept_prob = f_pdf(x) / (self.M * self.g_pdf(x))
        accepted = x[u <= accept_prob]

        acceptance_count = len(accepted)
        acceptance_rate = acceptance_count / n_samples
        return accepted, acceptance_count, acceptance_rate

    def gpu_rejection_sample(self, n_samples, f_pdf):
        # cupy/torch version
        # TODO fix later for CUPY instead
        import torch

        if torch.cuda.is_available():
            device = torch.device("cuda") # if gpu is nvidia
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): # for mac silicon users 
            device = torch.device("mps")
        else:
            device = torch.device("cpu") # else use cpu

        x = self.g_mu + self.g_sigma * torch.randn(n_samples, device=device)
        u = torch.rand(n_samples, device=device)

        pi = torch.tensor(torch.pi, device=device)
        g_pdf = (1.0 / (torch.sqrt(2 * pi) * self.g_sigma)) * torch.exp(
            -0.5 * ((x - self.g_mu) / self.g_sigma) ** 2
        )

        accept_prob = f_pdf(x) / (self.M * g_pdf)
        accepted_samples = x[u <= accept_prob]

        acceptance_count = accepted_samples.numel()
        acceptance_rate = acceptance_count / n_samples

        return accepted_samples.cpu().numpy(), acceptance_count, acceptance_rate