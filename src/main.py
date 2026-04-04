import numpy as np
import matplotlib.pyplot as plt
from baseline_sampler import BaselineSampler
from experiments.targets import gaussian_pdf
from experiments.plotting import plot_1d_sampling_result
from experiments.metrics import time_function, summarize_sampling   

def main():
    f_mu = 0.0
    f_sigma = 1.0
    n_samples = 10000

    sampler = BaselineSampler(f_mu=f_mu, f_sigma=f_sigma, scale=1.5)
    sampler.set_proposal()

    x_grid = np.linspace(-6, 6, 10000)
    M = sampler.find_M(gaussian_pdf, x_grid)

    (accepted_samples, acceptance_count, acceptance_rate), elapsed_time = time_function(
        sampler.rejection_sample,
        n_samples,
        gaussian_pdf
    )

    summary = summarize_sampling(
        acceptance_count=acceptance_count,
        n_samples=n_samples,
        elapsed_time=elapsed_time
    )

    print(f"Proposal mean: {sampler.g_mu}")
    print(f"Proposal sigma: {sampler.g_sigma}")
    print(f"M: {M}")
    print(f"Accepted samples: {summary['accepted']}")
    print(f"Acceptance rate: {summary['acceptance_rate']:.4f}")
    print(f"Elapsed time (s): {summary['elapsed_time_sec']:.6f}")

    plot_1d_sampling_result(
        x_grid=x_grid,
        target_pdf=gaussian_pdf(x_grid),
        proposal_pdf=sampler.g_pdf(x_grid),
        envelope_pdf=M * sampler.g_pdf(x_grid),
        samples=accepted_samples,
        save_path="../figs/test_baseline_sampler.png"
    )


if __name__ == "__main__":
    main()