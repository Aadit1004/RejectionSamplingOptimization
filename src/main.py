import numpy as np
import matplotlib.pyplot as plt
from baseline_sampler import BaselineSampler


def f_pdf(x):
    return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)


def main():
    f_mu = 0.0
    f_sigma = 1.0

    sampler = BaselineSampler(f_mu=f_mu, f_sigma=f_sigma, scale=1.5)
    sampler.set_proposal()

    x_grid = np.linspace(-6, 6, 10000)
    M = sampler.find_M(f_pdf, x_grid)

    accepted_samples, acceptance_count, acceptance_rate = sampler.rejection_sample(
        n_samples=10000,
        f_pdf=f_pdf
    )

    print(f"Proposal mean: {sampler.g_mu}")
    print(f"Proposal sigma: {sampler.g_sigma}")
    print(f"M: {M}")
    print(f"Accepted samples: {acceptance_count}")
    print(f"Acceptance rate: {acceptance_rate:.4f}")

    plt.figure(figsize=(10, 6))

    plt.plot(x_grid, f_pdf(x_grid), label="Target f(x): N(0,1)", color="blue")
    plt.plot(x_grid, sampler.g_pdf(x_grid), label="Proposal g(x)", linestyle="--", color="red")
    plt.plot(x_grid, M * sampler.g_pdf(x_grid), label="Proposal M * g(x)", linestyle="-.", color="black")

    plt.hist(
        accepted_samples,
        bins=50,
        density=True,
        alpha=0.6,
        label="Accepted samples",
        color="limegreen"
    )

    plt.xlabel("x")
    plt.ylabel("Density")
    plt.title("Rejection Sampling with Gaussian Proposal")
    plt.legend()
    plt.grid(True)
    plt.savefig("../figs/test_baseline_sampler.png")
    plt.show()


if __name__ == "__main__":
    main()