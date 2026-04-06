import numpy as np
import matplotlib.pyplot as plt
from mixture_proposal import MixtureProposal
import random

TARGET_COMPONENTS = [
    {"mu": random.uniform(-6, 6), "sigma": random.uniform(0.1, 2.0), "weight": 0.1},
    {"mu": random.uniform(-6, 6), "sigma": random.uniform(0.1, 2.0), "weight": 0.2},
    {"mu": random.uniform(-6, 6), "sigma": random.uniform(0.1, 2.0), "weight": 0.3},
    {"mu": random.uniform(-6, 6), "sigma": random.uniform(0.1, 2.0), "weight": 0.4},
]


def gaussian_component(x, mu, sigma, weight):
    return (
        weight
        * (1.0 / np.sqrt(2 * np.pi * sigma**2))
        * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    )


def f_pdf(x):
    result = np.zeros_like(x, dtype=float)
    for component in TARGET_COMPONENTS:
        result += gaussian_component(
            x, mu=component["mu"], sigma=component["sigma"], weight=component["weight"]
        )
    return result


def main():
    sampler = MixtureProposal(f_pdf)
    sampler.set_proposal(x_range=(-10, 10))

    x_grid = np.linspace(-10, 10, 10000)
    M = sampler.find_M(x_grid)

    accepted_samples, acceptance_count, acceptance_rate = sampler.rejection_sample(
        n_samples=10000
    )

    print(f"\nTarget: Mixture of Gaussians")
    print(f"Target components: {TARGET_COMPONENTS}")
    print(f"M: {M:.6f}")
    print(f"Accepted samples: {acceptance_count}")
    print(f"Acceptance rate: {acceptance_rate:.4f}")

    plt.figure(figsize=(10, 6))

    plt.plot(
        x_grid, f_pdf(x_grid), label="Target f(x): Mixture of Gaussians", color="blue"
    )
    plt.plot(
        x_grid,
        sampler.g_pdf(x_grid),
        label="Proposal g(x): Mixture of Gaussians",
        linestyle="--",
        color="red",
    )
    plt.plot(
        x_grid,
        M * sampler.g_pdf(x_grid),
        label="Proposal M * g(x)",
        linestyle="-.",
        color="black",
    )

    plt.hist(
        accepted_samples,
        bins=50,
        density=True,
        alpha=0.6,
        label="Accepted samples",
        color="limegreen",
    )

    plt.xlabel("x")
    plt.ylabel("Density")
    plt.title("Rejection Sampling with Mixture of Gaussians Proposal")
    plt.legend()
    plt.grid(True)
    plt.savefig("../figs/test_mixture_proposal.png")
    plt.show()


if __name__ == "__main__":
    main()
