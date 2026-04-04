import numpy as np
import matplotlib.pyplot as plt
from mixture_proposal import MixtureProposal

TARGET_COMPONENTS = [
    {"mu": -3.0, "sigma": 0.6, "weight": 0.4},
    {"mu": -1.0, "sigma": 0.4, "weight": 0.2},
    {"mu": 2.0, "sigma": 1.0, "weight": 0.4},
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
    f_mus = [comp["mu"] for comp in TARGET_COMPONENTS]
    f_sigmas = [comp["sigma"] for comp in TARGET_COMPONENTS]
    f_weights = [comp["weight"] for comp in TARGET_COMPONENTS]

    sampler = MixtureProposal(
        f_mu=0.0, f_sigma=1.0, n_components=len(TARGET_COMPONENTS), scale=1.5
    )
    sampler.set_proposal(target_mus=f_mus, target_weights=f_weights)

    # below is largely identical to single gaussian version
    x_grid = np.linspace(-6, 6, 10000)
    M = sampler.find_M(f_pdf, x_grid)

    accepted_samples, acceptance_count, acceptance_rate = sampler.rejection_sample(
        n_samples=10000, f_pdf=f_pdf
    )

    print(f"Target: Mixture of Gaussians")
    print(f"Target component mus: {f_mus}")
    print(f"Target component sigmas: {f_sigmas}")
    print(f"Target component weights: {f_weights}")
    print(f"\nProposal: Mixture of Gaussians")
    print(f"Number of components: {sampler.n_components}")
    print(f"Component mus: {sampler.g_mus}")
    print(f"Component sigmas: {sampler.g_sigmas}")
    print(f"Component weights: {sampler.weights}")
    print(f"\nM: {M}")
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
