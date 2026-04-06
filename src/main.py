import numpy as np
from baseline_sampler import BaselineSampler
from experiments.uniform_sampler import UniformSampler
from experiments.targets import gaussian_pdf, bimodal_pdf, complex_multimodal_pdf
from experiments.plotting import plot_1d_sampling_result
from experiments.metrics import time_function


def run_baseline_experiment(target_name, target_pdf, f_mu, f_sigma, x_min, x_max, n_samples=10000,
    n_trials=5, plot_path=None):
    sampler = BaselineSampler(f_mu=f_mu, f_sigma=f_sigma, scale=1.5)
    sampler.set_proposal()

    x_grid = np.linspace(x_min, x_max, 10000)
    M = sampler.find_M(target_pdf, x_grid)

    trial_results = []

    for _ in range(n_trials):
        (accepted_samples, acceptance_count, acceptance_rate), elapsed_time = time_function(
            sampler.rejection_sample,
            n_samples,
            target_pdf
        )

        trial_results.append({
            "samples": accepted_samples,
            "acceptance_count": acceptance_count,
            "acceptance_rate": acceptance_rate,
            "runtime": elapsed_time,
        })

    runtimes = np.array([trial["runtime"] for trial in trial_results])
    acceptance_rates = np.array([trial["acceptance_rate"] for trial in trial_results])
    acceptance_counts = np.array([trial["acceptance_count"] for trial in trial_results])

    median_runtime = np.median(runtimes)
    median_acceptance_rate = np.median(acceptance_rates)
    median_acceptance_count = np.median(acceptance_counts)

    representative_idx = np.argmin(np.abs(acceptance_rates - median_acceptance_rate))
    representative_run = trial_results[representative_idx]

    print(f"\n=== {target_name} ===")
    print(f"Proposal mean: {sampler.g_mu}")
    print(f"Proposal sigma: {sampler.g_sigma}")
    print(f"M: {M:.6f}")
    print(f"Median accepted samples: {median_acceptance_count:.0f}")
    print(f"Median acceptance rate: {median_acceptance_rate:.4f}")
    print(f"Median runtime (s): {median_runtime:.6f}")
    print(f"Runtime std (s): {np.std(runtimes):.6f}")

    if plot_path is not None:
        plot_1d_sampling_result(
            x_grid=x_grid,
            target_pdf=target_pdf(x_grid),
            proposal_pdf=sampler.g_pdf(x_grid),
            envelope_pdf=M * sampler.g_pdf(x_grid),
            samples=representative_run["samples"],
            save_path=plot_path,
            plt_title="Rejection Sampling with Single Gaussian Proposal"
        )

    return {
        "target_name": target_name,
        "sampler": sampler,
        "x_grid": x_grid,
        "M": M,
        "median_runtime": median_runtime,
        "runtime_std": np.std(runtimes),
        "median_acceptance_rate": median_acceptance_rate,
        "median_acceptance_count": median_acceptance_count,
        "representative_samples": representative_run["samples"],
    }

def run_uniform_experiment(
    target_name,
    target_pdf,
    lower,
    upper,
    n_samples=10000,
    n_trials=5,
    plot_path=None
):
    sampler = UniformSampler(lower=lower, upper=upper)
    sampler.set_proposal()

    x_grid = np.linspace(lower, upper, 10000)
    M = sampler.find_M(target_pdf, x_grid)

    trial_results = []

    for _ in range(n_trials):
        (accepted_samples, acceptance_count, acceptance_rate), elapsed_time = time_function(
            sampler.rejection_sample,
            n_samples,
            target_pdf
        )

        trial_results.append({
            "samples": accepted_samples,
            "acceptance_count": acceptance_count,
            "acceptance_rate": acceptance_rate,
            "runtime": elapsed_time,
        })

    runtimes = np.array([trial["runtime"] for trial in trial_results])
    acceptance_rates = np.array([trial["acceptance_rate"] for trial in trial_results])
    acceptance_counts = np.array([trial["acceptance_count"] for trial in trial_results])

    median_runtime = np.median(runtimes)
    median_acceptance_rate = np.median(acceptance_rates)
    median_acceptance_count = np.median(acceptance_counts)

    representative_idx = np.argmin(np.abs(acceptance_rates - median_acceptance_rate))
    representative_run = trial_results[representative_idx]

    print(f"\n=== {target_name} (Uniform Proposal) ===")
    print(f"Proposal lower: {sampler.lower}")
    print(f"Proposal upper: {sampler.upper}")
    print(f"M: {M:.6f}")
    print(f"Median accepted samples: {median_acceptance_count:.0f}")
    print(f"Median acceptance rate: {median_acceptance_rate:.4f}")
    print(f"Median runtime (s): {median_runtime:.6f}")
    print(f"Runtime std (s): {np.std(runtimes):.6f}")

    if plot_path is not None:
        plot_1d_sampling_result(
            x_grid=x_grid,
            target_pdf=target_pdf(x_grid),
            proposal_pdf=sampler.g_pdf(x_grid),
            envelope_pdf=M * sampler.g_pdf(x_grid),
            samples=representative_run["samples"],
            save_path=plot_path,
            plt_title="Rejection Sampling with Uniform Distribution Proposal"
        )

    return {
        "target_name": target_name,
        "sampler": sampler,
        "x_grid": x_grid,
        "M": M,
        "median_runtime": median_runtime,
        "runtime_std": np.std(runtimes),
        "median_acceptance_rate": median_acceptance_rate,
        "median_acceptance_count": median_acceptance_count,
        "representative_samples": representative_run["samples"],
    }

def main():
    # baseline sampler experiments on single gaussian, bimodel, and multimodal
    baseline_results = []

    baseline_results.append(
        run_baseline_experiment(
            target_name="Gaussian Target",
            target_pdf=gaussian_pdf,
            f_mu=0.0,
            f_sigma=1.0,
            x_min=-6,
            x_max=6,
            plot_path="../figs/baseline_gaussian_target.png"
        )
    )

    baseline_results.append(
        run_baseline_experiment(
            target_name="Bimodal Target",
            target_pdf=bimodal_pdf,
            f_mu=0.0,
            f_sigma=2.0,
            x_min=-8,
            x_max=8,
            plot_path="../figs/baseline_bimodal_target.png"
        )
    )

    baseline_results.append(
        run_baseline_experiment(
            target_name="Complex Multimodal Target",
            target_pdf=complex_multimodal_pdf,
            f_mu=0.0,
            f_sigma=3.0,
            x_min=-8,
            x_max=8,
            plot_path="../figs/baseline_complex_multimodal_target.png"
        )
    )

    # uniform sampler on single gaussian, bimodel, and multimodal
    uniform_results = []

    uniform_results.append(
        run_uniform_experiment(
            target_name="Gaussian Target",
            target_pdf=gaussian_pdf,
            lower=-6,
            upper=6,
            plot_path="../figs/uniform_gaussian_target.png"
        )
    )

    uniform_results.append(
        run_uniform_experiment(
            target_name="Bimodal Target",
            target_pdf=bimodal_pdf,
            lower=-8,
            upper=8,
            plot_path="../figs/uniform_bimodal_target.png"
        )
    )

    uniform_results.append(
        run_uniform_experiment(
            target_name="Complex Multimodal Target",
            target_pdf=complex_multimodal_pdf,
            lower=-8,
            upper=8,
            plot_path="../figs/uniform_complex_multimodal_target.png"
        )
    )


if __name__ == "__main__":
    main()