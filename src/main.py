import numpy as np
from baseline_sampler import BaselineSampler
from uniform_sampler import UniformSampler
from two_dim_sampler import GaussianSampler2D
from mixture_proposal import MixtureProposal
from experiments.targets import (
    correlated_gaussian_pdf,
    gaussian_pdf,
    bimodal_pdf,
    complex_multimodal_pdf,
    gaussian_pdf_cupy,
    bimodal_pdf_cupy,
    complex_multimodal_pdf_cupy,
)
from experiments.plotting import (
    plot_1d_sampling_result,
    plot_2d_samples_with_contours,
    plot_runtime_benchmark,
)
from experiments.metrics import time_function
from whitening_transform import WhitenedGaussianSampler2D


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

def run_uniform_experiment(target_name, target_pdf, lower, upper, n_samples=10000, n_trials=5, plot_path=None):
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

def run_mixture_experiment(target_name, target_pdf, x_min, x_max, n_samples=10000, n_trials=5, plot_path=None):
    sampler = MixtureProposal(f_pdf=target_pdf, scale=1.3)
    sampler.set_proposal(x_range=(x_min, x_max))

    x_grid = np.linspace(x_min, x_max, 10000)
    M = sampler.find_M(x_grid)

    trial_results = []

    for _ in range(n_trials):
        (accepted_samples, acceptance_count, acceptance_rate), elapsed_time = time_function(
            sampler.rejection_sample,
            n_samples
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

    print(f"\n=== {target_name} (Mixture Gaussian Proposal) ===")
    print(f"Detected proposal components: {len(sampler.g_mus)}")
    print(f"Proposal means: {sampler.g_mus}")
    print(f"Proposal sigmas: {sampler.g_sigmas}")
    print(f"Proposal weights: {sampler.weights}")
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
            plt_title="Rejection Sampling with Mixture Gaussian Proposal"
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

def make_2d_grid(xmin, xmax, ymin, ymax, n_points=200):
    x = np.linspace(xmin, xmax, n_points)
    y = np.linspace(ymin, ymax, n_points)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.ravel(), Y.ravel()])
    return X, Y, points

def run_whitening_experiment():
    mu = np.array([0.0, 0.0])
    cov = np.array([
        [4.0, 3.2],
        [3.2, 4.0]
    ])

    n_samples = 50000

    # ----- original target in x-space -----
    X1, X2, X_grid = make_2d_grid(-8, 8, -8, 8, n_points=200)
    Z_x = correlated_gaussian_pdf(X_grid, mu, cov).reshape(X1.shape)

    # ----- baseline: direct sampling in x-space -----
    baseline_sampler = GaussianSampler2D(mu=mu, scale=2.5)
    baseline_sampler.find_M(
        X_grid,
        lambda X: correlated_gaussian_pdf(X, mu, cov)
    )

    (accepted_X_baseline, count_baseline, rate_baseline), time_baseline = time_function(
        baseline_sampler.vectorized_rejection_sample,
        n_samples,
        lambda X: correlated_gaussian_pdf(X, mu, cov)
    )

    print("\n=== 2D Baseline (No Whitening) ===")
    print(f"M: {baseline_sampler.M:.6f}")
    print(f"Accepted samples: {count_baseline}")
    print(f"Acceptance rate: {rate_baseline:.4f}")
    print(f"Runtime (s): {time_baseline:.6f}")

    plot_2d_samples_with_contours(
        X1, X2, Z_x,
        samples=accepted_X_baseline,
        title="2D Correlated Gaussian Target (No Whitening)",
        save_path="../figs/whitening_baseline_no_whitening.png"
    )

    # ----- whitened target in y-space -----
    whitened_sampler = WhitenedGaussianSampler2D(mu=mu, cov=cov, scale=1.5)

    Y1, Y2, Y_grid = make_2d_grid(-6, 6, -6, 6, n_points=200)
    Z_y = whitened_sampler.target_pdf_y(Y_grid).reshape(Y1.shape)

    whitened_sampler.find_M(Y_grid)

    (accepted_Y, accepted_X_whitened, count_whitened, rate_whitened), time_whitened = time_function(
        whitened_sampler.vectorized_rejection_sample,
        n_samples
    )

    print("\n=== 2D Whitening ===")
    print(f"M: {whitened_sampler.M:.6f}")
    print(f"Accepted samples: {count_whitened}")
    print(f"Acceptance rate: {rate_whitened:.4f}")
    print(f"Runtime (s): {time_whitened:.6f}")

    plot_2d_samples_with_contours(
        Y1, Y2, Z_y,
        samples=accepted_Y,
        title="Whitened Space y (Approximately Isotropic)",
        save_path="../figs/whitening_y_space.png"
    )

    plot_2d_samples_with_contours(
        X1, X2, Z_x,
        samples=accepted_X_whitened,
        title="Accepted Samples Mapped Back to x-Space",
        save_path="../figs/whitening_mapped_back_x_space.png"
    )

def gpu_available():
    try:
        import cupy as cp
        _ = cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False

def run_sampling_trials(sample_fn, n_trials=5):
    trial_results = []

    for _ in range(n_trials):
        (accepted_samples, acceptance_count, acceptance_rate), elapsed_time = time_function(sample_fn)

        trial_results.append({
            "acceptance_count": acceptance_count,
            "acceptance_rate": acceptance_rate,
            "runtime": elapsed_time,
        })

    runtimes = np.array([trial["runtime"] for trial in trial_results])
    acceptance_rates = np.array([trial["acceptance_rate"] for trial in trial_results])
    acceptance_counts = np.array([trial["acceptance_count"] for trial in trial_results])

    return {
        "median_runtime": np.median(runtimes),
        "runtime_std": np.std(runtimes),
        "median_acceptance_rate": np.median(acceptance_rates),
        "median_acceptance_count": np.median(acceptance_counts),
    }


def print_benchmark_case(case):
    print(f"\n=== Benchmark: {case['label']} ===")
    sequential_runtime = case["results"]["Sequential"]["median_runtime"]

    for method_name, summary in case["results"].items():
        speedup = sequential_runtime / summary["median_runtime"]
        print(f"{method_name}:")
        print(f"  Median runtime (s): {summary['median_runtime']:.6f}")
        print(f"  Runtime std (s): {summary['runtime_std']:.6f}")
        print(f"  Median acceptance rate: {summary['median_acceptance_rate']:.4f}")
        print(f"  Median accepted samples: {summary['median_acceptance_count']:.0f}")
        print(f"  Speedup vs sequential: {speedup:.2f}x")


def build_baseline_sampler(target_pdf, f_mu, f_sigma, x_min, x_max):
    sampler = BaselineSampler(f_mu=f_mu, f_sigma=f_sigma, scale=1.5)
    sampler.set_proposal()
    x_grid = np.linspace(x_min, x_max, 10000)
    sampler.find_M(target_pdf, x_grid)
    return sampler


def build_uniform_sampler(target_pdf, lower, upper):
    sampler = UniformSampler(lower=lower, upper=upper)
    sampler.set_proposal()
    x_grid = np.linspace(lower, upper, 10000)
    sampler.find_M(target_pdf, x_grid)
    return sampler


def build_mixture_sampler(target_pdf, x_min, x_max, target_pdf_cupy=None):
    sampler = MixtureProposal(
        f_pdf=target_pdf,
        scale=1.3,
        f_pdf_cupy=target_pdf_cupy
    )
    sampler.set_proposal(x_range=(x_min, x_max))
    x_grid = np.linspace(x_min, x_max, 10000)
    sampler.find_M(x_grid)
    return sampler


def benchmark_case(label, sampler, n_samples, n_trials, cpu_target_pdf=None, gpu_target_pdf=None, has_gpu=False):
    results = {}

    if cpu_target_pdf is None:
        results["Sequential"] = run_sampling_trials(
            lambda: sampler.rejection_sample(n_samples),
            n_trials=n_trials
        )
        results["Vectorized"] = run_sampling_trials(
            lambda: sampler.vectorized_rejection_sample(n_samples),
            n_trials=n_trials
        )
        if has_gpu and gpu_target_pdf is not None:
            results["GPU"] = run_sampling_trials(
                lambda: sampler.gpu_rejection_sample(n_samples),
                n_trials=n_trials
            )
    else:
        results["Sequential"] = run_sampling_trials(
            lambda: sampler.rejection_sample(n_samples, cpu_target_pdf),
            n_trials=n_trials
        )
        results["Vectorized"] = run_sampling_trials(
            lambda: sampler.vectorized_rejection_sample(n_samples, cpu_target_pdf),
            n_trials=n_trials
        )
        if has_gpu and gpu_target_pdf is not None:
            results["GPU"] = run_sampling_trials(
                lambda: sampler.gpu_rejection_sample(n_samples, gpu_target_pdf),
                n_trials=n_trials
            )

    case = {
        "label": label,
        "results": results,
    }

    print_benchmark_case(case)
    return case


def run_runtime_benchmarks(n_samples=300000, n_trials=5, plot_path="../figs/runtime_benchmark.png"):
    has_gpu = gpu_available()
    print(f"\nGPU available: {has_gpu}")
    print(f"Benchmark sample count: {n_samples}")
    print(f"Trials per method: {n_trials}")

    benchmark_cases = []

    # Baseline sampler
    benchmark_cases.append(
        benchmark_case(
            label="Baseline\nGaussian",
            sampler=build_baseline_sampler(gaussian_pdf, 0.0, 1.0, -6, 6),
            n_samples=n_samples,
            n_trials=n_trials,
            cpu_target_pdf=gaussian_pdf,
            gpu_target_pdf=gaussian_pdf_cupy,
            has_gpu=has_gpu
        )
    )

    benchmark_cases.append(
        benchmark_case(
            label="Baseline\nBimodal",
            sampler=build_baseline_sampler(bimodal_pdf, 0.0, 2.0, -8, 8),
            n_samples=n_samples,
            n_trials=n_trials,
            cpu_target_pdf=bimodal_pdf,
            gpu_target_pdf=bimodal_pdf_cupy,
            has_gpu=has_gpu
        )
    )

    benchmark_cases.append(
        benchmark_case(
            label="Baseline\nMultimodal",
            sampler=build_baseline_sampler(complex_multimodal_pdf, 0.0, 3.0, -8, 8),
            n_samples=n_samples,
            n_trials=n_trials,
            cpu_target_pdf=complex_multimodal_pdf,
            gpu_target_pdf=complex_multimodal_pdf_cupy,
            has_gpu=has_gpu
        )
    )

    # Uniform sampler
    benchmark_cases.append(
        benchmark_case(
            label="Uniform\nGaussian",
            sampler=build_uniform_sampler(gaussian_pdf, -6, 6),
            n_samples=n_samples,
            n_trials=n_trials,
            cpu_target_pdf=gaussian_pdf,
            gpu_target_pdf=gaussian_pdf_cupy,
            has_gpu=has_gpu
        )
    )

    benchmark_cases.append(
        benchmark_case(
            label="Uniform\nBimodal",
            sampler=build_uniform_sampler(bimodal_pdf, -8, 8),
            n_samples=n_samples,
            n_trials=n_trials,
            cpu_target_pdf=bimodal_pdf,
            gpu_target_pdf=bimodal_pdf_cupy,
            has_gpu=has_gpu
        )
    )

    benchmark_cases.append(
        benchmark_case(
            label="Uniform\nMultimodal",
            sampler=build_uniform_sampler(complex_multimodal_pdf, -8, 8),
            n_samples=n_samples,
            n_trials=n_trials,
            cpu_target_pdf=complex_multimodal_pdf,
            gpu_target_pdf=complex_multimodal_pdf_cupy,
            has_gpu=has_gpu
        )
    )

    # Mixture sampler
    benchmark_cases.append(
        benchmark_case(
            label="Mixture\nGaussian",
            sampler=build_mixture_sampler(gaussian_pdf, -6, 6, gaussian_pdf_cupy),
            n_samples=n_samples,
            n_trials=n_trials,
            has_gpu=has_gpu,
            gpu_target_pdf=gaussian_pdf_cupy
        )
    )

    benchmark_cases.append(
        benchmark_case(
            label="Mixture\nBimodal",
            sampler=build_mixture_sampler(bimodal_pdf, -8, 8, bimodal_pdf_cupy),
            n_samples=n_samples,
            n_trials=n_trials,
            has_gpu=has_gpu,
            gpu_target_pdf=bimodal_pdf_cupy
        )
    )

    benchmark_cases.append(
        benchmark_case(
            label="Mixture\nMultimodal",
            sampler=build_mixture_sampler(complex_multimodal_pdf, -8, 8, complex_multimodal_pdf_cupy),
            n_samples=n_samples,
            n_trials=n_trials,
            has_gpu=has_gpu,
            gpu_target_pdf=complex_multimodal_pdf_cupy
        )
    )

    plot_runtime_benchmark(
        benchmark_cases,
        save_path=plot_path,
        plt_title=f"Sequential vs Vectorized vs GPU Runtime Comparison (n={n_samples})"
    )

    return benchmark_cases

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
   
    # mixture gaussian sampler on single gaussian, bimodal, and multimodal
    mixture_results = []

    mixture_results.append(
        run_mixture_experiment(
            target_name="Gaussian Target",
            target_pdf=gaussian_pdf,
            x_min=-6,
            x_max=6,
            plot_path="../figs/mixture_gaussian_target.png"
        )
    )

    mixture_results.append(
        run_mixture_experiment(
            target_name="Bimodal Target",
            target_pdf=bimodal_pdf,
            x_min=-8,
            x_max=8,
            plot_path="../figs/mixture_bimodal_target.png"
        )
    )

    mixture_results.append(
        run_mixture_experiment(
            target_name="Complex Multimodal Target",
            target_pdf=complex_multimodal_pdf,
            x_min=-8,
            x_max=8,
            plot_path="../figs/mixture_complex_multimodal_target.png"
        )
    )

    # whitening experiment on correlated 2D Gaussian
    run_whitening_experiment()

    # high-sample runtime benchmarks
    run_runtime_benchmarks(
        n_samples=300000,
        n_trials=5,
        plot_path="../figs/runtime_benchmark.png"
    )

if __name__ == "__main__":
    main()