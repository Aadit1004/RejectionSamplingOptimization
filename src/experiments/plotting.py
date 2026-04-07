import matplotlib.pyplot as plt

def plot_1d_sampling_result(x_grid, target_pdf, proposal_pdf, envelope_pdf, samples, save_path=None, plt_title="Rejection Sampling"):
    plt.figure(figsize=(10, 6))
    plt.plot(x_grid, target_pdf, label="Target f(x)")
    plt.plot(x_grid, proposal_pdf, label="Proposal g(x)", linestyle="--")
    plt.plot(x_grid, envelope_pdf, label="Proposal M * g(x)", linestyle="-.")
    plt.hist(samples, bins=50, density=True, alpha=0.6, label="Accepted samples")

    plt.xlabel("x")
    plt.ylabel("Density")
    plt.title(plt_title)

    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)

    plt.show()

def plot_2d_samples_with_contours(X, Y, Z, samples=None, title="2D Target Distribution", save_path=None):
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, Z, levels=30, alpha=0.75)
    plt.colorbar()

    if samples is not None and len(samples) > 0:
        plt.scatter(samples[:, 0], samples[:, 1], s=8, alpha=0.25)

    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")

    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_runtime_benchmark(benchmark_cases, save_path=None, plt_title="Runtime Comparison"):
    import numpy as np
    labels = [case["label"] for case in benchmark_cases]

    sequential_times = [case["results"]["Sequential"]["median_runtime"] for case in benchmark_cases]
    vectorized_times = [case["results"]["Vectorized"]["median_runtime"] for case in benchmark_cases]

    has_gpu = all("GPU" in case["results"] for case in benchmark_cases)

    x = np.arange(len(labels))
    width = 0.25 if has_gpu else 0.35

    plt.figure(figsize=(16, 7))

    if has_gpu:
        gpu_times = [case["results"]["GPU"]["median_runtime"] for case in benchmark_cases]
        plt.bar(x - width, sequential_times, width=width, label="Sequential")
        plt.bar(x, vectorized_times, width=width, label="Vectorized")
        plt.bar(x + width, gpu_times, width=width, label="GPU")
    else:
        plt.bar(x - width / 2, sequential_times, width=width, label="Sequential")
        plt.bar(x + width / 2, vectorized_times, width=width, label="Vectorized")

    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("Median runtime (s)")
    plt.title(plt_title)
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()