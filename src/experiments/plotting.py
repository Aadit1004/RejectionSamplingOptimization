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

def plot_2d_contours(X, Y, Z, title="2D Target Distribution"):
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, Z, levels=30)
    plt.colorbar()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()



# TODO: will need to plot comparisons for runtime between models (all combinations)
# TODO: will need to plot comparisons for acceptance rate between models (all combinations), how to know which is best?