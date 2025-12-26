import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter, MaxNLocator
from scipy.stats import multivariate_normal
import os

class PrecisionScalarFormatter(ScalarFormatter):
    def _set_format(self):
        self.format = "%.2f"

def plot_solution_space_evolution(file_path, start_step=None, end_step=None, save_dir="solution_space_plots"):
    """
    Plots the evolution of the solution space (Gaussian distribution) for the first two dimensions
    with filled gradients (heatmap style).
    
    Args:
        file_path: Path to the .npz history file.
        start_step: Starting iteration index (inclusive). If None, defaults to last 10 steps start.
        end_step: Ending iteration index (exclusive). If None, defaults to the end.
        save_dir: Directory to save the plot.
    """
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    try:
        data = np.load(file_path)
        center_hist = data['center']
        A_hist = data['A']
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    n_iters = center_hist.shape[0]
    dim = center_hist.shape[1]

    # --- Determine Range ---
    if end_step is None or end_step > n_iters:
        end_step = n_iters
    
    if start_step is None:
        # Default behavior: last 10 steps
        start_step = max(0, end_step - 10)
    
    # Validation
    start_step = max(0, start_step)
    if start_step >= end_step:
        print(f"Error: Start step ({start_step}) must be less than end step ({end_step}).")
        return

    print(f"Loaded data with {n_iters} iterations.")
    print(f"Plotting steps from {start_step} to {end_step-1} (Total: {end_step - start_step} frames)...")

    indices = np.arange(start_step, end_step)
    
    # Indices of dimensions to plot
    d1, d2 = 0, 1

    # --- Auto-Zoom Calculation ---
    min_x_list, max_x_list = [], []
    min_y_list, max_y_list = [], []

    for idx in indices:
        mu = center_hist[idx, [d1, d2]]
        A_curr = A_hist[idx]
        Sigma_full = A_curr @ A_curr.T
        Sigma_sub = Sigma_full[[d1, d2]][:, [d1, d2]]
        
        std_x = np.sqrt(Sigma_sub[0, 0])
        std_y = np.sqrt(Sigma_sub[1, 1])
        
        # Use 2.5 sigma for zooming
        zoom_sigma = 2.5
        min_x_list.append(mu[0] - zoom_sigma * std_x)
        max_x_list.append(mu[0] + zoom_sigma * std_x)
        min_y_list.append(mu[1] - zoom_sigma * std_y)
        max_y_list.append(mu[1] + zoom_sigma * std_y)
    
    plot_min_x = min(min_x_list)
    plot_max_x = max(max_x_list)
    plot_min_y = min(min_y_list)
    plot_max_y = max(max_y_list)
    
    # Padding
    span_x = plot_max_x - plot_min_x
    span_y = plot_max_y - plot_min_y
    if span_x < 1e-9: span_x = 0.01
    if span_y < 1e-9: span_y = 0.01

    plot_min_x -= span_x * 0.02
    plot_max_x += span_x * 0.02
    plot_min_y -= span_y * 0.02
    plot_max_y += span_y * 0.02

    # Grid
    x_grid = np.linspace(plot_min_x, plot_max_x, 300)
    y_grid = np.linspace(plot_min_y, plot_max_y, 300)
    X, Y = np.meshgrid(x_grid, y_grid)
    pos = np.dstack((X, Y))

    fig, ax = plt.subplots(figsize=(10, 8))
    base_cmap = plt.get_cmap('viridis') 
    
    for i, idx in enumerate(indices):
        mu = center_hist[idx, [d1, d2]]
        A_curr = A_hist[idx]
        Sigma_full = A_curr @ A_curr.T
        Sigma_sub = Sigma_full[[d1, d2]][:, [d1, d2]]
        
        rv = multivariate_normal(mu, Sigma_sub)
        Z = rv.pdf(pos)
        
        # Color progress relative to the SELECTED range
        progress = i / (len(indices) - 1) if len(indices) > 1 else 0.5
        main_color = base_cmap(progress)
        
        n_bins = 200
        colors_array = np.zeros((n_bins, 4))
        colors_array[:, :3] = main_color[:3] 
        colors_array[:, 3] = np.linspace(0.0, 0.6, n_bins)
        
        custom_cmap = LinearSegmentedColormap.from_list(f'cmap_{idx}', colors_array)
        
        z_peak = Z.max()
        levels = np.linspace(0, z_peak, 60)
        
        cntr = ax.contourf(X, Y, Z, levels=levels, cmap=custom_cmap, antialiased=True)
        ax.scatter(mu[0], mu[1], color=main_color, s=40, edgecolors='white', linewidth=0.5, zorder=100)

    ax.set_xlim(plot_min_x, plot_max_x)
    ax.set_ylim(plot_min_y, plot_max_y)

    # Use scientific notation offset with 2 decimal places for ticks
    fmt_x = PrecisionScalarFormatter(useMathText=True)
    fmt_x.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(fmt_x)

    # Limit number of ticks to prevent overlap
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    fmt_y = PrecisionScalarFormatter(useMathText=True)
    fmt_y.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(fmt_y)
    
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(20)
        label.set_fontweight('bold')
    
    # Save file with range in name
    filename = f'wave1d_{start_step}_{end_step}.pdf'
    output_file = os.path.join(save_dir, filename)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Gradient plot saved to {output_file}")

if __name__ == "__main__":
    file_path = "wave1d.npz"
    plot_solution_space_evolution(file_path, start_step=401, end_step=411)
