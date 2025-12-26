import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator
import os

class PrecisionScalarFormatter(ScalarFormatter):
    def _set_format(self):
        self.format = "%.2f"

def plot_gd_solutions_style(npy_path, start_iter, end_iter, output_dir="solution_space_plots"):
    """
    Plots the first two dimensions of GD solutions with styling referenced from plot_solution_space.py.
    """
    
    if not os.path.exists(npy_path):
        print(f"Error: File not found at {npy_path}")
        return

    try:
        data = np.load(npy_path)
    except Exception as e:
        print(f"Error loading .npy file: {e}")
        return

    # Data shape: (total_iters, num_params)
    total_iters, num_dims = data.shape
    print(f"Loaded data with shape: {data.shape}")

    # Handle indexing
    # User input is likely 1-based (iteration count), array is 0-based.
    # Start: iteration 10 -> index 9
    start_idx = max(0, start_iter - 1)
    end_idx = min(total_iters, end_iter)

    if start_idx >= end_idx:
        print(f"Error: Invalid range {start_iter}-{end_iter} for data size {total_iters}.")
        return

    # Extract subset
    subset = data[start_idx:end_idx]
    if num_dims < 2:
        print("Error: Dimensions < 2.")
        return

    x = subset[:, 0]
    y = subset[:, 1]
    
    # --- Auto-Zoom Calculation (from reference) ---
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    
    span_x = max_x - min_x
    span_y = max_y - min_y
    
    # Prevent singular axis if all points are identical
    if span_x < 1e-9: span_x = 0.01
    if span_y < 1e-9: span_y = 0.01

    # Padding (2% from reference)
    plot_min_x = min_x - span_x * 0.1  # Increased padding slightly for scatter visibility
    plot_max_x = max_x + span_x * 0.1
    plot_min_y = min_y - span_y * 0.1
    plot_max_y = max_y + span_y * 0.1

    # Setup Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    base_cmap = plt.get_cmap('viridis')

    num_points = len(x)
    
    # Generate colors for each point based on progress
    # Progress from 0.0 to 1.0
    if num_points > 1:
        progress = np.linspace(0, 1, num_points)
    else:
        progress = np.array([0.5])
        
    colors = base_cmap(progress)

    # Plot Scatter with Legend
    # Plot each point individually to generate a legend entry
    for i in range(num_points):
        iter_num = start_iter + i
        # Ensure color is a 2D array (1, 4) for scatter or just pass tuple
        c = colors[i].reshape(1, -1)
        ax.scatter(x[i], y[i], c=c, s=200, edgecolors='white', linewidth=0.5, zorder=100, label=f"Iter {iter_num}")

    # Add legend to top right with larger, bold font matching ticks
    # Using size 16 to match the ticks as requested
    legend = ax.legend(loc='upper right', title="Iteration", prop={'size': 16, 'weight': 'bold'})
    legend.get_title().set_fontsize(18)
    legend.get_title().set_fontweight('bold')

    # Set Limits
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

    # Thicken borders (spines) to match reference style
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

    # Styling Ticks (from reference)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(20)
        label.set_fontweight('bold')

    # Ensure output directory
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = f"gd_{start_iter}_{end_iter}.pdf"
    output_path = os.path.join(output_dir, output_filename)
    
    # Save (using bbox_inches='tight' from reference)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot GD solution trajectory.")
    parser.add_argument("npy_file", type=str, help="Path to .npy file")
    parser.add_argument("--start", type=int, required=True, help="Start iteration")
    parser.add_argument("--end", type=int, required=True, help="End iteration")
    parser.add_argument("--output_dir", type=str, default="solution_space_plots")

    args = parser.parse_args()

    plot_gd_solutions_style(args.npy_file, args.start, args.end, args.output_dir)

