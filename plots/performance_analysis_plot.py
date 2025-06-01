import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import numpy as np

def plot_performance_analysis(figsize=(5.8, 2.6), save_plots=True):
    """
    Create performance analysis plot showing model accuracy across task types and difficulty levels
    
    Parameters:
    -----------
    figsize : tuple
        Figure size (width, height) (default: (5.8, 2.6) for LaTeX compatibility)
    save_plots : bool
        Whether to save plots to files (default: True)
    """
    
    # Use science style with vibrant colors and no LaTeX
    plt.style.use(['science', 'vibrant', 'no-latex'])
    
    # Override font sizes - smaller for LaTeX compatibility
    plt.rcParams.update({
        'font.size': 9,           # Base font size
        'axes.labelsize': 10,     # Axis labels
        'axes.titlesize': 11,     # Title
        'xtick.labelsize': 8,     # X-axis tick labels
        'ytick.labelsize': 8,     # Y-axis tick labels
        'legend.fontsize': 8,     # Legend
    })
    
    # Performance data
    task_types = ['Inductive', 'Deductive', 'Abductive']
    task_accuracies = [0.2278, 0.0962, 0.0305]
    task_counts = [(41, 180), (15, 156), (5, 164)]
    
    difficulty_levels = ['Very Easy', 'Easy', 'Medium', 'Hard', 'Very Hard']
    difficulty_accuracies = [0.1495, 0.1596, 0.1212, 0.1089, 0.0707]
    difficulty_counts = [(16, 107), (15, 94), (12, 99), (11, 101), (7, 99)]
    
    overall_accuracy = 0.1220
    overall_count = (61, 500)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Colors
    color1 = 'C0'  # Blue for task types
    color2 = 'C1'  # Orange for difficulty levels
    
    # Plot 1: Task Types
    x1 = np.arange(len(task_types))
    bars1 = ax1.bar(x1, task_accuracies, color=color1, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('Task Type')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Performance by Task Type')
    ax1.set_xticks(x1)
    ax1.set_xticklabels(task_types, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')
    ax1.set_ylim(0, max(task_accuracies) * 1.2)
    
    # Add value labels on bars
    for i, (bar, acc, (correct, total)) in enumerate(zip(bars1, task_accuracies, task_counts)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{acc:.3f}\n({correct}/{total})',
                ha='center', va='bottom', fontsize=7)
    
    # Add overall accuracy reference line
    ax1.axhline(y=overall_accuracy, color='red', linestyle='--', alpha=0.7, linewidth=1.5, 
                label=f'Overall: {overall_accuracy:.3f}')
    ax1.legend(loc='upper right', framealpha=0.9)
    
    # Plot 2: Difficulty Levels
    x2 = np.arange(len(difficulty_levels))
    bars2 = ax2.bar(x2, difficulty_accuracies, color=color2, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax2.set_xlabel('Difficulty Level')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Performance by Difficulty')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(difficulty_levels, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')
    ax2.set_ylim(0, max(difficulty_accuracies) * 1.2)
    
    # Add value labels on bars
    for i, (bar, acc, (correct, total)) in enumerate(zip(bars2, difficulty_accuracies, difficulty_counts)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{acc:.3f}\n({correct}/{total})',
                ha='center', va='bottom', fontsize=7)
    
    # Add overall accuracy reference line
    ax2.axhline(y=overall_accuracy, color='red', linestyle='--', alpha=0.7, linewidth=1.5,
                label=f'Overall: {overall_accuracy:.3f}')
    ax2.legend(loc='upper right', framealpha=0.9)
    
    # Adjust spacing for better layout
    plt.subplots_adjust(left=0.08, bottom=0.20, right=0.95, top=0.90, wspace=0.35)
    
    # Save plots if requested
    if save_plots:
        plt.savefig('plots/performance_analysis_plot.pdf', bbox_inches='tight', dpi=300)
        plt.savefig('plots/performance_analysis_plot.png', dpi=300, bbox_inches='tight')
        print("Plots saved as 'performance_analysis_plot.pdf' and 'performance_analysis_plot.png'")
    
    plt.show()
    
    # Print summary statistics
    print("Performance Analysis Summary:")
    print(f"Overall Accuracy: {overall_accuracy:.3f} ({overall_count[0]}/{overall_count[1]})")
    print("\nTask Type Performance:")
    for task, acc, (correct, total) in zip(task_types, task_accuracies, task_counts):
        print(f"  {task:10s}: {acc:.3f} ({correct:2d}/{total:3d})")
    print("\nDifficulty Level Performance:")
    for diff, acc, (correct, total) in zip(difficulty_levels, difficulty_accuracies, difficulty_counts):
        print(f"  {diff:10s}: {acc:.3f} ({correct:2d}/{total:3d})")
    
    return fig, (ax1, ax2)

# Alternative single plot version
def plot_performance_analysis_single(figsize=(5.8, 2.6), save_plots=True):
    """
    Create a single grouped bar chart showing both task types and difficulty levels
    """
    
    # Use science style with vibrant colors and no LaTeX
    plt.style.use(['science', 'vibrant', 'no-latex'])
    
    # Override font sizes - smaller for LaTeX compatibility
    plt.rcParams.update({
        'font.size': 9,           # Base font size
        'axes.labelsize': 10,     # Axis labels
        'axes.titlesize': 11,     # Title
        'xtick.labelsize': 8,     # X-axis tick labels
        'ytick.labelsize': 8,     # Y-axis tick labels
        'legend.fontsize': 8,     # Legend
    })
    
    # Performance data
    categories = ['Inductive', 'Deductive', 'Abductive', '', 'Very Easy', 'Easy', 'Medium', 'Hard', 'Very Hard']
    accuracies = [0.2278, 0.0962, 0.0305, np.nan, 0.1495, 0.1596, 0.1212, 0.1089, 0.0707]
    counts = [(41, 180), (15, 156), (5, 164), (0, 0), (16, 107), (15, 94), (12, 99), (11, 101), (7, 99)]
    
    # Colors: blue for task types, orange for difficulty levels, white for separator
    colors = ['C0', 'C0', 'C0', 'white', 'C1', 'C1', 'C1', 'C1', 'C1']
    alphas = [0.8, 0.8, 0.8, 0.0, 0.8, 0.8, 0.8, 0.8, 0.8]
    
    overall_accuracy = 0.1220
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(categories))
    bars = []
    
    for i, (cat, acc, color, alpha) in enumerate(zip(categories, accuracies, colors, alphas)):
        if cat == '':  # Skip separator
            continue
        bar = ax.bar(i, acc, color=color, alpha=alpha, edgecolor='black', linewidth=0.5)
        bars.append(bar)
    
    ax.set_xlabel('Category')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Performance by Task Type and Difficulty')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')
    ax.set_ylim(0, 0.25)
    
    # Add value labels on bars (excluding separator)
    for i, (cat, acc, (correct, total)) in enumerate(zip(categories, accuracies, counts)):
        if cat == '' or np.isnan(acc):
            continue
        ax.text(i, acc + 0.005,
                f'{acc:.3f}\n({correct}/{total})',
                ha='center', va='bottom', fontsize=6)
    
    # Add overall accuracy reference line
    ax.axhline(y=overall_accuracy, color='red', linestyle='--', alpha=0.7, linewidth=1.5,
               label=f'Overall: {overall_accuracy:.3f}')
    
    # Add section labels
    ax.text(1, 0.24, 'Task Types', ha='center', va='top', fontweight='bold', fontsize=9)
    ax.text(6.5, 0.24, 'Difficulty Levels', ha='center', va='top', fontweight='bold', fontsize=9)
    
    # Add vertical separator line
    ax.axvline(x=3.5, color='gray', linestyle='-', alpha=0.5, linewidth=1)
    
    ax.legend(loc='upper right', framealpha=0.9)
    
    # Adjust spacing
    plt.subplots_adjust(left=0.10, bottom=0.20, right=0.95, top=0.88)
    
    # Save plots if requested
    if save_plots:
        plt.savefig('plots/performance_analysis_single_plot.pdf', bbox_inches='tight', dpi=300)
        plt.savefig('plots/performance_analysis_single_plot.png', dpi=300, bbox_inches='tight')
        print("Plots saved as 'performance_analysis_single_plot.pdf' and 'performance_analysis_single_plot.png'")
    
    plt.show()
    
    return fig, ax

if __name__ == "__main__":
    # Generate both versions
    print("Creating dual subplot version:")
    plot_performance_analysis()
    
    print("\nCreating single plot version:")
    plot_performance_analysis_single() 