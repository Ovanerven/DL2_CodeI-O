# Cell 1: Imports and Setup
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import numpy as np

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

# Cell 2: Define your model data
# MODEL 1 DATA (replace with your first model's results)
model1_name = "Qwen2.5-3B"
model1_data = {
    'task_types': ['Inductive', 'Deductive', 'Abductive'],
    'task_accuracies': [0.2278, 0.0962, 0.0305],
    'task_counts': [(41, 180), (15, 156), (5, 164)],
    
    'difficulty_levels': ['Very Easy', 'Easy', 'Medium', 'Hard', 'Very Hard'],
    'difficulty_accuracies': [0.1495, 0.1596, 0.1212, 0.1089, 0.0707],
    'difficulty_counts': [(16, 107), (15, 94), (12, 99), (11, 101), (7, 99)],
    
    'overall_accuracy': 0.1220,
    'overall_count': (61, 500)
}

# MODEL 2 DATA (replace with your second model's results)
model2_name = "Model-X"  # Change this name
model2_data = {
    'task_types': ['Inductive', 'Deductive', 'Abductive'],
    'task_accuracies': [0.25, 0.12, 0.08],  # Replace with actual values
    'task_counts': [(50, 200), (24, 200), (16, 200)],  # Replace with actual values
    
    'difficulty_levels': ['Very Easy', 'Easy', 'Medium', 'Hard', 'Very Hard'],
    'difficulty_accuracies': [0.18, 0.16, 0.15, 0.14, 0.12],  # Replace with actual values
    'difficulty_counts': [(36, 200), (32, 200), (30, 200), (28, 200), (24, 200)],  # Replace with actual values
    
    'overall_accuracy': 0.15,  # Replace with actual value
    'overall_count': (90, 600)  # Replace with actual values
}

# Cell 3: Task Type Comparison Plot
def plot_task_type_comparison(model1_data, model2_data, model1_name, model2_name, figsize=(5.8, 2.6)):
    """
    Create task type comparison plot
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Colors
    color1 = 'C0'  # Blue for model 1
    color2 = 'C1'  # Orange for model 2
    
    # Task Types Comparison
    x = np.arange(len(model1_data['task_types']))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, model1_data['task_accuracies'], width, 
                   label=model1_name, color=color1, alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, model2_data['task_accuracies'], width,
                   label=model2_name, color=color2, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Task Type')
    ax.set_ylabel('Accuracy')
    ax.set_title('Performance by Task Type')
    ax.set_xticks(x)
    ax.set_xticklabels(model1_data['task_types'])
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')
    
    # Set y-limit with some padding for text
    max_acc = max(max(model1_data['task_accuracies']), max(model2_data['task_accuracies']))
    ax.set_ylim(0, max_acc * 1.25)
    
    # Add overall accuracy reference lines
    ax.axhline(y=model1_data['overall_accuracy'], color=color1, linestyle='--', alpha=0.7, 
               linewidth=1.5, label=f'{model1_name} Overall: {model1_data["overall_accuracy"]:.3f}')
    ax.axhline(y=model2_data['overall_accuracy'], color=color2, linestyle='--', alpha=0.7, 
               linewidth=1.5, label=f'{model2_name} Overall: {model2_data["overall_accuracy"]:.3f}')
    
    # Add value labels on task type bars
    for bars, accuracies in [(bars1, model1_data['task_accuracies']),
                             (bars2, model2_data['task_accuracies'])]:
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_acc * 0.02,
                    f'{acc:.3f}',
                    ha='center', va='bottom', fontsize=7)
    
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    return fig, ax

# Cell 4: Difficulty Level Comparison Plot
def plot_difficulty_comparison(model1_data, model2_data, model1_name, model2_name, figsize=(5.8, 2.6)):
    """
    Create difficulty level comparison plot
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Colors
    color1 = 'C0'  # Blue for model 1
    color2 = 'C1'  # Orange for model 2
    
    # Difficulty Levels Comparison  
    x = np.arange(len(model1_data['difficulty_levels']))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, model1_data['difficulty_accuracies'], width,
                   label=model1_name, color=color1, alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, model2_data['difficulty_accuracies'], width,
                   label=model2_name, color=color2, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Difficulty Level')
    ax.set_ylabel('Accuracy')
    ax.set_title('Performance by Difficulty')
    ax.set_xticks(x)
    ax.set_xticklabels(model1_data['difficulty_levels'], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')
    
    # Set y-limit with some padding for text
    max_acc = max(max(model1_data['difficulty_accuracies']), max(model2_data['difficulty_accuracies']))
    ax.set_ylim(0, max_acc * 1.25)
    
    # Add overall accuracy reference lines
    ax.axhline(y=model1_data['overall_accuracy'], color=color1, linestyle='--', alpha=0.7, 
               linewidth=1.5, label=f'{model1_name} Overall: {model1_data["overall_accuracy"]:.3f}')
    ax.axhline(y=model2_data['overall_accuracy'], color=color2, linestyle='--', alpha=0.7, 
               linewidth=1.5, label=f'{model2_name} Overall: {model2_data["overall_accuracy"]:.3f}')
    
    # Add value labels on difficulty bars
    for bars, accuracies in [(bars1, model1_data['difficulty_accuracies']),
                             (bars2, model2_data['difficulty_accuracies'])]:
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_acc * 0.02,
                    f'{acc:.3f}',
                    ha='center', va='bottom', fontsize=7)
    
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    return fig, ax

# Cell 5: Generate Task Type Plot
print("Task Type Performance Comparison:")
fig1, ax1 = plot_task_type_comparison(model1_data, model2_data, model1_name, model2_name)

# Cell 6: Generate Difficulty Plot
print("\nDifficulty Level Performance Comparison:")
fig2, ax2 = plot_difficulty_comparison(model1_data, model2_data, model1_name, model2_name)

# Cell 7: Save plots (optional)
# Uncomment to save
# fig1.savefig('task_type_comparison.pdf', bbox_inches='tight', dpi=300)
# fig1.savefig('task_type_comparison.png', dpi=300, bbox_inches='tight')
# fig2.savefig('difficulty_comparison.pdf', bbox_inches='tight', dpi=300)
# fig2.savefig('difficulty_comparison.png', dpi=300, bbox_inches='tight')
# print("Plots saved!")

# Cell 8: Individual model plots (if you want them separately)
def plot_single_model(model_data, model_name, figsize=(5.8, 2.6)):
    """
    Create plot for a single model
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Colors
    color1 = 'C0'  # Blue for task types
    color2 = 'C1'  # Orange for difficulty levels
    
    # Plot 1: Task Types
    x1 = np.arange(len(model_data['task_types']))
    bars1 = ax1.bar(x1, model_data['task_accuracies'], color=color1, alpha=0.8, 
                    edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('Task Type')
    ax1.set_ylabel('Accuracy')
    ax1.set_title(f'{model_name} - Task Performance')
    ax1.set_xticks(x1)
    ax1.set_xticklabels(model_data['task_types'], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')
    
    # Set y-limit with padding
    max_task_acc = max(model_data['task_accuracies'])
    ax1.set_ylim(0, max_task_acc * 1.3)
    
    # Add value labels
    for bar, acc, (correct, total) in zip(bars1, model_data['task_accuracies'], model_data['task_counts']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max_task_acc * 0.02,
                f'{acc:.3f}\n({correct}/{total})',
                ha='center', va='bottom', fontsize=7)
    
    # Add overall accuracy reference line
    ax1.axhline(y=model_data['overall_accuracy'], color='red', linestyle='--', alpha=0.7, 
                linewidth=1.5, label=f'Overall: {model_data["overall_accuracy"]:.3f}')
    ax1.legend(loc='upper right', framealpha=0.9)
    
    # Plot 2: Difficulty Levels
    x2 = np.arange(len(model_data['difficulty_levels']))
    bars2 = ax2.bar(x2, model_data['difficulty_accuracies'], color=color2, alpha=0.8, 
                    edgecolor='black', linewidth=0.5)
    
    ax2.set_xlabel('Difficulty Level')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{model_name} - Difficulty Performance')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(model_data['difficulty_levels'], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')
    
    # Set y-limit with padding
    max_diff_acc = max(model_data['difficulty_accuracies'])
    ax2.set_ylim(0, max_diff_acc * 1.3)
    
    # Add value labels
    for bar, acc, (correct, total) in zip(bars2, model_data['difficulty_accuracies'], model_data['difficulty_counts']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max_diff_acc * 0.02,
                f'{acc:.3f}\n({correct}/{total})',
                ha='center', va='bottom', fontsize=7)
    
    # Add overall accuracy reference line
    ax2.axhline(y=model_data['overall_accuracy'], color='red', linestyle='--', alpha=0.7, 
                linewidth=1.5, label=f'Overall: {model_data["overall_accuracy"]:.3f}')
    ax2.legend(loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    plt.show()
    
    return fig, (ax1, ax2)

# Cell 9: Generate individual plots (optional)
# Uncomment to generate individual model plots
# print("Model 1 Performance:")
# fig3, axes3 = plot_single_model(model1_data, model1_name)

# print(f"\nModel 2 Performance:")
# fig4, axes4 = plot_single_model(model2_data, model2_name) 