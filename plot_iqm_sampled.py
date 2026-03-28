#!/usr/bin/env python3
"""
Plot IQM (Interquartile Mean) with standard error for benchmark runs.
Samples n data points from each run to fix step mismatch problems.
Focuses only on the 'eval/episode_success_any' metric.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def calculate_iqm_with_se(data_array):
    """
    Calculate Interquartile Mean (IQM) and its standard error.
    
    Args:
        data_array: 2D numpy array of shape (n_runs, n_points)
        
    Returns:
        iqm: Interquartile mean across runs for each point
        se: Standard error of the IQM for each point
        lower_bound: IQM - SE
        upper_bound: IQM + SE
    """
    n_runs = data_array.shape[0]
    
    # Calculate IQM: mean of middle 50% of data (between 25th and 75th percentiles)
    iqm = np.zeros(data_array.shape[1])
    se = np.zeros(data_array.shape[1])
    
    for i in range(data_array.shape[1]):
        step_data = data_array[:, i]
        
        # Calculate percentiles
        q25 = np.percentile(step_data, 25)
        q75 = np.percentile(step_data, 75)
        
        # Get data within interquartile range
        iqr_data = step_data[(step_data >= q25) & (step_data <= q75)]
        
        if len(iqr_data) > 0:
            # IQM is mean of IQR data
            iqm[i] = np.mean(iqr_data)
            
            # Standard error of the mean for IQR data
            if len(iqr_data) > 1:
                se[i] = np.std(iqr_data, ddof=1) / np.sqrt(len(iqr_data))
            else:
                se[i] = 0
        else:
            # Fallback to regular mean if no data in IQR
            iqm[i] = np.mean(step_data)
            se[i] = np.std(step_data, ddof=1) / np.sqrt(n_runs) if n_runs > 1 else 0
    
    lower_bound = iqm - se
    upper_bound = iqm + se
    
    return iqm, se, lower_bound, upper_bound


def load_and_sample_runs_data(runs_dir, n_points=100, success_column='eval/episode_success_any'):
    """
    Load all CSV runs from a directory and sample n points from each.
    
    Args:
        runs_dir: Directory containing CSV files
        n_points: Number of points to sample from each run
        success_column: Column name for success rate
        
    Returns:
        sampled_success_data: 2D numpy array of shape (n_runs, n_points)
        sampled_steps: 1D numpy array of sampled step values
        run_files: List of file names
    """
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(f"Directory not found: {runs_dir}")
    
    csv_files = list(runs_dir.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {runs_dir}")
    
    all_success_data = []
    all_steps_data = []
    run_files = []
    
    for csv_file in sorted(csv_files):
        try:
            df = pd.read_csv(csv_file)
            
            # Check if success column exists
            if success_column not in df.columns:
                # Try alternative column names
                if 'success_rate' in df.columns:
                    success_data = df['success_rate'].values
                elif 'eval_episode_success_any' in df.columns:
                    success_data = df['eval_episode_success_any'].values
                else:
                    print(f"Warning: Success column '{success_column}' not found in {csv_file.name}")
                    print(f"Available columns: {list(df.columns)}")
                    continue
            else:
                success_data = df[success_column].values
            
            # Get steps
            if 'step' in df.columns:
                steps = df['step'].values
            elif '_step' in df.columns:
                steps = df['_step'].values
            else:
                # Use index as steps
                steps = np.arange(len(success_data))
            
            # Remove NaN values
            valid_mask = ~np.isnan(success_data)
            if np.any(valid_mask):
                success_data = success_data[valid_mask]
                steps = steps[valid_mask]
                
                all_success_data.append(success_data)
                all_steps_data.append(steps)
                run_files.append(csv_file.name)
            else:
                print(f"Warning: No valid success data in {csv_file.name}")
                
        except Exception as e:
            print(f"Error loading {csv_file.name}: {e}")
    
    if not all_success_data:
        raise ValueError(f"No valid success data found in {runs_dir}")
    
    # Sample n points from each run
    sampled_success_data = []
    sampled_steps_list = []
    
    for success_data, steps in zip(all_success_data, all_steps_data):
        n_total = len(success_data)
        
        if n_total <= n_points:
            # If we have fewer points than requested, use all points
            sampled_success_data.append(success_data)
            sampled_steps_list.append(steps)
        else:
            # Sample n_points evenly spaced indices
            indices = np.linspace(0, n_total - 1, n_points, dtype=int)
            sampled_success_data.append(success_data[indices])
            sampled_steps_list.append(steps[indices])
    
    # Convert to numpy arrays
    sampled_success_array = np.array(sampled_success_data)
    
    # Use steps from first run (they should be similar across runs after sampling)
    sampled_steps = sampled_steps_list[0]
    
    return sampled_success_array, sampled_steps, run_files


def plot_iqm_sampled(runs_dir, model_name, env_name, n_points=100,
                     success_column='eval/episode_success_any',
                     ax=None, color=None, label=None,
                     show_plot=True, save_path=None):
    """
    Plot IQM with standard error for a model's runs, sampling n points from each run.
    
    Args:
        runs_dir: Directory containing CSV files
        model_name: Name of the model (for labeling)
        env_name: Name of the environment (for labeling)
        n_points: Number of points to sample from each run
        success_column: Column name for success rate
        ax: Matplotlib axis to plot on (if None, creates new figure)
        color: Color for the plot
        label: Label for the plot (if None, uses model_name)
        show_plot: Whether to show the plot
        save_path: Path to save the plot (if None, doesn't save)
        
    Returns:
        fig, ax: Figure and axis objects
        iqm_data: Dictionary containing IQM statistics
    """
    # Load and sample data
    try:
        sampled_success, sampled_steps, run_files = load_and_sample_runs_data(
            runs_dir, n_points, success_column
        )
    except Exception as e:
        print(f"Error loading data from {runs_dir}: {e}")
        return None, None, None
    
    print(f"Loaded {len(run_files)} runs from {runs_dir}, sampled {n_points} points each")
    
    # Calculate IQM and standard error
    iqm, se, lower_bound, upper_bound = calculate_iqm_with_se(sampled_success)
    
    # Calculate final statistics
    final_success = sampled_success[:, -1]  # Last sampled point from each run
    mean_final = np.mean(final_success)
    std_final = np.std(final_success)
    median_final = np.median(final_success)
    
    # Create plot if no axis provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
        created_new_figure = True
    else:
        fig = ax.figure
        created_new_figure = False
    
    # Set default color if not provided
    if color is None:
        color = 'blue'
    
    # Set default label if not provided
    if label is None:
        label = model_name
    
    # Plot IQM line
    ax.plot(sampled_steps, iqm, '-', linewidth=2.5, color=color, 
            alpha=0.9, label=f"{label} (final: {mean_final:.3f})")
    
    # Plot standard error shading
    ax.fill_between(sampled_steps, lower_bound, upper_bound, 
                    alpha=0.2, color=color, label=f"{label} ± SE")
    
    # Set labels and title
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Success Rate (eval/episode_success_any)', fontsize=12)
    
    if created_new_figure:
        ax.set_title(f'{model_name} - {env_name.replace("_", " ").title()}\n'
                     f'{len(run_files)} runs, {n_points} sampled points\n'
                     f'Final IQM: {iqm[-1]:.3f} ± {se[-1]:.3f}', 
                     fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    # Save plot if requested
    if save_path and created_new_figure:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    # Show plot if requested and we created the figure
    if show_plot and created_new_figure:
        plt.tight_layout()
        plt.show()
    elif created_new_figure:
        plt.close(fig)  # Close figure if not showing
    
    # Prepare return data
    iqm_data = {
        'iqm': iqm,
        'se': se,
        'steps': sampled_steps,
        'n_runs': len(run_files),
        'n_points': n_points,
        'mean_final': mean_final,
        'std_final': std_final,
        'median_final': median_final,
        'iqm_final': iqm[-1],
        'se_final': se[-1],
        'sampled_success': sampled_success
    }
    
    return fig, ax, iqm_data


def plot_multiple_models_comparison_sampled(model_configs, env_name, n_points=100,
                                            output_dir='env_plots'):
    """
    Plot multiple models' IQM curves for the same environment in a single plot.
    Samples n points from each run.
    
    Args:
        model_configs: List of dictionaries, each with:
            - 'runs_dir': Directory containing CSV files
            - 'model_name': Name of the model
            - 'color': (Optional) Color for the plot
            - 'label': (Optional) Label for the plot
            - 'success_column': (Optional) Column name for success rate
        env_name: Name of the environment
        n_points: Number of points to sample from each run
        output_dir: Directory to save the plot
        
    Returns:
        fig, ax: Figure and axis objects
        all_iqm_data: Dictionary mapping model names to their IQM data
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Colors for different models
    default_colors = ['orange', 'brown', 'green', 'pink', 'blue', 'purple', 'gray', 'red']
    
    all_iqm_data = {}
    
    # Plot each model
    for i, config in enumerate(model_configs):
        # Get color - use default if not specified or if None is explicitly set
        color = config.get('color')
        if color is None:  # If color is None or not provided
            color = default_colors[i % len(default_colors)]
        label = config.get('label', config['model_name'])
        success_column = config.get('success_column', 'eval/episode_success_any')
        
        print(f"\nProcessing {config['model_name']}...")
        
        fig, ax, iqm_data = plot_iqm_sampled(
            runs_dir=config['runs_dir'],
            model_name=config['model_name'],
            env_name=env_name,
            n_points=n_points,
            success_column=success_column,
            ax=ax,
            color=color,
            label=label,
            show_plot=False
        )
        
        if iqm_data is not None:
            all_iqm_data[config['model_name']] = iqm_data
            
            # Plot individual IQM data points
            steps = iqm_data['steps']
            iqm = iqm_data['iqm']
            ax.scatter(steps, iqm, color=color, alpha=1.0, s=30, zorder=3)
    
    # Set title and labels
    ax.set_title(f'{env_name.replace("_", " ").title()} - Model Comparison\n'
                 f'IQM with Standard Error', 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Training Steps', fontsize=14)
    ax.set_ylabel('Success Rate (eval/episode_success_any)', fontsize=14)
    # ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11)
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    comparison_file = f"{output_dir}/{env_name}_model_comparison_iqm_sampled_{n_points}.png"
    plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison plot to {comparison_file}")
    
    plt.tight_layout()
    # Don't show by default - just save
    plt.close(fig)
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"SUMMARY STATISTICS for {env_name} ({n_points} sampled points)")
    print(f"{'='*60}")
    for model_name, data in all_iqm_data.items():
        print(f"\n{model_name}:")
        print(f"  Runs: {data['n_runs']}")
        print(f"  Sampled Points: {data['n_points']}")
        print(f"  Final Mean: {data['mean_final']:.4f}")
        print(f"  Final Std: {data['std_final']:.4f}")
        print(f"  Final Median: {data['median_final']:.4f}")
        print(f"  Final IQM: {data['iqm_final']:.4f} ± {data['se_final']:.4f}")
    
    return fig, ax, all_iqm_data


def main():
    """Example usage of the plotting functions with sampling."""
    print("IQM with Standard Error Plotting Script (Sampled Points)")
    print("="*60)
    
    # Example 1: Plot single model with 50 sampled points
    print("\nExample 1: Plotting PPO Reacher runs (50 sampled points)...")
    
    # PPO reacher runs
    ppo_reacher_dir = "runs/ppo/reacher_runs"
    if os.path.exists(ppo_reacher_dir):
        fig, ax, iqm_data = plot_iqm_sampled(
            runs_dir=ppo_reacher_dir,
            model_name="PPO",
            env_name="reacher",
            n_points=50,
            success_column='success_rate',
            save_path="env_plots/ppo_reacher_iqm_sampled_50.png"
        )
    else:
        print(f"Directory not found: {ppo_reacher_dir}")
    
    # Example 2: Plot multiple models comparison for ant_ball with 50 points
    print("\n" + "="*60)
    print("Example 2: Comparing multiple models for ant_ball (50 sampled points)...")
    
    model_configs = []
    
    # Check which models have ant_ball data
    potential_models = [
        ("PPO", "runs/ppo/ant_ball_runs", "success_rate"),
        ("SAC", "runs/sac/ant_ball", "eval/episode_success_any"),
        ("SAC+HER", "runs/sac_her/ant_ball", "eval/episode_success_any"),
        ("TD3", "runs/td3/ant_ball", "eval/episode_success_any")
    ]
    
    for model_name, runs_dir, success_column in potential_models:
        if os.path.exists(runs_dir):
            model_configs.append({
                'runs_dir': runs_dir,
                'model_name': model_name,
                'success_column': success_column
            })
            print(f"  Found {model_name} data at {runs_dir}")
    
    if len(model_configs) >= 2:
        fig, ax, all_iqm_data = plot_multiple_models_comparison_sampled(
            model_configs=model_configs,
            env_name="ant_ball",
            n_points=50,
            output_dir="env_plots"
        )
    else:
        print("Need at least 2 models with data for comparison plot.")
    
    print("\n" + "="*60)
    print("Script execution complete!")
    print("="*60)


if __name__ == "__main__":
    main()