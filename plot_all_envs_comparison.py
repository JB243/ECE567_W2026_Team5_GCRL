#!/usr/bin/env python3
"""
Plot comparison of all models across all 4 environments.
Uses IQM with standard error and samples points for consistent comparison.
"""

import sys
sys.path.insert(0, '.')
from plot_iqm_sampled import plot_multiple_models_comparison_sampled
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

def get_available_models_for_env(env_name):
    """
    Return list of available models for a given environment.
    Checks which model directories exist and have data.
    """
    potential_models = []
    
    # Define all possible model configurations for this environment
    # Note: Different models may have different directory structures and column names
    
    # For PPO - uses {env}_runs directory structure and 'success_rate' column
    ppo_dir = f"runs/ppo/{env_name}_runs"
    if os.path.exists(ppo_dir):
        potential_models.append({
            'runs_dir': ppo_dir,
            'model_name': 'PPO',
            'success_column': 'success_rate',
            'label': 'PPO',
            'color': 'brown'
        })
    
    # For SAC - uses 'eval/episode_success_any' column
    sac_dir = f"runs/sac/{env_name}"
    if os.path.exists(sac_dir):
        potential_models.append({
            'runs_dir': sac_dir,
            'model_name': 'SAC',
            'success_column': 'eval/episode_success_any',
            'label': 'SAC',
            'color': 'orange'
        })
    
    # For SAC+HER - uses 'eval/episode_success_any' column
    sac_her_dir = f"runs/sac_her/{env_name}"
    if os.path.exists(sac_her_dir):
        potential_models.append({
            'runs_dir': sac_her_dir,
            'model_name': 'SAC_HER',
            'success_column': 'eval/episode_success_any',
            'label': 'SAC+HER',
            'color': 'mediumseagreen'
        })
    
    # For TD3 - special handling for different environments
    td3_dir = f"runs/td3/{env_name}"
    if os.path.exists(td3_dir) and not (env_name == 'reacher_table2'):
        # Check what column name to use for TD3
        
        potential_models.append({
            'runs_dir': td3_dir,
            'model_name': 'TD3',
            'success_column': 'eval/episode_success_any',
            'label': 'TD3',
            'color': 'violet'
        })
    
    # For TD3_HER - uses 'success_rate' column
    td3_her_dir = f"runs/td3_her/{env_name}"
    if os.path.exists(td3_her_dir):
        potential_models.append({
            'runs_dir': td3_her_dir,
            'model_name': 'TD3_HER',
            'success_column': 'success_rate',
            'label': 'TD3+HER',
            'color': 'wheat'

        })
    
    # For CRL - uses 'success_rate' column
    crl_dir = f"runs/crl/{env_name}"
    if os.path.exists(crl_dir):
        potential_models.append({
            'runs_dir': crl_dir,
            'model_name': 'CRL',
            'success_column': 'eval/episode_success_any',
            'label': 'CRL',
            'color': 'cornflowerblue'
        })
    
    return potential_models

def plot_env_comparison(env_name, n_points=50, output_dir="env_plots"):
    """
    Create comparison plot for a single environment.
    """
    print(f"\n{'='*70}")
    print(f"Comparing models on {env_name} environment...")
    print(f"{'='*70}")
    
    # Get available models for this environment
    model_configs = get_available_models_for_env(env_name)
    
    if len(model_configs) < 2:
        print(f"  Need at least 2 models. Found {len(model_configs)} for {env_name}.")
        print(f"  Available: {[m['model_name'] for m in model_configs]}")
        return None
    
    print(f"  Found {len(model_configs)} models:")
    for config in model_configs:
        print(f"    - {config['label']}: {config['runs_dir']}")
    
    # Create comparison plot
    print(f"\n  Creating comparison plot with {n_points} sampled points...")
    
    fig, ax, all_iqm_data = plot_multiple_models_comparison_sampled(
        model_configs=model_configs,
        env_name=env_name,
        n_points=n_points,
        output_dir=output_dir
    )
    
    # Create summary table
    print(f"\n  {'='*60}")
    print(f"  PERFORMANCE SUMMARY for {env_name} (sorted by final IQM)")
    print(f"  {'='*60}")
    
    # Sort by final IQM (descending)
    sorted_models = sorted(all_iqm_data.items(), 
                          key=lambda x: x[1]['iqm_final'], 
                          reverse=True)
    
    summary_data = []
    for model_name, data in sorted_models:
        # Find the label for this model
        label = next((cfg['label'] for cfg in model_configs 
                     if cfg['model_name'] == model_name), model_name)
        
        summary_data.append({
            'Model': label,
            'Runs': data['n_runs'],
            'Final IQM': f"{data['iqm_final']:.4f}",
            '± SE': f"{data['se_final']:.4f}",
            'Final Mean': f"{data['mean_final']:.4f}",
            'Final Median': f"{data['median_final']:.4f}"
        })
    
    # Print summary table
    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False))
    
    # Save summary to CSV
    summary_csv = f"{output_dir}/{env_name}_model_comparison_summary.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"\n  Saved summary to: {summary_csv}")
    
    return all_iqm_data

def create_final_summary_table(all_env_results, output_dir="env_plots"):
    """
    Create a final summary table comparing all models across all environments.
    """
    print(f"\n{'='*70}")
    print("FINAL SUMMARY: Model Performance Across All Environments")
    print(f"{'='*70}")
    
    # Collect data for final table
    final_data = []
    
    for env_name, env_results in all_env_results.items():
        if env_results is None:
            continue
            
        # Sort by final IQM for this environment
        sorted_models = sorted(env_results.items(), 
                              key=lambda x: x[1]['iqm_final'], 
                              reverse=True)
        
        # Get top 3 models for this environment
        for rank, (model_name, data) in enumerate(sorted_models[:3], 1):
            final_data.append({
                'Environment': env_name.replace('_', ' ').title(),
                'Rank': rank,
                'Model': model_name,
                'Final IQM': data['iqm_final'],
                '± SE': data['se_final'],
                'Runs': data['n_runs']
            })
    
    if not final_data:
        print("No data available for final summary.")
        return
    
    # Create DataFrame and sort
    df_final = pd.DataFrame(final_data)
    df_final = df_final.sort_values(['Environment', 'Rank'])
    
    # Print final summary
    print("\nTop 3 Models per Environment (sorted by Final IQM):")
    print("-" * 80)
    print(df_final.to_string(index=False))
    
    # Save final summary
    final_csv = f"{output_dir}/all_envs_top_models_summary.csv"
    df_final.to_csv(final_csv, index=False)
    print(f"\nSaved final summary to: {final_csv}")
    
    # Create a pivot table for easier comparison
    pivot_data = []
    for env_name, env_results in all_env_results.items():
        if env_results is None:
            continue
            
        for model_name, data in env_results.items():
            pivot_data.append({
                'Environment': env_name.replace('_', ' ').title(),
                'Model': model_name,
                'Final IQM': data['iqm_final'],
                'Final Mean': data['mean_final'],
                'Runs': data['n_runs']
            })
    
    if pivot_data:
        df_pivot = pd.DataFrame(pivot_data)
        pivot_table = df_pivot.pivot_table(
            index='Model', 
            columns='Environment', 
            values='Final IQM',
            aggfunc='first'
        ).fillna('N/A')
        
        print("\n\nFinal IQM Performance Matrix:")
        print("-" * 80)
        print(pivot_table.to_string())
        
        # Save pivot table
        pivot_csv = f"{output_dir}/all_envs_iqm_matrix.csv"
        pivot_table.to_csv(pivot_csv)
        print(f"\nSaved performance matrix to: {pivot_csv}")

def main():
    """Main function to plot comparisons for all 4 environments."""
    print("MODEL COMPARISON ACROSS ALL ENVIRONMENTS")
    print("="*70)
    print("Creating IQM comparison plots for 4 environments...")
    print("Using 50 sampled points per run for consistent comparison.")
    print("="*70)
    
    # Define the 4 environments to plot
    environments = [
        'ant_ball',
        'ant_u_maze', 
        'humanoid',
        'reacher'
    ]
    
    # Create output directory
    output_dir = "plots/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot each environment
    all_results = {}
    for env in environments:
        env_results = plot_env_comparison(env, n_points=11, output_dir=output_dir)
        all_results[env] = env_results
    
    # Create final summary
    create_final_summary_table(all_results, output_dir)
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print(f"All plots saved to: {output_dir}/")
    print(f"Summary files created:")
    print(f"  - Each environment: [env_name]_model_comparison_iqm_sampled_50.png")
    print(f"  - Each environment: [env_name]_model_comparison_summary.csv")
    print(f"  - Final summary: all_envs_top_models_summary.csv")
    print(f"  - Performance matrix: all_envs_iqm_matrix.csv")
    print(f"\nNext steps:")
    print(f"  1. Review the plots in {output_dir}/")
    print(f"  2. Check summary CSV files for detailed statistics")
    print(f"  3. Use the performance matrix for cross-environment comparisons")

if __name__ == "__main__":
    main()