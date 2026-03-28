import wandb
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# specify the projects to download (without username prefix)
PROJECT_NAMES = [
    "jaxgcrl_benchmark_ant",
    "jaxgcrl_benchmark_humanoid", 
    "jaxgcrl_benchmark_ant_ball",
    "jaxgcrl_benchmark_ant_u_maze"
]

USER = "your_wandb_username"  # Replace with your actual wandb username

def download_project_runs(project_name):
    """
    Download all runs from a wandb project.
    
    Args:
        project_name: Full wandb project path (e.g., "your_wandb_username/jaxgcrl_benchmark_ant")
    
    Returns:
        Dictionary with metadata and saved file paths
    """
    print(f"\n{'='*60}")
    print(f"Processing project: {project_name}")
    print(f"{'='*60}")
    
    api = wandb.Api()
    
    try:
        # Get all runs from the project
        runs = api.runs(project_name)
        print(f"Found {len(runs)} total runs in project")
        
        if not runs:
            print(f"Warning: No runs found in project '{project_name}'")
            return None
        
        # Extract environment name from project name
        # Remove 'jackytu/' prefix if present
        if '/' in project_name:
            project_part = project_name.split('/')[1]
        else:
            project_part = project_name
        
        # Remove 'jaxgcrl_benchmark_' prefix if present
        if project_part.startswith('jaxgcrl_benchmark_'):
            env_name = project_part[len('jaxgcrl_benchmark_'):]
        else:
            env_name = project_part
        
        print(f"Environment name: {env_name}")
        
        # Convert to list and sort by name for consistent ordering
        runs_list = list(runs)
        runs_list.sort(key=lambda x: x.name)
        
        # Create directory for saved data
        save_dir = f"{env_name}_runs"
        os.makedirs(save_dir, exist_ok=True)
        
        # Save metadata about all runs
        metadata = {
            "project": project_name,
            "env_name": env_name,
            "total_runs_found": len(runs),
            "fetch_date": datetime.now().isoformat(),
            "runs": []
        }
        
        # Process each run
        for run in runs_list:
            print(f"\nProcessing {run.name} (ID: {run.id})...")
            
            run_data = {
                "name": run.name,
                "id": run.id,
                "config": run.config
            }
            
            # Get history data
            try:
                history_df = run.history(pandas=True)
                print(f"  Loaded {len(history_df)} rows of history data")
                
                # Extract success rate data if available
                if 'eval/episode_success_any' in history_df.columns:
                    success_mask = history_df['eval/episode_success_any'].notna()
                    success_data = history_df.loc[success_mask, 'eval/episode_success_any'].tolist()
                    steps = history_df.loc[success_mask, '_step'].tolist()
                    
                    # Save success data to CSV
                    success_df = pd.DataFrame({
                        'step': steps,
                        'success_rate': success_data
                    })
                    
                    csv_filename = f"{save_dir}/{run.name}_success_data.csv"
                    success_df.to_csv(csv_filename, index=False)
                    print(f"  Saved success data to {csv_filename} ({len(success_data)} points)")
                    
                    run_data["success_data_file"] = csv_filename
                    run_data["success_data_points"] = len(success_data)
                    run_data["final_success_rate"] = float(success_data[-1]) if success_data else None
                    run_data["max_success_rate"] = float(max(success_data)) if success_data else None
                    run_data["avg_success_rate"] = float(np.mean(success_data)) if success_data else None
                else:
                    print(f"  Warning: 'eval/episode_success_any' not found in {run.name}")
                    run_data["success_data_file"] = None
                
            except Exception as e:
                print(f"  Error processing history for {run.name}: {e}")
                run_data["error"] = str(e)
            
            metadata["runs"].append(run_data)
        
        # Save metadata
        metadata_file = f"{save_dir}/metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"\n✓ Successfully saved data for {len(runs_list)} runs to '{save_dir}/'")
        print(f"✓ Metadata saved to {metadata_file}")
        
        # Print summary
        print("\n=== Run Summary ===")
        final_rates = []
        for run_data in metadata["runs"]:
            if "final_success_rate" in run_data and run_data["final_success_rate"] is not None:
                final_rate = run_data["final_success_rate"]
                final_rates.append(final_rate)
                print(f"{run_data['name']}:")
                print(f"  Final success rate: {final_rate:.4f}")
                print(f"  Max success rate: {run_data['max_success_rate']:.4f}")
                print(f"  Data points: {run_data['success_data_points']}")
                print()
        
        if final_rates:
            print(f"Overall Statistics for {env_name}:")
            print(f"  Mean final success: {np.mean(final_rates):.4f}")
            print(f"  Std final success: {np.std(final_rates):.4f}")
            print(f"  Min final success: {np.min(final_rates):.4f}")
            print(f"  Max final success: {np.max(final_rates):.4f}")
        
        return metadata
        
    except Exception as e:
        print(f"Error processing {project_name}: {e}")
        return None

def main():
    """Main function to download all projects."""
    
    print(f"Found {len(PROJECT_NAMES)} projects to process")
    
    all_metadata = {}
    
    # Download data for each project
    for project_name in PROJECT_NAMES:
        # Add username prefix if not present
        if '/' not in project_name:
            project_name = f"{USER}/{project_name}"
        
        metadata = download_project_runs(project_name)
        if metadata:
            env_name = metadata.get("env_name", project_name)
            all_metadata[env_name] = metadata
    
    # Save overall summary
    if all_metadata:
        summary_file = "all_projects_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(all_metadata, f, indent=2, default=str)
        print(f"\n✓ Overall summary saved to {summary_file}")
    
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE")
    print("="*60)
    
    # Print quick overview
    for env_name, metadata in all_metadata.items():
        if metadata and "runs" in metadata:
            final_rates = []
            for run in metadata["runs"]:
                if run.get("final_success_rate"):
                    final_rates.append(run["final_success_rate"])
            
            if final_rates:
                print(f"{env_name}: {len(final_rates)} runs, avg success: {np.mean(final_rates):.4f}")

if __name__ == "__main__":
    main()