import mlflow
import pandas as pd
import os

# Global variables
RESULTS_FOLDER="results"
OUTPUT_FILE = f"{RESULTS_FOLDER}/final_benchmark_results_no_PA.csv"
TRACKING_URI = "http://127.0.0.1:8080"

# Create the folder if it doesn't exist
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def export_all_metrics():
    print(f"Connecting to MLflow at {TRACKING_URI}...")
    mlflow.set_tracking_uri(TRACKING_URI)
    
    # Get all Experiments
    experiments = mlflow.search_experiments()
    print(f"Found {len(experiments)} experiments.")
    
    all_runs = []
    
    for exp in experiments:
        print(f"Scanning: {exp.name}...")
        
        # Fetch runs for this experiment
        df = mlflow.search_runs(
            experiment_ids=[exp.experiment_id], 
            output_format="pandas"
        )
        
        # Skip empty experiments
        if df.empty:
            continue
            
        # Clean up columns
        # We keep only useful columns to keep the CSV readable
        # Identify interesting columns
        cols_to_keep = [
            'run_id', 
            'status', 
            'start_time'
        ]
        
        # Add all metrics and params dynamically
        cols_to_keep += [c for c in df.columns if c.startswith('metrics.')]
        cols_to_keep += [c for c in df.columns if c.startswith('params.')]

        target_tags = [
            'tags.model_type', 
            'tags.processing_lib',
            'tags.dataset',
            'tags.use_pa'

        ]
        cols_to_keep += [c for c in df.columns if c in target_tags]
        
        # Filter the DataFrame
        valid_cols = [c for c in cols_to_keep if c in df.columns]
        df_clean = df[valid_cols].copy()
        
        # Add Experiment Name for context
        df_clean.insert(0, 'experiment_name', exp.name)
        
        all_runs.append(df_clean)

    # Concatenate and Save
    if all_runs:
        final_df = pd.concat(all_runs, ignore_index=True)
    
        # Rename columns to remove prefixes for cleaner CSV headers
        final_df.columns = [c.replace('metrics.', '').replace('params.', '').replace('tags.', '') 
                            for c in final_df.columns]
        
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nExported {len(final_df)} runs to '{OUTPUT_FILE}'")
    else:
        print("\nNo runs found to export.")

if __name__ == "__main__":
    export_all_metrics()