#!/usr/bin/env python
import mlflow
import pandas as pd
from tabulate import tabulate

# Set tracking URI to the SQLite database
mlflow.set_tracking_uri('sqlite:///mlflow-artifacts/mlflow.db')

# Get all experiments
experiments = mlflow.search_experiments()
print(f"Found {len(experiments)} experiments")

for experiment in experiments:
    print(f"\nExperiment: {experiment.name} (ID: {experiment.experiment_id})")
    
    # Get runs for this experiment
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    if runs.empty:
        print("  No runs found for this experiment")
        continue
    
    # Get metrics columns
    metric_columns = [col for col in runs.columns if col.startswith('metrics.')]
    
    if not metric_columns:
        print("  No metrics found for this experiment")
        continue
    
    # Select important metrics
    important_metrics = [col for col in metric_columns if any(m in col for m in 
                        ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'])]
    
    if important_metrics:
        print("\nImportant Metrics:")
        metrics_df = runs[important_metrics]
        print(tabulate(metrics_df, headers='keys', tablefmt='psql', showindex=False))
    
    # Show parameters for the best run (by accuracy)
    if 'metrics.test_accuracy' in runs.columns and runs['metrics.test_accuracy'].max() > 0:
        best_run = runs.loc[runs['metrics.test_accuracy'].idxmax()]
        run_id = best_run['run_id']
        
        print(f"\nBest Run (ID: {run_id}):")
        param_columns = [col for col in runs.columns if col.startswith('params.')]
        if param_columns:
            print("\nParameters:")
            params_df = pd.DataFrame([best_run[param_columns].to_dict()])
            print(tabulate(params_df, headers='keys', tablefmt='psql', showindex=False))
    
print("\nTo view complete results with a UI, run:")
print("uv run mlflow server --backend-store-uri sqlite:///mlflow-artifacts/mlflow.db --default-artifact-root file:./mlflow-artifacts/mlruns")
print("Then open http://localhost:5000 in your browser")