#!/usr/bin/env python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri('sqlite:///mlflow-artifacts/mlflow.db')

# Get experiments
experiments = mlflow.search_experiments()
print(f"Found {len(experiments)} experiments")

for experiment in experiments:
    print(f"\nExperiment: {experiment.name}")
    
    # Get all runs for this experiment
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    if len(runs) > 0:
        # Show the metrics for each run
        for i, (index, run) in enumerate(runs.iterrows()):
            print(f"\nRun {i+1}:")
            
            # Show metrics
            metrics = {k.replace('metrics.', ''): v for k, v in run.items() 
                      if k.startswith('metrics.') and v > 0}
            
            if metrics:
                print("  Metrics:")
                for metric, value in metrics.items():
                    print(f"    {metric}: {value:.4f}")
            
            # Show parameters
            params = {k.replace('params.', ''): v for k, v in run.items() 
                     if k.startswith('params.')}
            
            if params:
                print("  Parameters:")
                for param, value in params.items():
                    print(f"    {param}: {value}")
    else:
        print("  No runs found")