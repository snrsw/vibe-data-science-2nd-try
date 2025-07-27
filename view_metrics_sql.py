#!/usr/bin/env python
import sqlite3
import pandas as pd

# Connect to the SQLite database
conn = sqlite3.connect('mlflow-artifacts/mlflow.db')

# Query to get experiments
experiments_query = """
SELECT experiment_id, name, artifact_location
FROM experiments
ORDER BY experiment_id
"""

# Get experiments
experiments = pd.read_sql_query(experiments_query, conn)
print(f"Found {len(experiments)} experiments:")
print(experiments[['experiment_id', 'name']])
print("\n")

# Get metrics for a specific experiment (example with experiment_id=0)
metrics_query = """
SELECT 
    r.run_uuid,
    r.experiment_id,
    m.key as metric_name,
    m.value as metric_value
FROM runs r
JOIN metrics m ON r.run_uuid = m.run_uuid
WHERE r.experiment_id = 0
"""

metrics = pd.read_sql_query(metrics_query, conn)
print("Metrics for Default experiment:")
if len(metrics) > 0:
    # Pivot the data to get metrics as columns
    metrics_pivot = metrics.pivot_table(
        index='run_uuid',
        columns='metric_name',
        values='metric_value'
    )
    print(metrics_pivot)
else:
    print("No metrics found")

# Get parameters for the same experiment
params_query = """
SELECT 
    r.run_uuid,
    p.key as param_name,
    p.value as param_value
FROM runs r
JOIN params p ON r.run_uuid = p.run_uuid
WHERE r.experiment_id = 0
LIMIT 10
"""

params = pd.read_sql_query(params_query, conn)
print("\nParameters (sample):")
if len(params) > 0:
    # Pivot the data to get parameters as columns
    params_pivot = params.pivot_table(
        index='run_uuid',
        columns='param_name',
        values='param_value',
        aggfunc='first'
    )
    print(params_pivot)
else:
    print("No parameters found")

# Close the connection
conn.close()