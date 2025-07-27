import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Union

import mlflow
import structlog
from mlflow.entities import Run
from mlflow.exceptions import MlflowException

from vibe_data_science_2nd_try.config import MLFlowConfig

logger = structlog.get_logger()


def setup_mlflow(config: MLFlowConfig) -> None:
    """
    Set up MLflow with the provided configuration.
    
    Args:
        config: MLFlow configuration
    """
    logger.info("Setting up MLflow", 
                tracking_uri=config.tracking_uri, 
                experiment_name=config.experiment_name)
    
    mlflow.set_tracking_uri(config.tracking_uri)
    
    try:
        # Get or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(config.experiment_name)
            if experiment:
                logger.info("Using existing experiment", 
                            experiment_id=experiment.experiment_id)
            else:
                experiment_id = mlflow.create_experiment(config.experiment_name)
                logger.info("Created new experiment", experiment_id=experiment_id)
        except MlflowException as e:
            logger.error("Error getting/creating experiment", error=str(e))
            raise
        
    except Exception as e:
        logger.error("Failed to set up MLflow", error=str(e))
        raise


@contextmanager
def mlflow_run(
    run_name: str,
    params: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, str]] = None,
    log_artifacts: bool = True,
    artifact_path: Optional[Union[str, Path]] = None,
    nested: bool = False,
) -> Generator[Run, None, None]:
    """
    Context manager for an MLflow run.
    
    Args:
        run_name: Name for the run
        params: Parameters to log
        tags: Tags to set
        log_artifacts: Whether to log artifacts
        artifact_path: Path to artifacts
        nested: Whether this is a nested run
    
    Yields:
        The MLflow run object
    """
    current_run = mlflow.active_run()
    run_id = current_run.info.run_id if current_run else None
    
    logger_context = {"run_name": run_name}
    if run_id:
        logger_context["parent_run_id"] = run_id
    
    logger.info("Starting MLflow run", **logger_context)
    
    try:
        with mlflow.start_run(run_name=run_name, nested=nested) as run:
            run_id = run.info.run_id
            logger.info("MLflow run started", run_id=run_id)
            
            # Log parameters
            if params:
                logger.debug("Logging parameters", count=len(params))
                mlflow.log_params(params)
            
            # Set tags
            if tags:
                logger.debug("Setting tags", count=len(tags))
                mlflow.set_tags(tags)
            
            yield run
            
            # Log artifacts
            if log_artifacts and artifact_path:
                artifact_path_str = str(artifact_path)
                if os.path.exists(artifact_path_str):
                    logger.debug("Logging artifacts", path=artifact_path_str)
                    mlflow.log_artifacts(artifact_path_str)
    
    except Exception as e:
        logger.error("Error in MLflow run", 
                     run_name=run_name, 
                     run_id=run_id, 
                     error=str(e))
        raise
    
    logger.info("MLflow run completed", run_id=run_id)