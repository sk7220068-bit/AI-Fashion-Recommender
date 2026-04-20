import mlflow
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ExperimentTracker:
    """
    Utility class for managing MLflow experiment tracking.
    """
    def __init__(self, experiment_name: str = "AI_Fashion_Detection"):
        self.experiment_name = experiment_name
        self._setup_mlflow()

    def _setup_mlflow(self):
        """Initializes MLflow tracking."""
        try:
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"MLflow Experiment initialized: {self.experiment_name}")
        except Exception as e:
            logger.warning(f"Could not initialize MLflow: {e}")

    def start_run(self, run_name: str = None):
        """Starts a new MLflow run."""
        return mlflow.start_run(run_name=run_name)

    def log_params(self, params: Dict[str, Any]):
        """Logs multiple parameters to MLflow."""
        if mlflow.active_run():
            mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, Any], step: int = None):
        """Logs multiple metrics to MLflow."""
        if mlflow.active_run():
            mlflow.log_metrics(metrics, step=step)

    def log_model(self, model: Any, artifact_path: str):
        """Logs a model artifact to MLflow."""
        if mlflow.active_run():
            # For YOLO/Ultralytics, we usually log the file or use their built-in integration
            mlflow.log_artifact(model, artifact_path)

def get_tracker(experiment_name: str = "AI_Fashion_Detection"):
    return ExperimentTracker(experiment_name)
