from typing import Any, Dict
import wandb


class WandBMetricOrchestrator:

    def __init__(self) -> None:
        """Class to handle pushing of metrics to currently initialised weights
        and biases run and storing them locally.
        """
        self.local_metrics = {}  # Store metrics locally
    
    def add_metric(*args, **kwargs):
        # For API compatibility (not used directly here)
        pass
    
    def update_metrics(self, metric_value_dict:Dict[str, Dict[str,Any]]):
        """Method for updating multiple metrics simultaneously

        Args:
            metric_value_dict (Dict[str, Dict[str, Any]]): A dictionary
            containing the relevant update values of the form:
            {*metric_name*:{"label": *value_label*, "value": *value_value*}}
        """
        # Log metrics to WandB
        wandb.log({
            metric:metric_value_dict[metric]["value"]
            for metric in metric_value_dict.keys()
        })

        # Store metrics locally
        for metric, details in metric_value_dict.items():
            if metric not in self.local_metrics:
                self.local_metrics[metric] = []
            self.local_metrics[metric].append(details["value"])

    def get_metric_values(self, metric_name: str):
        """Retrieve locally stored metrics."""
        return self.local_metrics.get(metric_name, [])
