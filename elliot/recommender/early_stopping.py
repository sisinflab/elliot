from typing import List, Tuple, Dict, Optional
import logging as pylog

from elliot.namespace import EarlyStoppingConfig
from elliot.utils import logging, split_metric


class EarlyStopping:
    """The EarlyStopping class implements an early stopping mechanism for monitoring training metrics.

    This class helps to terminate training when a monitored metric stops improving,
    potentially avoiding overfitting and saving computational resources.

    Supported early stopping strategies:

    - `no_improvement`: Terminates training when the metric stops improving (always active).
    - `min_delta`: Terminates training when the metric stops improving by a specified amount.
    - `rel_delta`: Terminates training when the metric stops improving by a relative amount.
    - `baseline`: Terminates training when the metric reaches a baseline value.

    Args:
        early_stopping_config (EarlyStoppingConfig, optional): Configuration object containing
            early stopping parameters. Defaults to None.

    To configure the early stopping, include the appropriate
    settings in the configuration file using the pattern shown below.

    .. code:: yaml

      evaluation:
        simple_metrics: [nDCG, Recall]
      models:
        BPRMF:
          early_stopping:
            monitor: loss|nDCG|Recall
            patience: 3
            mode: min|max|auto
            min_delta: 0.01
            rel_delta: 0.05
            baseline: 0.01
            verbose: True|False
    """

    def __init__(self, early_stopping_config: Optional[EarlyStoppingConfig] = None):
        self.logger = logging.get_logger(self.__class__.__name__, pylog.DEBUG)

        # Do not activate early stopping without any configuration
        if early_stopping_config is None:
            self.active = False

        # Activate early stopping with the provided configuration
        else:
            monitor = early_stopping_config.monitor
            self.mode = early_stopping_config.mode

            # Automatically set mode to 'min' or 'max' based on the selected metric
            if self.mode in (None, "auto"):
                self.mode = "min" if monitor == "loss" else "max"

            if monitor == "loss":
                self.metric, self.metric_k = "", None
            else:
                self.metric, self.metric_k = split_metric(monitor)

            # Other parameters
            self.patience = early_stopping_config.patience

            self.min_delta = early_stopping_config.min_delta
            self.rel_delta = early_stopping_config.rel_delta
            self.baseline = early_stopping_config.baseline

            self.verbose = early_stopping_config.verbose

            self.active = True

    def stop(
        self,
        losses: List[float],
        results: List[dict]
    ) -> Tuple[bool, List[List[str]]]:
        """Evaluate stopping criteria based on observed metric values or losses.

        Args:
            losses (List[float]): List of loss values observed during training.
            results (List[dict]): List of dictionaries containing validation results
                and metrics.

        Returns:
            Tuple[bool, List[List[str]]]: A tuple where the first element indicates whether
                the stopping condition is met (True or False), while the second element is
                a list of triggered conditions if stopping is met (empty otherwise).
        """
        # If early stopping is not active, return False immediately
        if not self.active:
            return False, []

        # Pick the observed metric values or losses
        if not self.metric:
            observed = losses[:]
        else:
            observed = [
                r[self.metric_k]["val_results"][self.metric]
                for r in results
            ]

        # If there are not enough observations, return False immediately
        if len(observed) <= self.patience:
            return False, []

        # Keep only the last 'patience' observations
        observed = observed[-(self.patience + 1):]

        # Reverse the list if the mode is 'min'
        if self.mode == "min":
            observed = list(reversed(observed))

        # Check conditions for each pair of observations
        checks = []
        triggered_conditions = []
        for a, b in zip(observed[1:], observed):
            conds = self.check_conditions(a, b)

            pair_stop = any(conds.values())
            checks.append(pair_stop)

            triggered_conditions.append(
                [name for name, v in conds.items() if v]
            )

            if self.verbose:
                self.logger.info(f"Analyzed pair ({a:.5f}, {b:.5f}) -> {conds}")

        if self.verbose:
            self.logger.info(f"Check List: {checks}")

        return True if all(checks) else False, triggered_conditions

    def check_conditions(self, obs_0: float, obs_1: float) -> Dict[str, bool]:
        """Check various conditions based on the provided observations and thresholds.

        Args:
            obs_0 (float): First observation value used in condition checks.
            obs_1 (float): Second observation value used in condition checks.

        Returns:
            Dict[str, bool]: A dictionary containing boolean flags for each condition:

            - "no_improvement": True if obs_1 is greater than obs_0.
            - "min_delta": Present if self.min_delta is not None; True if the difference
              (obs_0 - obs_1) is less than or equal to self.min_delta.
            - "rel_delta": Present if self.rel_delta is not None; True if the difference
              (obs_0 - obs_1) is less than or equal to a fraction (obs_0 * self.rel_delta).
            - "baseline": Present if self.baseline is not None; evaluates to True if both
              obs_0 and obs_1 satisfy the baseline condition depending on the mode ("min" or "max").
        """
        conditions = {"no_improvement": obs_1 > obs_0}

        if self.min_delta is not None:
            conditions["min_delta"] = (obs_0 - obs_1) <= self.min_delta

        if self.rel_delta is not None:
            conditions["rel_delta"] = (obs_0 - obs_1) <= obs_0 * self.rel_delta

        if self.baseline is not None:
            if self.mode == "min":
                conditions["baseline"] = (obs_0 >= self.baseline) and (obs_1 >= self.baseline)
            else:
                conditions["baseline"] = (obs_0 <= self.baseline) and (obs_1 <= self.baseline)

        return conditions
