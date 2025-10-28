"""
A/B Testing Framework for AlMenn - Manages model experiments and traffic splitting

This module provides:
- Experiment configuration and management
- Traffic splitting between model versions
- Performance metrics collection
- Statistical analysis of experiment results
- Automated winner determination
"""

import os
import json
import random
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
import models
import logging
from scipy import stats
import numpy as np

logger = logging.getLogger(__name__)

class ABTestingFramework:
    """A/B testing framework for comparing model versions"""

    def __init__(self):
        self.random_seed = int(os.getenv('AB_TEST_SEED', '42'))

    def create_experiment(self, db: Session, name: str, description: str,
                         model_a_version: str, model_b_version: str,
                         traffic_split: float = 0.5, target_metric: str = "satisfaction",
                         created_by: str = None) -> str:
        """Create a new A/B test experiment"""

        # Validate model versions exist
        model_a = db.query(models.ModelVersion).filter(models.ModelVersion.id == model_a_version).first()
        model_b = db.query(models.ModelVersion).filter(models.ModelVersion.id == model_b_version).first()

        if not model_a or not model_b:
            raise ValueError("Invalid model version IDs")

        if model_a.model_name != model_b.model_name:
            raise ValueError("Model versions must be of the same model")

        # Create experiment
        experiment = models.ABTest(
            name=name,
            description=description,
            model_a_version=model_a_version,
            model_b_version=model_b_version,
            traffic_split=traffic_split,
            target_metric=target_metric,
            created_by=created_by
        )

        db.add(experiment)
        db.commit()
        db.refresh(experiment)

        logger.info(f"Created A/B test experiment: {name}")
        return experiment.id

    def assign_user_to_variant(self, user_id: str, experiment_id: str) -> str:
        """Assign a user to either variant A or B using consistent hashing"""

        # Create consistent hash for user-experiment pair
        hash_input = f"{user_id}:{experiment_id}:{self.random_seed}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)

        # Use hash to determine variant (deterministic for same user-experiment pair)
        if hash_value % 100 < (self.get_experiment_traffic_split(experiment_id) * 100):
            return "A"
        else:
            return "B"

    def get_experiment_traffic_split(self, experiment_id: str) -> float:
        """Get traffic split for an experiment (cached for performance)"""
        # In production, this would be cached in Redis
        # For now, return default 0.5
        return 0.5

    def record_experiment_metric(self, db: Session, experiment_id: str,
                               user_id: str, variant: str, metric_value: float,
                               metadata: Dict[str, Any] = None):
        """Record a metric observation for an experiment"""

        # Store in experiment results (simplified - in production use time-series DB)
        experiment = db.query(models.ABTest).filter(models.ABTest.id == experiment_id).first()
        if not experiment:
            return

        current_results = experiment.results or {"observations": []}

        observation = {
            "user_id": user_id,
            "variant": variant,
            "metric_value": metric_value,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata
        }

        current_results["observations"].append(observation)
        experiment.results = current_results
        db.commit()

    def analyze_experiment_results(self, db: Session, experiment_id: str) -> Dict[str, Any]:
        """Analyze experiment results and determine statistical significance"""

        experiment = db.query(models.ABTest).filter(models.ABTest.id == experiment_id).first()
        if not experiment or not experiment.results:
            return {"error": "No experiment data available"}

        observations = experiment.results.get("observations", [])

        # Separate observations by variant
        variant_a_values = [obs["metric_value"] for obs in observations if obs["variant"] == "A"]
        variant_b_values = [obs["metric_value"] for obs in observations if obs["variant"] == "B"]

        if len(variant_a_values) < 10 or len(variant_b_values) < 10:
            return {"error": "Insufficient data for statistical analysis"}

        # Calculate basic statistics
        analysis = {
            "experiment_id": experiment_id,
            "variant_a_count": len(variant_a_values),
            "variant_b_count": len(variant_b_values),
            "variant_a_mean": np.mean(variant_a_values),
            "variant_b_mean": np.mean(variant_b_values),
            "variant_a_std": np.std(variant_a_values),
            "variant_b_std": np.std(variant_b_values)
        }

        # Perform t-test for statistical significance
        try:
            t_stat, p_value = stats.ttest_ind(variant_a_values, variant_b_values)
            analysis["t_statistic"] = t_stat
            analysis["p_value"] = p_value
            analysis["statistically_significant"] = p_value < 0.05

            # Determine winner
            if analysis["statistically_significant"]:
                if analysis["variant_b_mean"] > analysis["variant_a_mean"]:
                    analysis["winner"] = "B"
                    analysis["improvement"] = ((analysis["variant_b_mean"] - analysis["variant_a_mean"]) / analysis["variant_a_mean"]) * 100
                else:
                    analysis["winner"] = "A"
                    analysis["improvement"] = ((analysis["variant_a_mean"] - analysis["variant_b_mean"]) / analysis["variant_b_mean"]) * 100
            else:
                analysis["winner"] = "inconclusive"

        except Exception as e:
            logger.error(f"Error in statistical analysis: {e}")
            analysis["error"] = "Statistical analysis failed"

        return analysis

    def stop_experiment(self, db: Session, experiment_id: str, winner: str = None) -> bool:
        """Stop an experiment and optionally declare a winner"""

        experiment = db.query(models.ABTest).filter(models.ABTest.id == experiment_id).first()
        if not experiment:
            return False

        experiment.status = "completed"
        experiment.end_date = datetime.utcnow()

        if winner:
            if winner == "A":
                experiment.winner = experiment.model_a_version
            elif winner == "B":
                experiment.winner = experiment.model_b_version

        db.commit()
        logger.info(f"Stopped experiment {experiment_id}, winner: {winner}")
        return True

    def get_active_experiments(self, db: Session, model_name: str = None) -> List[models.ABTest]:
        """Get all active experiments, optionally filtered by model"""

        query = db.query(models.ABTest).filter(models.ABTest.status == "active")

        if model_name:
            # Join with model versions to filter by model name
            query = query.join(models.ModelVersion, models.ABTest.model_a_version == models.ModelVersion.id).filter(
                models.ModelVersion.model_name == model_name
            )

        return query.all()

    def get_user_variant_for_experiment(self, db: Session, user_id: str,
                                     experiment_id: str) -> Optional[str]:
        """Get which variant a user was assigned to in an experiment"""

        experiment = db.query(models.ABTest).filter(models.ABTest.id == experiment_id).first()
        if not experiment:
            return None

        return self.assign_user_to_variant(user_id, experiment_id)

    def get_experiment_metrics_summary(self, db: Session, experiment_id: str) -> Dict[str, Any]:
        """Get summary metrics for an experiment"""

        experiment = db.query(models.ABTest).filter(models.ABTest.id == experiment_id).first()
        if not experiment or not experiment.results:
            return {}

        observations = experiment.results.get("observations", [])

        # Calculate metrics by variant
        metrics = {"A": {"count": 0, "sum": 0, "values": []},
                  "B": {"count": 0, "sum": 0, "values": []}}

        for obs in observations:
            variant = obs["variant"]
            value = obs["metric_value"]
            metrics[variant]["count"] += 1
            metrics[variant]["sum"] += value
            metrics[variant]["values"].append(value)

        # Calculate averages
        for variant in metrics:
            if metrics[variant]["count"] > 0:
                metrics[variant]["average"] = metrics[variant]["sum"] / metrics[variant]["count"]
            else:
                metrics[variant]["average"] = 0

        return {
            "experiment_id": experiment_id,
            "total_observations": len(observations),
            "metrics": metrics,
            "last_updated": experiment.results.get("last_updated")
        }

# Global A/B testing instance
ab_testing = ABTestingFramework()
