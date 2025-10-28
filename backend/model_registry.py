"""
Model Registry for AlMenn - Manages AI model versions, artifacts, and deployments

This module provides:
- Model versioning and metadata management
- Model artifact storage and retrieval
- Deployment status tracking
- Model performance monitoring
- Integration with human vetting pipeline
"""

import os
import json
import boto3
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from sqlalchemy.orm import Session
import models
from botocore.client import Config
import logging

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Central registry for managing AI model versions and deployments"""

    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'us-east-1'),
            config=Config(signature_version='s3v4')
        )
        self.bucket_name = os.getenv('MODEL_REGISTRY_BUCKET', 'almenn-model-registry')

    def register_model_version(self, db: Session, model_name: str, version: str,
                             model_type: str, artifacts_path: str,
                             performance_metrics: Dict[str, Any] = None,
                             training_metadata: Dict[str, Any] = None) -> str:
        """Register a new model version in the registry"""

        # Upload artifacts to S3
        s3_key = f"models/{model_name}/{version}/artifacts/"
        self._upload_artifacts(artifacts_path, s3_key)

        # Create registry entry
        model_version = models.ModelVersion(
            model_name=model_name,
            version=version,
            model_type=model_type,
            status="staging",
            s3_path=s3_key,
            performance_metrics=performance_metrics,
            training_metadata=training_metadata
        )

        db.add(model_version)
        db.commit()
        db.refresh(model_version)

        logger.info(f"Registered model version: {model_name} v{version}")
        return model_version.id

    def get_model_version(self, db: Session, model_version_id: str) -> Optional[models.ModelVersion]:
        """Retrieve model version details"""
        return db.query(models.ModelVersion).filter(models.ModelVersion.id == model_version_id).first()

    def list_model_versions(self, db: Session, model_name: str = None,
                          status: str = None) -> List[models.ModelVersion]:
        """List model versions with optional filtering"""
        query = db.query(models.ModelVersion)

        if model_name:
            query = query.filter(models.ModelVersion.model_name == model_name)
        if status:
            query = query.filter(models.ModelVersion.status == status)

        return query.order_by(models.ModelVersion.created_at.desc()).all()

    def update_model_status(self, db: Session, model_version_id: str,
                          status: str, vetted_by: str = None) -> bool:
        """Update model deployment status"""
        model_version = self.get_model_version(db, model_version_id)
        if not model_version:
            return False

        model_version.status = status

        if status == "production":
            model_version.deployed_at = datetime.utcnow()
        if vetted_by:
            model_version.vetted_by = vetted_by
            model_version.vetted_at = datetime.utcnow()

        db.commit()
        logger.info(f"Updated model {model_version_id} status to {status}")
        return True

    def deploy_model(self, db: Session, model_version_id: str,
                   deployment_target: str = "production") -> bool:
        """Deploy a model version to production"""
        model_version = self.get_model_version(db, model_version_id)
        if not model_version or model_version.status != "vetted":
            logger.error(f"Cannot deploy model {model_version_id}: not vetted")
            return False

        # Mark previous production versions as deprecated
        if deployment_target == "production":
            db.query(models.ModelVersion).filter(
                models.ModelVersion.model_name == model_version.model_name,
                models.ModelVersion.status == "production"
            ).update({"status": "deprecated"})

        # Deploy the new version
        success = self.update_model_status(db, model_version_id, "production")
        if success:
            logger.info(f"Successfully deployed model {model_version_id} to {deployment_target}")
        return success

    def get_active_model(self, db: Session, model_name: str,
                        model_type: str = None) -> Optional[models.ModelVersion]:
        """Get the currently active model version for a given model name"""
        query = db.query(models.ModelVersion).filter(
            models.ModelVersion.model_name == model_name,
            models.ModelVersion.status == "production"
        )

        if model_type:
            query = query.filter(models.ModelVersion.model_type == model_type)

        return query.order_by(models.ModelVersion.deployed_at.desc()).first()

    def record_model_metrics(self, db: Session, model_version_id: str,
                           metric_type: str, value: float,
                           metadata: Dict[str, Any] = None):
        """Record performance metrics for a model version"""
        metric = models.ModelMetrics(
            model_version_id=model_version_id,
            metric_type=metric_type,
            value=value,
            metadata=metadata
        )

        db.add(metric)
        db.commit()

    def get_model_metrics(self, db: Session, model_version_id: str,
                         metric_type: str = None, limit: int = 100) -> List[models.ModelMetrics]:
        """Retrieve metrics for a model version"""
        query = db.query(models.ModelMetrics).filter(
            models.ModelMetrics.model_version_id == model_version_id
        )

        if metric_type:
            query = query.filter(models.ModelMetrics.metric_type == metric_type)

        return query.order_by(models.ModelMetrics.collected_at.desc()).limit(limit).all()

    def compare_model_versions(self, db: Session, model_version_a: str,
                             model_version_b: str) -> Dict[str, Any]:
        """Compare performance metrics between two model versions"""
        metrics_a = self.get_model_metrics(db, model_version_a)
        metrics_b = self.get_model_metrics(db, model_version_b)

        comparison = {
            "version_a": model_version_a,
            "version_b": model_version_b,
            "metrics_comparison": {}
        }

        # Group metrics by type
        metrics_a_by_type = {}
        metrics_b_by_type = {}

        for metric in metrics_a:
            if metric.metric_type not in metrics_a_by_type:
                metrics_a_by_type[metric.metric_type] = []
            metrics_a_by_type[metric.metric_type].append(metric.value)

        for metric in metrics_b:
            if metric.metric_type not in metrics_b_by_type:
                metrics_b_by_type[metric.metric_type] = []
            metrics_b_by_type[metric.metric_type].append(metric.value)

        # Calculate averages and compare
        for metric_type in set(metrics_a_by_type.keys()) | set(metrics_b_by_type.keys()):
            avg_a = sum(metrics_a_by_type.get(metric_type, [0])) / len(metrics_a_by_type.get(metric_type, [1]))
            avg_b = sum(metrics_b_by_type.get(metric_type, [0])) / len(metrics_b_by_type.get(metric_type, [1]))

            comparison["metrics_comparison"][metric_type] = {
                "version_a_avg": avg_a,
                "version_b_avg": avg_b,
                "improvement": ((avg_b - avg_a) / avg_a * 100) if avg_a != 0 else 0
            }

        return comparison

    def _upload_artifacts(self, local_path: str, s3_key: str):
        """Upload model artifacts to S3"""
        if os.path.isfile(local_path):
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key + os.path.basename(local_path))
        elif os.path.isdir(local_path):
            for root, dirs, files in os.walk(local_path):
                for file in files:
                    local_file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_file_path, local_path)
                    s3_file_key = s3_key + relative_path
                    self.s3_client.upload_file(local_file_path, self.bucket_name, s3_file_key)

    def download_artifacts(self, s3_key: str, local_path: str):
        """Download model artifacts from S3"""
        try:
            # List objects with the prefix
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=s3_key)
            if 'Contents' not in response:
                return False

            # Download each file
            for obj in response['Contents']:
                file_key = obj['Key']
                relative_path = os.path.relpath(file_key, s3_key)
                local_file_path = os.path.join(local_path, relative_path)
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                self.s3_client.download_file(self.bucket_name, file_key, local_file_path)

            return True
        except Exception as e:
            logger.error(f"Failed to download artifacts: {e}")
            return False

# Global registry instance
model_registry = ModelRegistry()
