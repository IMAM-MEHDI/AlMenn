"""
Metrics Collector for AlMenn - Comprehensive AI model and system metrics

This module provides:
- Token usage tracking per user/session
- Bad response flagging and analysis
- User satisfaction score collection
- Model performance monitoring
- System health metrics
"""

import os
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
import models
import logging
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import psutil

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Comprehensive metrics collection system"""

    def __init__(self):
        # Prometheus metrics
        self.registry = CollectorRegistry()

        # Token usage metrics
        self.token_usage_total = Counter(
            'almenn_token_usage_total',
            'Total tokens used by users',
            ['user_id', 'model_version', 'operation_type'],
            registry=self.registry
        )

        self.token_usage_per_minute = Counter(
            'almenn_token_usage_per_minute',
            'Tokens used per minute',
            ['user_id'],
            registry=self.registry
        )

        # Response quality metrics
        self.response_quality_score = Histogram(
            'almenn_response_quality_score',
            'Distribution of response quality scores',
            ['model_version'],
            registry=self.registry
        )

        self.bad_responses_total = Counter(
            'almenn_bad_responses_total',
            'Total bad responses flagged',
            ['reason', 'model_version'],
            registry=self.registry
        )

        # User satisfaction metrics
        self.user_satisfaction_score = Histogram(
            'almenn_user_satisfaction_score',
            'Distribution of user satisfaction scores (1-5)',
            ['model_version'],
            registry=self.registry
        )

        # Model performance metrics
        self.model_inference_latency = Histogram(
            'almenn_model_inference_latency_seconds',
            'Model inference latency in seconds',
            ['model_version', 'operation_type'],
            registry=self.registry
        )

        self.model_inference_errors = Counter(
            'almenn_model_inference_errors_total',
            'Total model inference errors',
            ['model_version', 'error_type'],
            registry=self.registry
        )

        # System metrics
        self.active_sessions = Gauge(
            'almenn_active_sessions',
            'Number of active AI sessions',
            registry=self.registry
        )

        self.system_cpu_usage = Gauge(
            'almenn_system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )

        self.system_memory_usage = Gauge(
            'almenn_system_memory_usage_percent',
            'System memory usage percentage',
            registry=self.registry
        )

    def record_token_usage(self, user_id: str, model_version_id: str,
                          tokens_used: int, operation_type: str = "inference"):
        """Record token usage for a user interaction"""

        self.token_usage_total.labels(
            user_id=user_id,
            model_version=model_version_id,
            operation_type=operation_type
        ).inc(tokens_used)

        # Store in database for detailed analysis
        # This would be called from the main application

    def record_response_quality(self, model_version_id: str, quality_score: float):
        """Record response quality score (0.0 to 1.0)"""

        self.response_quality_score.labels(model_version=model_version_id).observe(quality_score)

    def flag_bad_response(self, db: Session, user_id: str, session_id: str,
                         chat_id: str, model_version_id: str,
                         flagged_content: str, reason: str):
        """Flag a bad response for analysis"""

        self.bad_responses_total.labels(reason=reason, model_version=model_version_id).inc()

        # Store detailed information in database
        feedback = models.UserFeedback(
            user_id=user_id,
            session_id=session_id,
            chat_id=chat_id,
            model_version_id=model_version_id,
            feedback_type="flag_bad_response",
            flagged_content=flagged_content,
            pii_sanitized=True,  # Assume content is sanitized before flagging
            metadata={"reason": reason, "flagged_at": datetime.utcnow().isoformat()}
        )

        db.add(feedback)
        db.commit()

        logger.info(f"Bad response flagged by user {user_id}, reason: {reason}")

    def record_user_satisfaction(self, db: Session, user_id: str, session_id: str,
                               chat_id: str, model_version_id: str, rating: int,
                               consent_given: bool = False):
        """Record user satisfaction rating (1-5)"""

        if not 1 <= rating <= 5:
            raise ValueError("Rating must be between 1 and 5")

        self.user_satisfaction_score.labels(model_version=model_version_id).observe(rating)

        # Store in database
        feedback = models.UserFeedback(
            user_id=user_id,
            session_id=session_id,
            chat_id=chat_id,
            model_version_id=model_version_id,
            feedback_type="rating",
            rating=rating,
            consent_given=consent_given,
            pii_sanitized=True
        )

        db.add(feedback)
        db.commit()

        logger.info(f"User satisfaction recorded: {rating}/5 by user {user_id}")

    def record_user_correction(self, db: Session, user_id: str, session_id: str,
                             chat_id: str, model_version_id: str,
                             user_correction: str, consent_given: bool = False):
        """Record user-provided correction for continual learning"""

        feedback = models.UserFeedback(
            user_id=user_id,
            session_id=session_id,
            chat_id=chat_id,
            model_version_id=model_version_id,
            feedback_type="correction",
            user_correction=user_correction,
            consent_given=consent_given,
            pii_sanitized=self._sanitize_pii(user_correction)  # Sanitize before storing
        )

        db.add(feedback)
        db.commit()

        logger.info(f"User correction recorded by user {user_id}")

    def record_model_inference(self, model_version_id: str, operation_type: str,
                             latency_seconds: float, success: bool = True,
                             error_type: str = None):
        """Record model inference metrics"""

        self.model_inference_latency.labels(
            model_version=model_version_id,
            operation_type=operation_type
        ).observe(latency_seconds)

        if not success and error_type:
            self.model_inference_errors.labels(
                model_version=model_version_id,
                error_type=error_type
            ).inc()

    def update_system_metrics(self):
        """Update system-level metrics"""

        # Active sessions (would need to be passed from main app)
        # self.active_sessions.set(active_session_count)

        # System resource usage
        self.system_cpu_usage.set(psutil.cpu_percent())
        self.system_memory_usage.set(psutil.virtual_memory().percent)

    def get_user_metrics_summary(self, db: Session, user_id: str,
                               days: int = 30) -> Dict[str, Any]:
        """Get comprehensive metrics summary for a user"""

        since_date = datetime.utcnow() - timedelta(days=days)

        # Token usage
        token_usage = db.query(
            models.ModelMetrics
        ).filter(
            models.ModelMetrics.metric_type == "token_usage",
            models.ModelMetrics.metadata.contains({"user_id": user_id}),
            models.ModelMetrics.collected_at >= since_date
        ).all()

        total_tokens = sum(metric.value for metric in token_usage)

        # User feedback
        feedback = db.query(models.UserFeedback).filter(
            models.UserFeedback.user_id == user_id,
            models.UserFeedback.created_at >= since_date
        ).all()

        ratings = [f.rating for f in feedback if f.rating is not None]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0

        bad_responses_flagged = len([f for f in feedback if f.feedback_type == "flag_bad_response"])
        corrections_provided = len([f for f in feedback if f.feedback_type == "correction"])

        return {
            "user_id": user_id,
            "period_days": days,
            "total_tokens_used": total_tokens,
            "average_rating": avg_rating,
            "total_ratings": len(ratings),
            "bad_responses_flagged": bad_responses_flagged,
            "corrections_provided": corrections_provided,
            "consent_given_count": len([f for f in feedback if f.consent_given])
        }

    def get_model_performance_summary(self, db: Session, model_version_id: str,
                                    days: int = 7) -> Dict[str, Any]:
        """Get performance summary for a model version"""

        since_date = datetime.utcnow() - timedelta(days=days)

        # Get metrics from database
        metrics = db.query(models.ModelMetrics).filter(
            models.ModelMetrics.model_version_id == model_version_id,
            models.ModelMetrics.collected_at >= since_date
        ).all()

        # Aggregate metrics
        metric_summary = {}
        for metric in metrics:
            if metric.metric_type not in metric_summary:
                metric_summary[metric.metric_type] = {
                    "count": 0,
                    "sum": 0,
                    "avg": 0,
                    "min": float('inf'),
                    "max": 0
                }

            metric_summary[metric.metric_type]["count"] += 1
            metric_summary[metric.metric_type]["sum"] += metric.value
            metric_summary[metric.metric_type]["min"] = min(metric_summary[metric.metric_type]["min"], metric.value)
            metric_summary[metric.metric_type]["max"] = max(metric_summary[metric.metric_type]["max"], metric.value)

        # Calculate averages
        for metric_type in metric_summary:
            if metric_summary[metric_type]["count"] > 0:
                metric_summary[metric_type]["avg"] = metric_summary[metric_type]["sum"] / metric_summary[metric_type]["count"]

        # Get user feedback for this model
        feedback = db.query(models.UserFeedback).filter(
            models.UserFeedback.model_version_id == model_version_id,
            models.UserFeedback.created_at >= since_date
        ).all()

        ratings = [f.rating for f in feedback if f.rating is not None]
        avg_user_rating = sum(ratings) / len(ratings) if ratings else 0

        bad_responses = len([f for f in feedback if f.feedback_type == "flag_bad_response"])

        return {
            "model_version_id": model_version_id,
            "period_days": days,
            "metrics": metric_summary,
            "user_feedback": {
                "total_ratings": len(ratings),
                "average_rating": avg_user_rating,
                "bad_responses_flagged": bad_responses
            }
        }

    def _sanitize_pii(self, text: str) -> bool:
        """Check if text has been sanitized of PII"""
        # Basic PII detection (in production, use more sophisticated detection)
        pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{16}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{10}\b'  # Phone number
        ]

        import re
        for pattern in pii_patterns:
            if re.search(pattern, text):
                return False
        return True

    def export_metrics_for_analysis(self, db: Session, start_date: datetime,
                                  end_date: datetime) -> Dict[str, Any]:
        """Export metrics data for external analysis"""

        # Get all metrics in date range
        metrics = db.query(models.ModelMetrics).filter(
            models.ModelMetrics.collected_at.between(start_date, end_date)
        ).all()

        # Get user feedback in date range
        feedback = db.query(models.UserFeedback).filter(
            models.UserFeedback.created_at.between(start_date, end_date)
        ).all()

        return {
            "export_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "metrics_data": [
                {
                    "model_version_id": m.model_version_id,
                    "metric_type": m.metric_type,
                    "value": m.value,
                    "timestamp": m.collected_at.isoformat(),
                    "metadata": m.metadata
                } for m in metrics
            ],
            "feedback_data": [
                {
                    "user_id": f.user_id,
                    "model_version_id": f.model_version_id,
                    "feedback_type": f.feedback_type,
                    "rating": f.rating,
                    "consent_given": f.consent_given,
                    "timestamp": f.created_at.isoformat()
                } for f in feedback
            ]
        }

# Global metrics collector instance
metrics_collector = MetricsCollector()
