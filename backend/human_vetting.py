"""
Human-in-Loop Vetting Pipeline for AlMenn - Ensures model safety before deployment

This module provides:
- Automated pre-vetting checks
- Human review workflow management
- Approval/rejection tracking
- Integration with model registry
- Safety and quality scoring
"""

import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from sqlalchemy.orm import Session
import models
import logging

logger = logging.getLogger(__name__)

class HumanVettingPipeline:
    """Human-in-loop vetting system for model deployments"""

    def __init__(self):
        self.review_criteria = {
            "safety": [
                "No harmful content generation",
                "PII protection maintained",
                "Bias mitigation implemented",
                "Content moderation effective",
                "Emergency stop mechanisms present"
            ],
            "quality": [
                "Response accuracy verified",
                "Context understanding adequate",
                "Educational value maintained",
                "Response coherence good",
                "Performance metrics acceptable"
            ],
            "compliance": [
                "Data privacy regulations followed",
                "Training data consent verified",
                "Model documentation complete",
                "Audit trail maintained",
                "Rollback plan documented"
            ]
        }

    def submit_for_review(self, db: Session, model_version_id: str,
                         reviewer_id: str) -> str:
        """Submit a model version for human review"""

        # Check if model version exists and is in staging
        model_version = db.query(models.ModelVersion).filter(
            models.ModelVersion.id == model_version_id
        ).first()

        if not model_version or model_version.status != "staging":
            raise ValueError("Model version not eligible for review")

        # Create review record
        review = models.HumanReview(
            model_version_id=model_version_id,
            reviewer_id=reviewer_id,
            review_criteria=self.review_criteria
        )

        db.add(review)
        db.commit()
        db.refresh(review)

        logger.info(f"Submitted model {model_version_id} for human review")
        return review.id

    def perform_automated_checks(self, db: Session, model_version_id: str) -> Dict[str, Any]:
        """Run automated pre-vetting checks"""

        model_version = db.query(models.ModelVersion).filter(
            models.ModelVersion.id == model_version_id
        ).first()

        if not model_version:
            return {"error": "Model version not found"}

        checks_results = {
            "model_version_id": model_version_id,
            "checks_passed": True,
            "automated_checks": {},
            "recommendations": []
        }

        # Check 1: Performance metrics thresholds
        performance = model_version.performance_metrics or {}
        if performance.get("accuracy", 0) < 0.8:
            checks_results["automated_checks"]["accuracy_threshold"] = False
            checks_results["recommendations"].append("Model accuracy below 80% threshold")
            checks_results["checks_passed"] = False
        else:
            checks_results["automated_checks"]["accuracy_threshold"] = True

        # Check 2: Training data consent verification
        training_metadata = model_version.training_metadata or {}
        if not training_metadata.get("consent_verified", False):
            checks_results["automated_checks"]["consent_verified"] = False
            checks_results["recommendations"].append("Training data consent not verified")
            checks_results["checks_passed"] = False
        else:
            checks_results["automated_checks"]["consent_verified"] = True

        # Check 3: Safety metrics
        if performance.get("toxicity_score", 1.0) > 0.1:
            checks_results["automated_checks"]["toxicity_check"] = False
            checks_results["recommendations"].append("High toxicity score detected")
            checks_results["checks_passed"] = False
        else:
            checks_results["automated_checks"]["toxicity_check"] = True

        # Check 4: Model size and performance
        if performance.get("latency_ms", 0) > 5000:  # 5 second threshold
            checks_results["automated_checks"]["latency_check"] = False
            checks_results["recommendations"].append("Response latency too high")
            checks_results["checks_passed"] = False
        else:
            checks_results["automated_checks"]["latency_check"] = True

        return checks_results

    def submit_review_decision(self, db: Session, review_id: str,
                             decision: str, reviewer_id: str,
                             safety_score: int = None, quality_score: int = None,
                             comments: str = None) -> bool:
        """Submit human review decision"""

        review = db.query(models.HumanReview).filter(models.HumanReview.id == review_id).first()
        if not review:
            return False

        if review.reviewer_id != reviewer_id:
            raise ValueError("Unauthorized reviewer")

        if decision not in ["approved", "rejected"]:
            raise ValueError("Invalid decision")

        review.status = decision
        review.safety_score = safety_score
        review.quality_score = quality_score
        review.comments = comments
        review.reviewed_at = datetime.utcnow()

        # Update model version status based on decision
        model_version = review.model_version
        if decision == "approved":
            model_version.status = "vetted"
            model_version.vetted_by = reviewer_id
            model_version.vetted_at = datetime.utcnow()
        else:
            model_version.status = "rejected"

        db.commit()

        logger.info(f"Review {review_id} decision: {decision} by {reviewer_id}")
        return True

    def get_pending_reviews(self, db: Session, reviewer_id: str = None) -> List[models.HumanReview]:
        """Get pending reviews, optionally filtered by reviewer"""

        query = db.query(models.HumanReview).filter(models.HumanReview.status == "pending")

        if reviewer_id:
            query = query.filter(models.HumanReview.reviewer_id == reviewer_id)

        return query.join(models.ModelVersion).order_by(models.ModelVersion.created_at.desc()).all()

    def get_review_details(self, db: Session, review_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed review information"""

        review = db.query(models.HumanReview).filter(models.HumanReview.id == review_id).first()
        if not review:
            return None

        # Run automated checks for context
        automated_checks = self.perform_automated_checks(db, review.model_version_id)

        return {
            "review_id": review.id,
            "model_version": {
                "id": review.model_version.id,
                "name": review.model_version.model_name,
                "version": review.model_version.version,
                "type": review.model_version.model_type,
                "performance_metrics": review.model_version.performance_metrics,
                "training_metadata": review.model_version.training_metadata
            },
            "reviewer_id": review.reviewer_id,
            "status": review.status,
            "review_criteria": review.review_criteria,
            "automated_checks": automated_checks,
            "safety_score": review.safety_score,
            "quality_score": review.quality_score,
            "comments": review.comments,
            "created_at": review.created_at.isoformat(),
            "reviewed_at": review.reviewed_at.isoformat() if review.reviewed_at else None
        }

    def get_review_history(self, db: Session, model_version_id: str = None,
                          reviewer_id: str = None) -> List[models.HumanReview]:
        """Get review history with optional filtering"""

        query = db.query(models.HumanReview)

        if model_version_id:
            query = query.filter(models.HumanReview.model_version_id == model_version_id)
        if reviewer_id:
            query = query.filter(models.HumanReview.reviewer_id == reviewer_id)

        return query.order_by(models.HumanReview.created_at.desc()).all()

    def escalate_review(self, db: Session, review_id: str, reason: str) -> bool:
        """Escalate a review to senior reviewers"""

        review = db.query(models.HumanReview).filter(models.HumanReview.id == review_id).first()
        if not review:
            return False

        # Add escalation metadata
        current_comments = review.comments or ""
        review.comments = f"{current_comments}\n\n[ESCALATED] {reason}".strip()

        # In production, this would trigger notifications to senior reviewers
        logger.warning(f"Review {review_id} escalated: {reason}")

        db.commit()
        return True

    def get_review_statistics(self, db: Session) -> Dict[str, Any]:
        """Get statistics about the review process"""

        total_reviews = db.query(models.HumanReview).count()
        pending_reviews = db.query(models.HumanReview).filter(models.HumanReview.status == "pending").count()
        approved_reviews = db.query(models.HumanReview).filter(models.HumanReview.status == "approved").count()
        rejected_reviews = db.query(models.HumanReview).filter(models.HumanReview.status == "rejected").count()

        # Average review time
        completed_reviews = db.query(models.HumanReview).filter(
            models.HumanReview.status.in_(["approved", "rejected"])
        ).all()

        if completed_reviews:
            total_time = sum(
                (review.reviewed_at - review.created_at).total_seconds()
                for review in completed_reviews if review.reviewed_at
            )
            avg_review_time_hours = total_time / len(completed_reviews) / 3600
        else:
            avg_review_time_hours = 0

        return {
            "total_reviews": total_reviews,
            "pending_reviews": pending_reviews,
            "approved_reviews": approved_reviews,
            "rejected_reviews": rejected_reviews,
            "approval_rate": approved_reviews / max(total_reviews, 1),
            "average_review_time_hours": avg_review_time_hours
        }

# Global vetting pipeline instance
vetting_pipeline = HumanVettingPipeline()
