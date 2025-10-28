"""
Feedback Processor for AlMenn - PII-safe feedback loop with consent management

This module provides:
- PII sanitization and anonymization
- Explicit user consent management
- Feedback aggregation for model improvement
- NO auto-training on raw user data without consent
- Privacy-preserving feedback processing
"""

import os
import re
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
import models
import logging

logger = logging.getLogger(__name__)

class FeedbackProcessor:
    """PII-safe feedback processing system"""

    def __init__(self):
        # PII detection patterns
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            'address': r'\b\d+\s+[A-Za-z0-9\s,.-]+\b',  # Basic address pattern
            'name': r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'  # Basic name pattern
        }

        # Consent management
        self.consent_required_actions = [
            "model_training",
            "data_retention",
            "analytics_sharing"
        ]

    def sanitize_pii(self, text: str, aggressive: bool = False) -> Tuple[str, bool, List[str]]:
        """
        Sanitize PII from text

        Returns:
            Tuple of (sanitized_text, is_fully_sanitized, detected_pii_types)
        """
        if not text:
            return "", True, []

        sanitized_text = text
        detected_pii = []
        fully_sanitized = True

        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                detected_pii.append(pii_type)
                if aggressive:
                    # Replace with generic placeholders
                    replacement = f"[REDACTED_{pii_type.upper()}]"
                    sanitized_text = re.sub(pattern, replacement, sanitized_text, flags=re.IGNORECASE)
                else:
                    # Mark as not fully sanitized
                    fully_sanitized = False

        return sanitized_text, fully_sanitized, detected_pii

    def anonymize_user_data(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize user data for feedback processing"""

        anonymized = user_data.copy()

        # Hash sensitive identifiers
        if 'user_id' in anonymized:
            anonymized['user_hash'] = hashlib.sha256(anonymized['user_id'].encode()).hexdigest()[:16]
            del anonymized['user_id']

        if 'email' in anonymized:
            anonymized['email_domain'] = anonymized['email'].split('@')[-1] if '@' in anonymized['email'] else 'unknown'
            del anonymized['email']

        # Remove or generalize location data
        if 'ip_address' in anonymized:
            # Convert IP to rough geographic area (very basic)
            ip_parts = anonymized['ip_address'].split('.')
            if len(ip_parts) == 4:
                anonymized['ip_region'] = f"{ip_parts[0]}.{ip_parts[1]}.x.x"
            del anonymized['ip_address']

        return anonymized

    def process_user_feedback(self, db: Session, user_id: str, feedback_data: Dict[str, Any]) -> str:
        """Process user feedback with PII sanitization and consent checks"""

        feedback_type = feedback_data.get('type', 'general')

        # Check consent for training-related feedback
        consent_given = feedback_data.get('consent_given', False)
        if feedback_type in ['correction', 'training_suggestion'] and not consent_given:
            raise ValueError("Explicit consent required for training-related feedback")

        # Sanitize content based on feedback type
        sanitized_content = {}
        detected_pii = []

        if 'correction' in feedback_data:
            correction_text, is_sanitized, pii_types = self.sanitize_pii(feedback_data['correction'])
            sanitized_content['user_correction'] = correction_text
            detected_pii.extend(pii_types)
            if not is_sanitized:
                logger.warning(f"PII detected in correction from user {user_id}: {pii_types}")

        if 'flagged_content' in feedback_data:
            flagged_text, is_sanitized, pii_types = self.sanitize_pii(feedback_data['flagged_content'], aggressive=True)
            sanitized_content['flagged_content'] = flagged_text
            detected_pii.extend(pii_types)

        # Anonymize metadata
        anonymized_metadata = self.anonymize_user_data(feedback_data.get('metadata', {}))

        # Add PII detection info to metadata
        if detected_pii:
            anonymized_metadata['pii_detected'] = list(set(detected_pii))
            anonymized_metadata['pii_sanitization_performed'] = True

        # Create feedback record
        feedback = models.UserFeedback(
            user_id=user_id,
            session_id=feedback_data.get('session_id'),
            chat_id=feedback_data.get('chat_id'),
            model_version_id=feedback_data.get('model_version_id'),
            feedback_type=feedback_type,
            rating=feedback_data.get('rating'),
            consent_given=consent_given,
            pii_sanitized=len(detected_pii) == 0 or sanitized_content.get('user_correction') != feedback_data.get('correction'),
            metadata=anonymized_metadata,
            **sanitized_content
        )

        db.add(feedback)
        db.commit()
        db.refresh(feedback)

        logger.info(f"Processed {feedback_type} feedback from user {user_id}, consent: {consent_given}")
        return feedback.id

    def get_consent_status(self, db: Session, user_id: str) -> Dict[str, Any]:
        """Get user's consent status for different data uses"""

        # Check recent feedback for consent
        recent_feedback = db.query(models.UserFeedback).filter(
            models.UserFeedback.user_id == user_id
        ).order_by(models.UserFeedback.created_at.desc()).limit(10).all()

        consent_summary = {
            'training_consent_given': False,
            'last_consent_check': None,
            'consent_history': []
        }

        for feedback in recent_feedback:
            if feedback.consent_given:
                consent_summary['training_consent_given'] = True
                if not consent_summary['last_consent_check'] or feedback.created_at > consent_summary['last_consent_check']:
                    consent_summary['last_consent_check'] = feedback.created_at

            consent_summary['consent_history'].append({
                'feedback_type': feedback.feedback_type,
                'consent_given': feedback.consent_given,
                'timestamp': feedback.created_at.isoformat()
            })

        return consent_summary

    def request_user_consent(self, db: Session, user_id: str, consent_type: str,
                           purpose: str, retention_period_days: int = 365) -> str:
        """Create a consent request record"""

        # In production, this would trigger a user-facing consent flow
        consent_request = {
            'user_id': user_id,
            'consent_type': consent_type,
            'purpose': purpose,
            'retention_period_days': retention_period_days,
            'requested_at': datetime.utcnow().isoformat(),
            'status': 'pending'
        }

        # Store in user feedback table as a special type
        feedback = models.UserFeedback(
            user_id=user_id,
            feedback_type="consent_request",
            consent_given=False,  # Will be updated when user responds
            metadata=consent_request
        )

        db.add(feedback)
        db.commit()
        db.refresh(feedback)

        logger.info(f"Consent request created for user {user_id}, type: {consent_type}")
        return feedback.id

    def aggregate_feedback_for_training(self, db: Session, consent_only: bool = True,
                                      min_ratings: int = 10) -> List[Dict[str, Any]]:
        """Aggregate feedback data for model training (only with consent)"""

        query = db.query(models.UserFeedback).filter(
            models.UserFeedback.feedback_type.in_(['correction', 'training_suggestion']),
            models.UserFeedback.pii_sanitized == True
        )

        if consent_only:
            query = query.filter(models.UserFeedback.consent_given == True)

        feedback_items = query.all()

        # Group by model version and aggregate
        aggregated_feedback = {}

        for feedback in feedback_items:
            model_version = feedback.model_version_id or 'unknown'

            if model_version not in aggregated_feedback:
                aggregated_feedback[model_version] = {
                    'model_version': model_version,
                    'corrections': [],
                    'ratings': [],
                    'total_feedback': 0
                }

            if feedback.user_correction:
                aggregated_feedback[model_version]['corrections'].append({
                    'correction': feedback.user_correction,
                    'user_hash': hashlib.sha256(feedback.user_id.encode()).hexdigest()[:16],
                    'timestamp': feedback.created_at.isoformat()
                })

            if feedback.rating:
                aggregated_feedback[model_version]['ratings'].append(feedback.rating)

            aggregated_feedback[model_version]['total_feedback'] += 1

        # Filter out models with insufficient data
        filtered_feedback = []
        for model_data in aggregated_feedback.values():
            if len(model_data['ratings']) >= min_ratings or len(model_data['corrections']) >= 5:
                # Calculate average rating
                if model_data['ratings']:
                    model_data['average_rating'] = sum(model_data['ratings']) / len(model_data['ratings'])
                else:
                    model_data['average_rating'] = 0

                filtered_feedback.append(model_data)

        logger.info(f"Aggregated feedback for {len(filtered_feedback)} model versions")
        return filtered_feedback

    def generate_privacy_report(self, db: Session, user_id: str) -> Dict[str, Any]:
        """Generate privacy report for user data handling"""

        # Get all feedback for user
        feedback_history = db.query(models.UserFeedback).filter(
            models.UserFeedback.user_id == user_id
        ).order_by(models.UserFeedback.created_at.desc()).all()

        # Analyze data handling
        privacy_stats = {
            'total_feedback_items': len(feedback_history),
            'consent_given_count': len([f for f in feedback_history if f.consent_given]),
            'pii_sanitized_count': len([f for f in feedback_history if f.pii_sanitized]),
            'data_used_for_training': len([f for f in feedback_history if f.consent_given and f.feedback_type in ['correction', 'training_suggestion']]),
            'data_retention_period_days': 365,  # Configurable
            'last_activity': feedback_history[0].created_at.isoformat() if feedback_history else None
        }

        # PII detection summary
        pii_incidents = []
        for feedback in feedback_history:
            if feedback.metadata and feedback.metadata.get('pii_detected'):
                pii_incidents.append({
                    'timestamp': feedback.created_at.isoformat(),
                    'pii_types': feedback.metadata['pii_detected'],
                    'sanitized': feedback.pii_sanitized
                })

        return {
            'user_id': user_id,
            'privacy_stats': privacy_stats,
            'pii_incidents': pii_incidents,
            'consent_history': [
                {
                    'feedback_type': f.feedback_type,
                    'consent_given': f.consent_given,
                    'timestamp': f.created_at.isoformat()
                } for f in feedback_history
            ],
            'generated_at': datetime.utcnow().isoformat()
        }

    def purge_expired_data(self, db: Session, retention_days: int = 365):
        """Purge old feedback data that has exceeded retention period"""

        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)

        # Only purge data without explicit consent
        expired_feedback = db.query(models.UserFeedback).filter(
            models.UserFeedback.created_at < cutoff_date,
            models.UserFeedback.consent_given == False
        ).all()

        purged_count = len(expired_feedback)
        for feedback in expired_feedback:
            db.delete(feedback)

        db.commit()

        logger.info(f"Purged {purged_count} expired feedback items without consent")
        return purged_count

# Global feedback processor instance
feedback_processor = FeedbackProcessor()
