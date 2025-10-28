"""
Safety Guardrails for AlMenn - Content safety and abuse prevention

This module provides:
- Content moderation with AI-based detection
- Rate limiting and abuse detection
- Emergency stop mechanisms
- Incident response and logging
- Integration with safety incident tracking
"""

import os
import re
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
import models
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class SafetyGuardrails:
    """Comprehensive safety system for AI interactions"""

    def __init__(self):
        # Content safety patterns
        self.harmful_patterns = {
            'violence': [
                r'\b(kill|murder|attack|harm|hurt)\b.*\b(person|people|someone|anyone)\b',
                r'\b(weapon|gun|knife|bomb)\b.*\b(use|make|build)\b',
                r'\b(self-harm|suicide)\b'
            ],
            'hate_speech': [
                r'\b(hate|racist|sexist|homophobic|transphobic)\b',
                r'\b(nazi|supremacist|terrorist)\b',
                r'\b(slur|offensive|derogatory)\b.*\b(word|term|language)\b'
            ],
            'illegal_activities': [
                r'\b(drug|illegal|crime|fraud|scam)\b.*\b(make|sell|buy|trade)\b',
                r'\b(hack|exploit|breach|phishing)\b.*\b(system|account|password)\b',
                r'\b(forge|fake|falsify)\b.*\b(document|id|identity)\b'
            ],
            'explicit_content': [
                r'\b(porn|sex|naked|nude|erotic)\b',
                r'\b(explicit|graphic|disturbing)\b.*\b(content|material)\b'
            ]
        }

        # Rate limiting
        self.rate_limits = {
            'queries_per_minute': 10,
            'queries_per_hour': 100,
            'sessions_per_day': 5
        }

        # User activity tracking (in production, use Redis)
        self.user_activity = defaultdict(lambda: {
            'queries_last_minute': [],
            'queries_last_hour': [],
            'sessions_today': [],
            'last_incident': None
        })

        # Emergency stop flag
        self.emergency_stop = False
        self.emergency_stop_reason = None

    def check_content_safety(self, content: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Check content for safety violations"""

        safety_result = {
            'is_safe': True,
            'violations': [],
            'severity_score': 0.0,
            'recommendations': []
        }

        content_lower = content.lower()

        # Check against harmful patterns
        for category, patterns in self.harmful_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_lower, re.IGNORECASE):
                    safety_result['violations'].append({
                        'category': category,
                        'pattern': pattern,
                        'severity': self._get_severity_score(category)
                    })
                    safety_result['severity_score'] = max(safety_result['severity_score'],
                                                        self._get_severity_score(category))

        # Context-aware checks
        if context:
            if context.get('user_age', 18) < 18 and 'explicit_content' in [v['category'] for v in safety_result['violations']]:
                safety_result['severity_score'] += 0.5

        # Determine if content is safe
        safety_result['is_safe'] = safety_result['severity_score'] < 0.7

        # Generate recommendations
        if not safety_result['is_safe']:
            safety_result['recommendations'] = self._generate_safety_recommendations(safety_result['violations'])

        return safety_result

    def check_rate_limits(self, user_id: str, action_type: str = 'query') -> Dict[str, Any]:
        """Check if user has exceeded rate limits"""

        now = datetime.utcnow()
        user_data = self.user_activity[user_id]

        # Clean old entries
        self._clean_old_activity(user_data, now)

        rate_check = {
            'allowed': True,
            'violations': [],
            'retry_after_seconds': 0
        }

        # Check queries per minute
        if len(user_data['queries_last_minute']) >= self.rate_limits['queries_per_minute']:
            rate_check['violations'].append('queries_per_minute')
            rate_check['allowed'] = False
            rate_check['retry_after_seconds'] = 60

        # Check queries per hour
        if len(user_data['queries_last_hour']) >= self.rate_limits['queries_per_hour']:
            rate_check['violations'].append('queries_per_hour')
            rate_check['allowed'] = False
            rate_check['retry_after_seconds'] = 3600

        # Check sessions per day
        if action_type == 'session_start' and len(user_data['sessions_today']) >= self.rate_limits['sessions_per_day']:
            rate_check['violations'].append('sessions_per_day')
            rate_check['allowed'] = False
            rate_check['retry_after_seconds'] = 86400 - (now - user_data['sessions_today'][0]).seconds

        # Record the action if allowed
        if rate_check['allowed']:
            if action_type == 'query':
                user_data['queries_last_minute'].append(now)
                user_data['queries_last_hour'].append(now)
            elif action_type == 'session_start':
                user_data['sessions_today'].append(now)

        return rate_check

    def detect_abuse_patterns(self, db: Session, user_id: str,
                            recent_interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect potential abuse patterns"""

        abuse_detection = {
            'suspicious_activity': False,
            'patterns_detected': [],
            'risk_score': 0.0,
            'recommendations': []
        }

        # Check for repetitive queries
        if len(recent_interactions) > 5:
            queries = [interaction.get('query', '') for interaction in recent_interactions]
            if len(set(queries)) / len(queries) < 0.3:  # Less than 30% unique queries
                abuse_detection['patterns_detected'].append('repetitive_queries')
                abuse_detection['risk_score'] += 0.3

        # Check for rapid-fire queries
        if len(recent_interactions) > 10:
            timestamps = [datetime.fromisoformat(interaction.get('timestamp', datetime.utcnow().isoformat()))
                         for interaction in recent_interactions[-10:]]
            time_diffs = [(timestamps[i] - timestamps[i-1]).seconds for i in range(1, len(timestamps))]
            avg_time_between_queries = sum(time_diffs) / len(time_diffs)

            if avg_time_between_queries < 5:  # Less than 5 seconds between queries
                abuse_detection['patterns_detected'].append('rapid_fire_queries')
                abuse_detection['risk_score'] += 0.4

        # Check for consistently flagged content
        flagged_count = len([i for i in recent_interactions if i.get('flagged', False)])
        if len(recent_interactions) > 0 and flagged_count / len(recent_interactions) > 0.5:
            abuse_detection['patterns_detected'].append('consistently_flagged')
            abuse_detection['risk_score'] += 0.5

        abuse_detection['suspicious_activity'] = abuse_detection['risk_score'] > 0.5

        if abuse_detection['suspicious_activity']:
            abuse_detection['recommendations'] = [
                'Implement temporary rate limiting',
                'Require additional verification',
                'Monitor user activity closely'
            ]

        return abuse_detection

    def log_safety_incident(self, db: Session, incident_data: Dict[str, Any]) -> str:
        """Log a safety incident for tracking and analysis"""

        incident = models.SafetyIncident(
            incident_type=incident_data.get('incident_type', 'unknown'),
            severity=incident_data.get('severity', 'low'),
            description=incident_data.get('description', ''),
            user_id=incident_data.get('user_id'),
            session_id=incident_data.get('session_id'),
            model_version_id=incident_data.get('model_version_id'),
            content_snippet=incident_data.get('content_snippet', ''),
            actions_taken=incident_data.get('actions_taken', [])
        )

        db.add(incident)
        db.commit()
        db.refresh(incident)

        logger.warning(f"Safety incident logged: {incident.incident_type} (severity: {incident.severity})")
        return incident.id

    def activate_emergency_stop(self, reason: str, triggered_by: str):
        """Activate emergency stop mechanism"""

        self.emergency_stop = True
        self.emergency_stop_reason = reason

        logger.critical(f"EMERGENCY STOP ACTIVATED: {reason} (by {triggered_by})")

        # In production, this would:
        # - Stop all model inferences
        # - Alert administrators
        # - Log the incident
        # - Trigger rollback procedures

    def deactivate_emergency_stop(self, triggered_by: str):
        """Deactivate emergency stop mechanism"""

        self.emergency_stop = False
        self.emergency_stop_reason = None

        logger.info(f"Emergency stop deactivated by {triggered_by}")

    def get_emergency_status(self) -> Dict[str, Any]:
        """Get current emergency stop status"""

        return {
            'emergency_stop_active': self.emergency_stop,
            'reason': self.emergency_stop_reason,
            'activated_at': datetime.utcnow().isoformat() if self.emergency_stop else None
        }

    def moderate_global_chat_message(self, db: Session, message: str,
                                   user_id: str) -> Tuple[bool, str]:
        """Moderate a global chat message"""

        safety_check = self.check_content_safety(message)

        if not safety_check['is_safe']:
            # Log incident
            self.log_safety_incident(db, {
                'incident_type': 'harmful_content',
                'severity': 'medium' if safety_check['severity_score'] > 0.8 else 'low',
                'description': f'Harmful content detected in global chat',
                'user_id': user_id,
                'content_snippet': message[:200] + '...' if len(message) > 200 else message
            })

            return False, f"Message blocked due to safety concerns: {', '.join([v['category'] for v in safety_check['violations']])}"

        return True, "Message approved"

    def get_safety_statistics(self, db: Session, days: int = 7) -> Dict[str, Any]:
        """Get safety statistics for monitoring"""

        since_date = datetime.utcnow() - timedelta(days=days)

        incidents = db.query(models.SafetyIncident).filter(
            models.SafetyIncident.created_at >= since_date
        ).all()

        stats = {
            'period_days': days,
            'total_incidents': len(incidents),
            'incidents_by_type': {},
            'incidents_by_severity': {},
            'resolved_incidents': len([i for i in incidents if i.resolved]),
            'average_resolution_time_hours': 0
        }

        # Group by type and severity
        for incident in incidents:
            stats['incidents_by_type'][incident.incident_type] = stats['incidents_by_type'].get(incident.incident_type, 0) + 1
            stats['incidents_by_severity'][incident.severity] = stats['incidents_by_severity'].get(incident.severity, 0) + 1

        # Calculate average resolution time
        resolved_incidents = [i for i in incidents if i.resolved and i.resolved_at]
        if resolved_incidents:
            total_resolution_time = sum(
                (i.resolved_at - i.created_at).total_seconds() / 3600
                for i in resolved_incidents
            )
            stats['average_resolution_time_hours'] = total_resolution_time / len(resolved_incidents)

        return stats

    def _get_severity_score(self, category: str) -> float:
        """Get severity score for a violation category"""

        severity_map = {
            'violence': 0.9,
            'hate_speech': 0.8,
            'illegal_activities': 0.7,
            'explicit_content': 0.5
        }

        return severity_map.get(category, 0.3)

    def _generate_safety_recommendations(self, violations: List[Dict[str, Any]]) -> List[str]:
        """Generate safety recommendations based on violations"""

        recommendations = []

        categories = set(v['category'] for v in violations)

        if 'violence' in categories:
            recommendations.extend([
                'Block content and alert moderators',
                'Consider temporary user suspension',
                'Review user interaction history'
            ])

        if 'hate_speech' in categories:
            recommendations.extend([
                'Implement content filtering',
                'Flag user for review',
                'Log incident for pattern analysis'
            ])

        if 'illegal_activities' in categories:
            recommendations.extend([
                'Block content immediately',
                'Report to appropriate authorities if necessary',
                'Permanent user ban consideration'
            ])

        if 'explicit_content' in categories:
            recommendations.extend([
                'Age-appropriate content filtering',
                'Content warnings for sensitive topics'
            ])

        return recommendations

    def _clean_old_activity(self, user_data: Dict[str, Any], now: datetime):
        """Clean old activity data to prevent memory bloat"""

        # Clean queries older than 1 hour
        user_data['queries_last_hour'] = [
            ts for ts in user_data['queries_last_hour']
            if (now - ts).seconds < 3600
        ]

        # Clean queries older than 1 minute
        user_data['queries_last_minute'] = [
            ts for ts in user_data['queries_last_minute']
            if (now - ts).seconds < 60
        ]

        # Clean sessions older than 24 hours
        user_data['sessions_today'] = [
            ts for ts in user_data['sessions_today']
            if (now - ts).days < 1
        ]

# Global safety guardrails instance
safety_guardrails = SafetyGuardrails()
