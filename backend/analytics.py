"""
Analytics and Monitoring for AlMenn - Admin Dashboard Backend

This module provides:
- Daily active users tracking
- Coin purchase analytics
- AI session usage metrics
- Payment failure tracking
- System performance monitoring
- Support ticket management
- Admin dashboard data aggregation
"""

import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
import models
import logging

logger = logging.getLogger(__name__)

class AnalyticsService:
    """Comprehensive analytics service for admin dashboard"""

    def __init__(self):
        pass

    def get_daily_active_users(self, db: Session, days: int = 30) -> List[Dict[str, Any]]:
        """Get daily active users over the specified period"""

        since_date = datetime.utcnow() - timedelta(days=days)

        # Get daily active users from AI sessions (most reliable activity indicator)
        daily_stats = db.query(
            func.date(models.AI_Session.started_at).label('date'),
            func.count(func.distinct(models.AI_Session.user_id)).label('active_users')
        ).filter(
            models.AI_Session.started_at >= since_date
        ).group_by(
            func.date(models.AI_Session.started_at)
        ).order_by(
            func.date(models.AI_Session.started_at)
        ).all()

        return [
            {
                'date': stat.date.isoformat(),
                'active_users': stat.active_users
            } for stat in daily_stats
        ]

    def get_coin_purchase_analytics(self, db: Session, days: int = 30) -> Dict[str, Any]:
        """Get coin purchase analytics"""

        since_date = datetime.utcnow() - timedelta(days=days)

        # Total coin purchases
        total_purchases = db.query(
            func.sum(models.Transaction.amount).label('total_coins')
        ).filter(
            models.Transaction.type == 'purchase',
            models.Transaction.created_at >= since_date
        ).scalar() or 0

        # Purchase count and average
        purchase_stats = db.query(
            func.count(models.Transaction.id).label('purchase_count'),
            func.avg(models.Transaction.amount).label('avg_purchase')
        ).filter(
            models.Transaction.type == 'purchase',
            models.Transaction.created_at >= since_date
        ).first()

        # Daily purchase trends
        daily_purchases = db.query(
            func.date(models.Transaction.created_at).label('date'),
            func.sum(models.Transaction.amount).label('coins_purchased'),
            func.count(models.Transaction.id).label('transaction_count')
        ).filter(
            models.Transaction.type == 'purchase',
            models.Transaction.created_at >= since_date
        ).group_by(
            func.date(models.Transaction.created_at)
        ).order_by(
            func.date(models.Transaction.created_at)
        ).all()

        return {
            'total_coins_purchased': abs(total_purchases),  # Amount is negative for debits
            'total_purchase_transactions': purchase_stats.purchase_count or 0,
            'average_purchase_amount': abs(purchase_stats.avg_purchase or 0),
            'daily_trends': [
                {
                    'date': stat.date.isoformat(),
                    'coins_purchased': abs(stat.coins_purchased),
                    'transaction_count': stat.transaction_count
                } for stat in daily_purchases
            ]
        }

    def get_ai_usage_analytics(self, db: Session, days: int = 30) -> Dict[str, Any]:
        """Get AI session and usage analytics"""

        since_date = datetime.utcnow() - timedelta(days=days)

        # Total minutes consumed
        total_minutes = db.query(
            func.sum(models.AI_Session.minutes_used).label('total_minutes')
        ).filter(
            models.AI_Session.created_at >= since_date
        ).scalar() or 0

        # Session statistics
        session_stats = db.query(
            func.count(models.AI_Session.id).label('total_sessions'),
            func.avg(models.AI_Session.minutes_used).label('avg_session_length'),
            func.sum(models.AI_Session.total_duration_minutes).label('total_allocated_minutes')
        ).filter(
            models.AI_Session.created_at >= since_date
        ).first()

        # Daily usage trends
        daily_usage = db.query(
            func.date(models.AI_Session.started_at).label('date'),
            func.sum(models.AI_Session.minutes_used).label('minutes_used'),
            func.count(models.AI_Session.id).label('session_count')
        ).filter(
            models.AI_Session.started_at >= since_date
        ).group_by(
            func.date(models.AI_Session.started_at)
        ).order_by(
            func.date(models.AI_Session.started_at)
        ).all()

        return {
            'total_minutes_consumed': total_minutes,
            'total_sessions': session_stats.total_sessions or 0,
            'average_session_length': session_stats.avg_session_length or 0,
            'total_allocated_minutes': session_stats.total_allocated_minutes or 0,
            'daily_trends': [
                {
                    'date': stat.date.isoformat(),
                    'minutes_used': stat.minutes_used or 0,
                    'session_count': stat.session_count
                } for stat in daily_usage
            ]
        }

    def get_payment_failure_analytics(self, db: Session, days: int = 30) -> Dict[str, Any]:
        """Get payment failure analytics"""

        since_date = datetime.utcnow() - timedelta(days=days)

        # Failed payments (assuming failed transactions are marked in gateway_transaction_id or description)
        failed_payments = db.query(models.Transaction).filter(
            models.Transaction.type == 'purchase',
            models.Transaction.amount == 0,  # Failed payments might have 0 amount
            models.Transaction.created_at >= since_date
        ).all()

        # Group by failure reason (would need to be stored in metadata/description)
        failure_reasons = {}
        for payment in failed_payments:
            reason = getattr(payment, 'description', 'Unknown')
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

        return {
            'total_failed_payments': len(failed_payments),
            'failure_reasons': failure_reasons,
            'failure_rate': len(failed_payments) / max(1, len(db.query(models.Transaction).filter(
                models.Transaction.type == 'purchase',
                models.Transaction.created_at >= since_date
            ).all()))
        }

    def get_system_performance_metrics(self, db: Session, days: int = 7) -> Dict[str, Any]:
        """Get system performance metrics from stored metrics"""

        since_date = datetime.utcnow() - timedelta(days=days)

        # Get latency metrics
        latency_metrics = db.query(models.ModelMetrics).filter(
            models.ModelMetrics.metric_type == 'inference_latency',
            models.ModelMetrics.collected_at >= since_date
        ).all()

        if latency_metrics:
            latencies = [m.value for m in latency_metrics]
            avg_latency = sum(latencies) / len(latencies)
            p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        else:
            avg_latency = 0
            p95_latency = 0

        # Get error rates
        total_requests = db.query(models.ModelMetrics).filter(
            models.ModelMetrics.metric_type.in_(['inference_latency', 'inference_errors']),
            models.ModelMetrics.collected_at >= since_date
        ).count()

        error_count = db.query(models.ModelMetrics).filter(
            models.ModelMetrics.metric_type == 'inference_errors',
            models.ModelMetrics.collected_at >= since_date
        ).count()

        error_rate = error_count / max(1, total_requests)

        return {
            'average_latency_seconds': avg_latency,
            'p95_latency_seconds': p95_latency,
            'error_rate': error_rate,
            'total_requests': total_requests,
            'total_errors': error_count
        }

    def get_support_tickets(self, db: Session, status: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get support tickets for admin dashboard"""

        query = db.query(models.SupportTicket).order_by(models.SupportTicket.created_at.desc())

        if status:
            query = query.filter(models.SupportTicket.status == status)

        tickets = query.limit(limit).all()

        return [
            {
                'id': str(ticket.id),
                'user_id': str(ticket.user_id),
                'subject': ticket.subject,
                'category': ticket.category,
                'status': ticket.status,
                'priority': ticket.priority,
                'created_at': ticket.created_at.isoformat(),
                'last_updated': ticket.last_updated.isoformat() if ticket.last_updated else None,
                'assigned_to': str(ticket.assigned_to) if ticket.assigned_to else None
            } for ticket in tickets
        ]

    def get_abusive_content_reports(self, db: Session, days: int = 7) -> List[Dict[str, Any]]:
        """Get recent abusive content reports"""

        since_date = datetime.utcnow() - timedelta(days=days)

        # Get safety incidents
        incidents = db.query(models.SafetyIncident).filter(
            models.SafetyIncident.created_at >= since_date,
            models.SafetyIncident.incident_type.in_(['abuse', 'harmful_content'])
        ).order_by(models.SafetyIncident.created_at.desc()).all()

        return [
            {
                'id': str(incident.id),
                'incident_type': incident.incident_type,
                'severity': incident.severity,
                'description': incident.description,
                'user_id': str(incident.user_id) if incident.user_id else None,
                'content_snippet': incident.content_snippet,
                'resolved': incident.resolved,
                'created_at': incident.created_at.isoformat()
            } for incident in incidents
        ]

    def get_dashboard_summary(self, db: Session) -> Dict[str, Any]:
        """Get comprehensive dashboard summary"""

        today = datetime.utcnow().date()
        yesterday = today - timedelta(days=1)
        week_ago = today - timedelta(days=7)

        # Today's metrics
        today_active_users = db.query(
            func.count(func.distinct(models.AI_Session.user_id))
        ).filter(
            func.date(models.AI_Session.started_at) == today
        ).scalar() or 0

        today_sessions = db.query(models.AI_Session).filter(
            func.date(models.AI_Session.started_at) == today
        ).count()

        today_minutes = db.query(
            func.sum(models.AI_Session.minutes_used)
        ).filter(
            func.date(models.AI_Session.started_at) == today
        ).scalar() or 0

        today_purchases = db.query(
            func.sum(func.abs(models.Transaction.amount))
        ).filter(
            models.Transaction.type == 'purchase',
            func.date(models.Transaction.created_at) == today
        ).scalar() or 0

        # Yesterday's metrics for comparison
        yesterday_active_users = db.query(
            func.count(func.distinct(models.AI_Session.user_id))
        ).filter(
            func.date(models.AI_Session.started_at) == yesterday
        ).scalar() or 0

        yesterday_sessions = db.query(models.AI_Session).filter(
            func.date(models.AI_Session.started_at) == yesterday
        ).count()

        # Pending support tickets
        pending_tickets = db.query(models.SupportTicket).filter(
            models.SupportTicket.status == 'open'
        ).count()

        # Recent safety incidents
        recent_incidents = db.query(models.SafetyIncident).filter(
            models.SafetyIncident.created_at >= week_ago,
            models.SafetyIncident.resolved == False
        ).count()

        return {
            'today': {
                'active_users': today_active_users,
                'sessions': today_sessions,
                'minutes_used': today_minutes,
                'coins_purchased': today_purchases
            },
            'yesterday_comparison': {
                'active_users_change': today_active_users - yesterday_active_users,
                'sessions_change': today_sessions - yesterday_sessions
            },
            'pending_support_tickets': pending_tickets,
            'unresolved_safety_incidents': recent_incidents,
            'generated_at': datetime.utcnow().isoformat()
        }

    def create_support_ticket(self, db: Session, user_id: str, subject: str,
                            category: str, description: str, priority: str = 'medium') -> str:
        """Create a new support ticket"""

        ticket = models.SupportTicket(
            user_id=user_id,
            subject=subject,
            category=category,
            description=description,
            priority=priority,
            status='open'
        )

        db.add(ticket)
        db.commit()
        db.refresh(ticket)

        logger.info(f"Support ticket created by user {user_id}: {subject}")
        return str(ticket.id)

    def report_abusive_content(self, db: Session, user_id: str, content_type: str,
                             content_id: str, reason: str, content_snippet: str = None) -> str:
        """Report abusive content"""

        incident = models.SafetyIncident(
            incident_type='abuse',
            severity='medium',
            description=f"Abusive content reported: {reason}",
            user_id=user_id,
            content_snippet=content_snippet,
            actions_taken=[{
                'action': 'reported',
                'reported_by': user_id,
                'reason': reason,
                'content_type': content_type,
                'content_id': content_id,
                'timestamp': datetime.utcnow().isoformat()
            }]
        )

        db.add(incident)
        db.commit()
        db.refresh(incident)

        logger.info(f"Abusive content reported by user {user_id}: {reason}")
        return str(incident.id)

# Global analytics service instance
analytics_service = AnalyticsService()
