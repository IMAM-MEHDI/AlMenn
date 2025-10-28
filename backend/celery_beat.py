from celery import Celery
from celery.schedules import crontab
import os
from dotenv import load_dotenv

load_dotenv()

celery_app = Celery(
    "almenn",
    broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://localhost:6379/0")
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

# Import tasks
from tasks import grant_daily_free_coins, delete_old_chats

celery_app.conf.beat_schedule = {
    # Daily job: grant free +4 coins (cron at 00:00 UTC)
    'grant-daily-free-coins': {
        'task': 'tasks.grant_daily_free_coins',
        'schedule': crontab(hour=0, minute=0),  # Every day at 00:00 UTC
    },
    # Job: delete AI_Chats older than 7 days (run every hour)
    'delete-old-chats': {
        'task': 'tasks.delete_old_chats',
        'schedule': crontab(minute=0),  # Every hour
    },
}

celery_app.conf.beat_schedule_filename = "celerybeat-schedule"
