import time
from locust import HttpUser, task, between
from locust_plugins.csvreader import CSVReader
import json
import uuid


class AIChatUser(HttpUser):
    """Load testing user for AI Chat application"""

    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks

    def on_start(self):
        """Setup before starting tasks"""
        # Register/login to get auth token
        self.auth_token = None
        self.session_id = None
        self.user_id = str(uuid.uuid4())

        # Attempt to authenticate
        self.authenticate()

    def authenticate(self):
        """Authenticate user and get token"""
        try:
            # Send magic link
            response = self.client.post("/auth/send_magic_link", json={
                "email": f"loadtest-{self.user_id}@example.com"
            })
            if response.status_code != 200:
                return

            # For load testing, we'll create a direct token
            # In real scenario, this would come from email
            from main import create_access_token
            session_id = str(uuid.uuid4())
            token = create_access_token({
                "sub": self.user_id,
                "session_id": session_id
            })

            self.auth_token = token
            self.session_id = session_id

        except Exception as e:
            print(f"Authentication failed: {e}")

    @task(1)
    def health_check(self):
        """Health check endpoint"""
        self.client.get("/health")

    @task(2)
    def get_wallet(self):
        """Get user wallet/balance"""
        if self.auth_token:
            headers = {"Authorization": f"Bearer {self.auth_token}"}
            self.client.get("/wallet", headers=headers)

    @task(3)
    def start_ai_session(self):
        """Start AI session"""
        if self.auth_token:
            headers = {"Authorization": f"Bearer {self.auth_token}"}
            response = self.client.post("/ai/session/start", headers=headers)

            if response.status_code == 200:
                data = response.json()
                self.ai_session_id = data.get("session_id")

    @task(4)
    def ai_query(self):
        """Make AI query"""
        if self.auth_token and hasattr(self, 'ai_session_id'):
            headers = {"Authorization": f"Bearer {self.auth_token}"}

            # Random queries for load testing
            queries = [
                "What is machine learning?",
                "Explain quantum physics",
                "How does photosynthesis work?",
                "What are the benefits of exercise?",
                "Tell me about ancient Rome",
                "How to solve quadratic equations?",
                "What is the meaning of life?",
                "Explain blockchain technology",
                "How do computers work?",
                "What is climate change?"
            ]

            import random
            query_text = random.choice(queries)

            data = {
                "session_id": self.ai_session_id,
                "input_text": query_text,
                "mode": "rag"
            }

            self.client.post("/ai/query", json=data, headers=headers)

    @task(1)
    def get_metrics(self):
        """Get Prometheus metrics"""
        self.client.get("/metrics")

    @task(1)
    def extend_session(self):
        """Extend AI session"""
        if self.auth_token:
            headers = {"Authorization": f"Bearer {self.auth_token}"}
            self.client.post("/ai/session/extend", headers=headers)


class AdminUser(HttpUser):
    """Load testing for admin operations"""

    wait_time = between(5, 10)  # Less frequent admin operations

    def on_start(self):
        """Setup admin user"""
        # Admin token would be set differently in real scenario
        self.admin_token = "admin-token-placeholder"

    @task(1)
    def get_moderation_queue(self):
        """Check moderation queue"""
        if self.admin_token:
            headers = {"Authorization": f"Bearer {self.admin_token}"}
            self.client.get("/admin/moderation/queue", headers=headers)

    @task(1)
    def moderate_content(self):
        """Moderate content"""
        if self.admin_token:
            headers = {"Authorization": f"Bearer {self.admin_token}"}
            data = {"message_id": f"msg-{uuid.uuid4()}"}
            self.client.post("/admin/message/delete", json=data, headers=headers)


# Load testing configuration
# Run with: locust -f locustfile.py --host=http://localhost:8000
# Or for distributed testing: locust -f locustfile.py --master --host=http://localhost:8000

# Performance targets:
# - AI queries: < 2 second response time for 95th percentile
# - Concurrent users: Support 100+ simultaneous AI chat sessions
# - Error rate: < 1% under normal load
# - Memory usage: Stable under sustained load

# Test scenarios:
# 1. Ramp up: Gradually increase users to find breaking point
# 2. Spike test: Sudden load increase to test autoscaling
# 3. Endurance test: Sustained load over extended period
# 4. Stress test: Load beyond normal capacity to find limits
