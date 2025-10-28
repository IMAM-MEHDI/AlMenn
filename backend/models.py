from sqlalchemy import Column, String, Integer, Boolean, DateTime, ForeignKey, Text, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import uuid

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True, nullable=False)
    public_profile = Column(JSON, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    is_admin = Column(Boolean, default=False)
    upi_id = Column(String, nullable=True)  # UPI ID for payments/refunds
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login_at = Column(DateTime(timezone=True), nullable=True)

class Session(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    token = Column(String, unique=True, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Coins(Base):
    __tablename__ = "coins"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False, unique=True)
    balance = Column(Integer, default=0)
    free_daily_coins_last_grant_date = Column(DateTime(timezone=True), nullable=True)
    purchased_coins = Column(Integer, default=0)  # Track purchased vs free coins separately
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    type = Column(String, nullable=False)  # 'purchase', 'usage', 'free_grant'
    amount = Column(Integer, nullable=False)  # positive for credits, negative for debits
    description = Column(String, nullable=True)
    gateway_transaction_id = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class AI_Session(Base):
    __tablename__ = "ai_sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False)
    is_active = Column(Boolean, default=True)
    minutes_used = Column(Float, default=0.0)  # Track actual usage time
    total_duration_minutes = Column(Integer, default=60)  # Default 60 minutes per session

class AI_Chat(Base):
    __tablename__ = "ai_chats"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("ai_sessions.id"), nullable=False)
    user_message = Column(Text, nullable=False)
    ai_response = Column(Text, nullable=False)
    attachments = Column(JSON, nullable=True)  # List of file IDs
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class File(Base):
    __tablename__ = "files"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)  # S3 key or local path
    file_size = Column(Integer, nullable=False)
    mime_type = Column(String, nullable=False)
    extracted_text = Column(Text, nullable=True)
    embedding_id = Column(String, nullable=True)  # Vector DB ID
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class GlobalChatMessage(Base):
    __tablename__ = "global_chat_messages"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    message = Column(Text, nullable=False)
    attachments = Column(JSON, nullable=True)  # List of file IDs
    is_moderated = Column(Boolean, default=False)
    moderated_by = Column(String, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class AdminUserChat(Base):
    __tablename__ = "admin_user_chats"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    admin_id = Column(String, ForeignKey("users.id"), nullable=False)
    message = Column(Text, nullable=False)  # Encrypted message content
    is_from_admin = Column(Boolean, default=False)
    attachments = Column(JSON, nullable=True)  # List of file IDs
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class EmbeddingsIndex(Base):
    __tablename__ = "embeddings_index"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    file_id = Column(String, ForeignKey("files.id"), nullable=True)
    chat_id = Column(String, ForeignKey("ai_chats.id"), nullable=True)
    embedding_vector = Column(JSON, nullable=False)  # Store as JSON array
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

# ML Ops & Safety Tables
class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    model_name = Column(String, nullable=False, index=True)
    version = Column(String, nullable=False)  # Semantic version (e.g., "1.2.3")
    model_type = Column(String, nullable=False)  # "embedding", "llm", "moderation"
    status = Column(String, nullable=False, default="staging")  # "staging", "production", "deprecated"
    s3_path = Column(String, nullable=True)  # S3 path to model artifacts
    performance_metrics = Column(JSON, nullable=True)  # Accuracy, latency, etc.
    training_metadata = Column(JSON, nullable=True)  # Training data info, hyperparameters
    vetted_by = Column(String, ForeignKey("users.id"), nullable=True)
    vetted_at = Column(DateTime(timezone=True), nullable=True)
    deployed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class ABTest(Base):
    __tablename__ = "ab_tests"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False, unique=True)
    description = Column(Text, nullable=True)
    model_a_version = Column(String, ForeignKey("model_versions.id"), nullable=False)
    model_b_version = Column(String, ForeignKey("model_versions.id"), nullable=False)
    traffic_split = Column(Float, nullable=False, default=0.5)  # Percentage for model A (0.0-1.0)
    status = Column(String, nullable=False, default="active")  # "active", "completed", "stopped"
    start_date = Column(DateTime(timezone=True), server_default=func.now())
    end_date = Column(DateTime(timezone=True), nullable=True)
    target_metric = Column(String, nullable=False)  # "satisfaction", "accuracy", "latency"
    results = Column(JSON, nullable=True)  # Test results and statistics
    winner = Column(String, ForeignKey("model_versions.id"), nullable=True)
    created_by = Column(String, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class HumanReview(Base):
    __tablename__ = "human_reviews"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    model_version_id = Column(String, ForeignKey("model_versions.id"), nullable=False)
    reviewer_id = Column(String, ForeignKey("users.id"), nullable=False)
    status = Column(String, nullable=False, default="pending")  # "pending", "approved", "rejected"
    review_criteria = Column(JSON, nullable=True)  # Checklist of criteria to review
    comments = Column(Text, nullable=True)
    safety_score = Column(Integer, nullable=True)  # 1-10 safety rating
    quality_score = Column(Integer, nullable=True)  # 1-10 quality rating
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class UserFeedback(Base):
    __tablename__ = "user_feedback"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    session_id = Column(String, ForeignKey("ai_sessions.id"), nullable=True)
    chat_id = Column(String, ForeignKey("ai_chats.id"), nullable=True)
    model_version_id = Column(String, ForeignKey("model_versions.id"), nullable=True)
    feedback_type = Column(String, nullable=False)  # "rating", "flag_bad_response", "correction"
    rating = Column(Integer, nullable=True)  # 1-5 satisfaction rating
    flagged_content = Column(Text, nullable=True)  # Anonymized bad response content
    user_correction = Column(Text, nullable=True)  # User's suggested correction
    consent_given = Column(Boolean, nullable=False, default=False)  # Explicit consent for training use
    pii_sanitized = Column(Boolean, nullable=False, default=True)  # PII has been removed
    metadata_json = Column(JSON, nullable=True)  # Additional context (anonymized)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class SafetyIncident(Base):
    __tablename__ = "safety_incidents"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    incident_type = Column(String, nullable=False)  # "harmful_content", "pii_leak", "abuse"
    severity = Column(String, nullable=False)  # "low", "medium", "high", "critical"
    description = Column(Text, nullable=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)  # Anonymized if needed
    session_id = Column(String, ForeignKey("ai_sessions.id"), nullable=True)
    model_version_id = Column(String, ForeignKey("model_versions.id"), nullable=True)
    content_snippet = Column(Text, nullable=True)  # Sanitized content snippet
    actions_taken = Column(JSON, nullable=True)  # Response actions
    resolved = Column(Boolean, nullable=False, default=False)
    resolved_by = Column(String, ForeignKey("users.id"), nullable=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class ModelMetrics(Base):
    __tablename__ = "model_metrics"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    model_version_id = Column(String, ForeignKey("model_versions.id"), nullable=False)
    metric_type = Column(String, nullable=False)  # "token_usage", "response_quality", "latency"
    value = Column(Float, nullable=False)
    metadata_json = Column(JSON, nullable=True)  # Additional metric context
    collected_at = Column(DateTime(timezone=True), server_default=func.now())

# Support & Analytics Tables
class SupportTicket(Base):
    __tablename__ = "support_tickets"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    subject = Column(String, nullable=False)
    category = Column(String, nullable=False)  # "technical", "billing", "abuse", "general"
    description = Column(Text, nullable=False)
    status = Column(String, nullable=False, default="open")  # "open", "in_progress", "resolved", "closed"
    priority = Column(String, nullable=False, default="medium")  # "low", "medium", "high", "urgent"
    assigned_to = Column(String, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_updated = Column(DateTime(timezone=True), nullable=True, onupdate=func.now())
    resolution_notes = Column(Text, nullable=True)

class PaymentOrder(Base):
    __tablename__ = "payment_orders"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    gateway = Column(String, nullable=False, default="phonepe")  # "phonepe", "razorpay"
    gateway_order_id = Column(String, nullable=True)  # PhonePe transaction ID
    merchant_order_id = Column(String, nullable=False, unique=True)  # Our internal order ID
    amount = Column(Integer, nullable=False)  # Amount in paisa (PhonePe uses paisa)
    currency = Column(String, nullable=False, default="INR")
    status = Column(String, nullable=False, default="created")  # "created", "pending", "completed", "failed", "cancelled"
    package_id = Column(Integer, nullable=True)  # Reference to PAYMENT_PACKAGES
    coins_to_credit = Column(Integer, nullable=True)  # Coins to add on successful payment
    gateway_response = Column(JSON, nullable=True)  # Store full gateway response
    payment_instrument = Column(JSON, nullable=True)  # Payment method details
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class PaymentGatewaySettings(Base):
    __tablename__ = "payment_gateway_settings"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    gateway = Column(String, nullable=False, default="phonepe")  # "phonepe", "razorpay", etc.
    merchant_id = Column(String, nullable=True)
    salt_key = Column(String, nullable=True)
    salt_index = Column(String, nullable=True)
    qr_code_url = Column(String, nullable=True)  # URL to QR code image
    is_active = Column(Boolean, default=False)
    updated_by = Column(String, ForeignKey("users.id"), nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())
