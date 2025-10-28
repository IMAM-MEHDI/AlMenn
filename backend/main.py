from fastapi import FastAPI, Depends, HTTPException, status, WebSocket, WebSocketDisconnect, UploadFile, File as FastAPIFile, Request
from fastapi.responses import Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, UUID4
from typing import Optional, List, Dict, Any
import uuid
import jwt
import datetime
import os
import json
import logging
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import create_engine
from contextlib import asynccontextmanager
import models
import tasks
from cryptography.fernet import Fernet
import base64
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import pyclamd
import analytics

# PhonePe integration
try:
    from phonepe import PhonePe
    phonepe_available = True
except ImportError:
    phonepe_available = False
    logger.warning("PhonePe SDK not available")

# Lazy import AI engine components
ai_engine = None
SessionManagerAgent = None
VisualGeneratorAgent = None
session_manager = None

# WebSocket connection managers
global_chat_connections = set()  # Set of active WebSocket connections for global chat
admin_chat_connections = {}  # Dict[user_id, WebSocket] for admin chats

# Encryption setup for admin chats
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
fernet = Fernet(ENCRYPTION_KEY.encode())

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ai_engine, SessionManagerAgent, VisualGeneratorAgent, session_manager
    # Startup
    try:
        import ai_engine
        _ai_engine = ai_engine.ai_engine
        _SessionManagerAgent = ai_engine.SessionManagerAgent
        _VisualGeneratorAgent = ai_engine.VisualGeneratorAgent
        _session_manager = ai_engine.session_manager
        ai_engine = _ai_engine
        SessionManagerAgent = _SessionManagerAgent
        VisualGeneratorAgent = _VisualGeneratorAgent
        session_manager = _session_manager
        print("✓ AI Engine initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize AI Engine: {e}")
        # Continue without AI engine for now

    yield

    # Shutdown
    pass

load_dotenv()

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db")  # Use SQLite for development
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables
models.Base.metadata.create_all(bind=engine, checkfirst=True)

# Encryption setup for admin chats
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", Fernet.generate_key())
cipher = Fernet(ENCRYPTION_KEY)

# Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency', ['method', 'endpoint'])

app = FastAPI(
    title="AlMenn API",
    description="Backend for AlMenn education-focused AI assistant",
    version="1.0.0",
    lifespan=lifespan
)

# Rate limiting setup
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add SlowAPI middleware
app.add_middleware(SlowAPIMiddleware)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Middleware for request logging and metrics
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.datetime.utcnow()

    # Get user info if available
    user_id = "anonymous"
    try:
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            payload = verify_token(token)
            user_id = payload.get("sub", "anonymous")
    except:
        pass

    # Process request
    response = await call_next(request)

    # Calculate latency
    process_time = (datetime.datetime.utcnow() - start_time).total_seconds()

    # Log request
    logger.info(
        f"Request: {request.method} {request.url.path} "
        f"Status: {response.status_code} "
        f"User: {user_id} "
        f"IP: {request.client.host if request.client else 'unknown'} "
        f"Latency: {process_time:.3f}s"
    )

    # Update Prometheus metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=str(response.status_code)
    ).inc()

    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(process_time)

    return response

# Prometheus metrics endpoint
@app.get("/metrics")
async def metrics():
    return Response(media_type=CONTENT_TYPE_LATEST, content=generate_latest())

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],  # Add frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# JWT settings
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Pydantic models
class UserBase(BaseModel):
    email: EmailStr
    public_profile: Optional[Dict[str, Any]] = None

class UserCreate(UserBase):
    pass

class User(UserBase):
    id: UUID4
    created_at: datetime.datetime
    last_login_at: Optional[datetime.datetime] = None
    is_admin: bool = False

class MagicLinkRequest(BaseModel):
    email: EmailStr

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    session_id: UUID4

class CoinPackage(BaseModel):
    id: int
    price: int  # in rupees
    coins: int

class BuyCoinsRequest(BaseModel):
    package_id: int

class PaymentCallback(BaseModel):
    gateway_payload: Dict[str, Any]

class WalletResponse(BaseModel):
    balance: int
    transactions: List[Dict[str, Any]]

class AISessionStart(BaseModel):
    pass

class AIQuery(BaseModel):
    session_id: UUID4
    input_text: str
    attachments: Optional[List[UUID4]] = []
    mode: str = "rag"  # "rag" or "direct"

class AIResponse(BaseModel):
    response: str
    sources: Optional[List[Dict[str, Any]]] = []

class FileUploadResponse(BaseModel):
    file_id: UUID4
    filename: str
    size: int
    presigned_url: str

class ChatDownloadRequest(BaseModel):
    session_id: UUID4

class AdminMessageDelete(BaseModel):
    message_id: UUID4

class AdminUserBan(BaseModel):
    user_id: UUID4

class AdminUpdateUPIRequest(BaseModel):
    user_id: UUID4
    upi_id: str

class AdminUpdatePaymentGatewayRequest(BaseModel):
    gateway: str = "phonepe"
    merchant_id: str
    salt_key: str
    salt_index: str
    qr_code_url: Optional[str] = None
    is_active: bool = True

class SupportTicketCreate(BaseModel):
    subject: str
    category: str  # "technical", "billing", "abuse", "general"
    description: str
    priority: str = "medium"

class SupportTicketResponse(BaseModel):
    id: UUID4
    subject: str
    category: str
    description: str
    status: str
    priority: str
    created_at: datetime.datetime
    last_updated: Optional[datetime.datetime] = None

class ReportAbuseRequest(BaseModel):
    content_type: str  # "message", "file", "profile"
    content_id: UUID4
    reason: str
    description: Optional[str] = None

class GlobalChatMessage(BaseModel):
    message: str
    attachments: Optional[List[UUID4]] = []

class GlobalChatMessageResponse(BaseModel):
    id: UUID4
    user_id: UUID4
    message: str
    attachments: Optional[List[Dict[str, Any]]] = []
    created_at: datetime.datetime
    is_moderated: bool = False

class AdminChatMessage(BaseModel):
    message: str
    attachments: Optional[List[UUID4]] = []

class AdminChatMessageResponse(BaseModel):
    id: UUID4
    user_id: UUID4
    admin_id: UUID4
    message: str
    attachments: Optional[List[Dict[str, Any]]] = []
    is_from_admin: bool
    created_at: datetime.datetime

class FileUploadRequest(BaseModel):
    filename: str
    content_type: str

# Payment packages config
PAYMENT_PACKAGES = {
    1: {"price": 7, "coins": 15},
    2: {"price": 14, "coins": 35},
    3: {"price": 21, "coins": 75}
}

# PhonePe client setup
PHONEPE_MERCHANT_ID = os.getenv("PHONEPE_MERCHANT_ID")
PHONEPE_SALT_KEY = os.getenv("PHONEPE_SALT_KEY")
PHONEPE_SALT_INDEX = os.getenv("PHONEPE_SALT_INDEX")

if phonepe_available and PHONEPE_MERCHANT_ID and PHONEPE_SALT_KEY and PHONEPE_SALT_INDEX:
    phonepe_client = PhonePe(
        merchant_id=PHONEPE_MERCHANT_ID,
        salt_key=PHONEPE_SALT_KEY,
        salt_index=int(PHONEPE_SALT_INDEX)
    )
    logger.info("PhonePe client initialized successfully")
else:
    phonepe_client = None
    logger.warning("PhonePe credentials not configured or SDK not available")

# Utility functions
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def moderate_message(message: str) -> bool:
    """Basic content moderation - check for disallowed words"""
    # TODO: Implement more sophisticated moderation (AI-based, regex patterns, etc.)
    disallowed_words = ["spam", "abuse", "hate", "violence"]  # Example list
    message_lower = message.lower()
    return not any(word in message_lower for word in disallowed_words)

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = verify_token(token)
    user_id = payload.get("sub")
    session_id = payload.get("session_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")

    # Fetch user from database to check admin status
    db = SessionLocal()
    try:
        user = db.query(models.User).filter(models.User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return {"user_id": user_id, "session_id": session_id, "is_admin": user.is_admin}
    finally:
        db.close()

async def grant_daily_free_coins(user_id: str):
    """Grant daily free coins if eligible (once per day)"""
    db = SessionLocal()
    try:
        coins_record = db.query(models.Coins).filter(models.Coins.user_id == user_id).first()
        if not coins_record:
            coins_record = models.Coins(user_id=user_id, balance=0)
            db.add(coins_record)

        today = datetime.datetime.utcnow().date()
        last_grant_date = coins_record.free_daily_coins_last_grant_date

        if last_grant_date is None or last_grant_date.date() < today:
            # Grant 4 free coins
            coins_record.balance += 4
            coins_record.free_daily_coins_last_grant_date = datetime.datetime.utcnow()

            # Log transaction
            transaction = models.Transaction(
                user_id=user_id,
                type="free_grant",
                amount=4,
                description="Daily free coins granted"
            )
            db.add(transaction)
            db.commit()
    finally:
        db.close()

def deduct_coins_atomic(user_id: str, amount: int, description: str) -> bool:
    """Atomically deduct coins from user balance with row-level locking"""
    db = SessionLocal()
    try:
        # Use SELECT FOR UPDATE to lock the row
        coins_record = db.query(models.Coins).filter(
            models.Coins.user_id == user_id
        ).with_for_update().first()

        if not coins_record or coins_record.balance < amount:
            return False

        # Deduct coins
        coins_record.balance -= amount

        # Log transaction
        transaction = models.Transaction(
            user_id=user_id,
            type="usage",
            amount=-amount,
            description=description
        )
        db.add(transaction)
        db.commit()
        return True
    finally:
        db.close()

def get_active_session(user_id: str):
    """Get active AI session with remaining time"""
    db = SessionLocal()
    try:
        session = db.query(models.AI_Session).filter(
            models.AI_Session.user_id == user_id,
            models.AI_Session.is_active == True,
            models.AI_Session.expires_at > datetime.datetime.utcnow()
        ).first()

        if session:
            remaining_minutes = max(0, (session.expires_at - datetime.datetime.utcnow()).total_seconds() / 60)
            return {
                "session_id": session.id,
                "remaining_minutes": remaining_minutes,
                "expires_at": session.expires_at
            }
        return None
    finally:
        db.close()

async def send_magic_link_email(email: str, token: str):
    """Send magic link email to user"""
    # Get SMTP configuration from environment
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")
    from_email = os.getenv("FROM_EMAIL", smtp_username)

    if not smtp_username or not smtp_password:
        logger.error("SMTP credentials not configured")
        return

    # Create magic link URL (adjust base URL as needed)
    base_url = os.getenv("BASE_URL", "http://localhost:3000")
    magic_link_url = f"{base_url}/auth/verify?token={token}"

    # Create email message
    subject = "Your AlMenn Magic Link"
    body = f"""
    Hi there!

    Click the link below to sign in to AlMenn:

    {magic_link_url}

    This link will expire in 15 minutes.

    If you didn't request this link, please ignore this email.

    Best,
    The AlMenn Team
    """

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = email

    try:
        # Send email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.sendmail(from_email, email, msg.as_string())
        server.quit()
        logger.info(f"Magic link email sent to {email}")
    except Exception as e:
        logger.error(f"Failed to send magic link email to {email}: {e}")
        # In development, you might want to print the link instead
        print(f"Magic link for {email}: {magic_link_url}")

# Auth endpoints
@app.post("/auth/send_magic_link", response_model=dict)
async def send_magic_link(request: MagicLinkRequest):
    db = SessionLocal()
    try:
        # Check if user exists, create if not
        user = db.query(models.User).filter(models.User.email == request.email).first()
        if not user:
            user = models.User(email=request.email)
            db.add(user)
            db.commit()
            db.refresh(user)

        # Generate magic link token (JWT with short expiry)
        magic_token = jwt.encode(
            {"sub": str(user.id), "type": "magic_link", "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=15)},
            SECRET_KEY,
            algorithm=ALGORITHM
        )

        # Create session record
        session = models.Session(
            user_id=str(user.id),
            token=magic_token,
            expires_at=datetime.datetime.utcnow() + datetime.timedelta(minutes=15)
        )
        db.add(session)
        db.commit()

        # Send email with magic link
        await send_magic_link_email(request.email, magic_token)

        return {"message": "Magic link sent to email"}
    finally:
        db.close()

@app.post("/auth/verify_magic_link", response_model=TokenResponse)
async def verify_magic_link(token: str):
    db = SessionLocal()
    try:
        # Verify magic link token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "magic_link":
            raise HTTPException(status_code=400, detail="Invalid magic link")

        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=400, detail="Invalid magic link")

        # Check if session exists and is valid
        session = db.query(models.Session).filter(
            models.Session.token == token,
            models.Session.expires_at > datetime.datetime.utcnow()
        ).first()
        if not session:
            raise HTTPException(status_code=400, detail="Magic link expired or invalid")

        # Update user last login
        user = db.query(models.User).filter(models.User.id == user_id).first()
        if user:
            user.last_login_at = datetime.datetime.utcnow()
            db.commit()

        # Create new session for authenticated user
        session_id = uuid.uuid4()
        access_token = create_access_token({"sub": user_id, "session_id": str(session_id)})

        # Invalidate magic link session
        db.delete(session)
        db.commit()

        return TokenResponse(access_token=access_token, session_id=session_id)
    finally:
        db.close()

@app.get("/auth/logout", dependencies=[Depends(get_current_user)])
async def logout(current_user: dict = Depends(get_current_user)):
    # TODO: Invalidate session in database
    return {"message": "Logged out successfully"}

# User endpoints
@app.get("/me", response_model=User)
async def get_me(current_user: dict = Depends(get_current_user)):
    # TODO: Fetch user from database
    return User(
        id=uuid.uuid4(),
        email="user@example.com",
        created_at=datetime.datetime.utcnow()
    )

@app.put("/me", response_model=User)
async def update_me(user_data: UserBase, current_user: dict = Depends(get_current_user)):
    # TODO: Update user in database
    return User(
        id=uuid.uuid4(),
        email=user_data.email,
        public_profile=user_data.public_profile,
        created_at=datetime.datetime.utcnow()
    )

# Coins endpoints
@app.get("/wallet", response_model=WalletResponse)
async def get_wallet(current_user: dict = Depends(get_current_user)):
    # Grant daily free coins if eligible
    await grant_daily_free_coins(current_user["user_id"])

    # Fetch balance and transactions from database
    db = SessionLocal()
    try:
        coins_record = db.query(models.Coins).filter(models.Coins.user_id == current_user["user_id"]).first()
        if not coins_record:
            coins_record = models.Coins(user_id=current_user["user_id"], balance=0)
            db.add(coins_record)
            db.commit()

        transactions = db.query(models.Transaction).filter(
            models.Transaction.user_id == current_user["user_id"]
        ).order_by(models.Transaction.created_at.desc()).limit(10).all()

        return WalletResponse(
            balance=coins_record.balance,
            transactions=[{
                "id": str(t.id),
                "type": t.type,
                "amount": t.amount,
                "description": t.description,
                "created_at": t.created_at
            } for t in transactions]
        )
    finally:
        db.close()

@app.post("/buy_coins")
async def buy_coins(request: BuyCoinsRequest, current_user: dict = Depends(get_current_user)):
    if request.package_id not in PAYMENT_PACKAGES:
        raise HTTPException(status_code=400, detail="Invalid package")
    package = PAYMENT_PACKAGES[request.package_id]
    # TODO: Create payment order with gateway
    # TODO: Return gateway payload for frontend
    return {"order_id": "mock_order", "amount": package["price"], "gateway_payload": {}}

@app.post("/webhook/payment_callback")
async def payment_callback(callback: PaymentCallback):
    # TODO: Verify gateway signature
    # TODO: Update transaction status
    # TODO: Add coins to user balance
    return {"status": "success"}

# AI endpoints
@app.post("/ai/session/start")
async def start_ai_session(current_user: dict = Depends(get_current_user)):
    # Check if user has active session
    active_session = get_active_session(current_user["user_id"])
    if active_session:
        return {
            "session_id": str(active_session["session_id"]),
            "minutes_remaining": active_session["remaining_minutes"]
        }

    # Check coins and deduct 4 coins for new session
    success = await deduct_coins_atomic(current_user["user_id"], 4, "AI session start (60 minutes)")
    if not success:
        raise HTTPException(status_code=402, detail="Insufficient coins. Need 4 coins for AI session.")

    # Create new AI session
    db = SessionLocal()
    try:
        expires_at = datetime.datetime.utcnow() + datetime.timedelta(minutes=60)
        ai_session = models.AI_Session(
            user_id=current_user["user_id"],
            expires_at=expires_at,
            is_active=True,
            total_duration_minutes=60
        )
        db.add(ai_session)
        db.commit()
        db.refresh(ai_session)

        return {
            "session_id": str(ai_session.id),
            "minutes_remaining": 60
        }
    finally:
        db.close()

@app.post("/ai/session/extend")
async def extend_ai_session(current_user: dict = Depends(get_current_user)):
    # Check coins and deduct 4 coins for extension
    success = await deduct_coins_atomic(current_user["user_id"], 4, "AI session extension (60 minutes)")
    if not success:
        raise HTTPException(status_code=402, detail="Insufficient coins. Need 4 coins for session extension.")

    # Extend active session or create new one
    db = SessionLocal()
    try:
        session = db.query(models.AI_Session).filter(
            models.AI_Session.user_id == current_user["user_id"],
            models.AI_Session.is_active == True,
            models.AI_Session.expires_at > datetime.datetime.utcnow()
        ).first()

        if session:
            # Extend existing session
            session.expires_at += datetime.timedelta(minutes=60)
            session.total_duration_minutes += 60
        else:
            # Create new session if none active
            session = models.AI_Session(
                user_id=current_user["user_id"],
                expires_at=datetime.datetime.utcnow() + datetime.timedelta(minutes=60),
                is_active=True,
                total_duration_minutes=60
            )
            db.add(session)

        db.commit()
        db.refresh(session)

        remaining_minutes = max(0, (session.expires_at - datetime.datetime.utcnow()).total_seconds() / 60)
        return {"minutes_remaining": remaining_minutes}
    finally:
        db.close()

@app.get("/ai/session/status")
async def get_ai_session_status(current_user: dict = Depends(get_current_user)):
    # Get remaining minutes for active session
    active_session = get_active_session(current_user["user_id"])
    if active_session:
        return {"minutes_remaining": active_session["remaining_minutes"]}
    return {"minutes_remaining": 0}

@app.post("/ai/query", response_model=AIResponse)
@limiter.limit("10/minute")  # Rate limit AI queries to 10 per minute
async def ai_query(request: AIQuery, current_user: dict = Depends(get_current_user)):
    # Process query with RAG/NN
    # Check session validity
    if not await session_manager.validate_session(str(request.session_id), current_user["user_id"]):
        raise HTTPException(status_code=403, detail="AI session expired or inactive")

    # Process query using AI engine
    response_data = await ai_engine.process_query(
        query=request.input_text,
        session_id=str(request.session_id),
        user_id=current_user["user_id"],
        mode=request.mode,
        attachments=request.attachments
    )

    return AIResponse(
        response=response_data["response"],
        sources=response_data.get("sources", [])
    )

@app.post("/ai/upload_file", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = FastAPIFile(...), current_user: dict = Depends(get_current_user)):
    # Generate presigned URL for S3 upload
    import boto3
    from botocore.client import Config
    import os

    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION', 'us-east-1'),
        config=Config(signature_version='s3v4')
    )

    bucket_name = os.getenv('S3_BUCKET_NAME', 'almenn-files')
    file_key = f"uploads/{current_user['user_id']}/{uuid.uuid4()}_{file.filename}"

    # Generate presigned URL for upload
    presigned_url = s3_client.generate_presigned_url(
        'put_object',
        Params={
            'Bucket': bucket_name,
            'Key': file_key,
            'ContentType': file.content_type
        },
        ExpiresIn=3600  # 1 hour
    )

    # Store file metadata in database
    db = SessionLocal()
    try:
        file_record = models.File(
            user_id=current_user["user_id"],
            filename=file.filename,
            file_path=file_key,  # S3 key
            file_size=len(await file.read()),
            mime_type=file.content_type
        )
        db.add(file_record)
        db.commit()
        db.refresh(file_record)

        # Queue text extraction task
        from tasks import extract_text_from_file
        extract_text_from_file.delay(str(file_record.id))

        return FileUploadResponse(
            file_id=file_record.id,
            filename=file.filename,
            size=file_record.file_size,
            presigned_url=presigned_url
        )
    finally:
        db.close()

# Chat endpoints
@app.websocket("/ws/global")
async def global_chat_websocket(websocket: WebSocket):
    await websocket.accept()
    global_chat_connections.add(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)

            # Validate message
            if not moderate_message(message_data.get("message", "")):
                await websocket.send_json({"error": "Message contains disallowed content"})
                continue

            # Store message in database
            db = SessionLocal()
            try:
                # Get user_id from JWT token (simplified - in production, validate token)
                user_id = message_data.get("user_id", "anonymous")

                chat_message = models.GlobalChatMessage(
                    user_id=user_id,
                    message=message_data["message"],
                    attachments=message_data.get("attachments", [])
                )
                db.add(chat_message)
                db.commit()
                db.refresh(chat_message)

                # Generate presigned URLs for attachments
                attachments_with_urls = []
                if chat_message.attachments:
                    for file_id in chat_message.attachments:
                        file_record = db.query(models.File).filter(models.File.id == file_id).first()
                        if file_record:
                            # Generate presigned URL for download
                            import boto3
                            from botocore.client import Config
                            s3_client = boto3.client(
                                's3',
                                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                                region_name=os.getenv('AWS_REGION', 'us-east-1'),
                                config=Config(signature_version='s3v4')
                            )
                            presigned_url = s3_client.generate_presigned_url(
                                'get_object',
                                Params={'Bucket': os.getenv('S3_BUCKET_NAME', 'almenn-files'), 'Key': file_record.file_path},
                                ExpiresIn=3600  # 1 hour
                            )
                            attachments_with_urls.append({
                                "file_id": str(file_record.id),
                                "filename": file_record.filename,
                                "presigned_url": presigned_url
                            })

                # Broadcast to all connected clients
                response_data = {
                    "id": str(chat_message.id),
                    "user_id": chat_message.user_id,
                    "message": chat_message.message,
                    "attachments": attachments_with_urls,
                    "created_at": chat_message.created_at.isoformat(),
                    "is_moderated": chat_message.is_moderated
                }

                # Broadcast to all connected clients
                disconnected = set()
                for conn in global_chat_connections:
                    try:
                        await conn.send_json(response_data)
                    except:
                        disconnected.add(conn)

                # Remove disconnected clients
                global_chat_connections.difference_update(disconnected)

            finally:
                db.close()
    except WebSocketDisconnect:
        global_chat_connections.discard(websocket)

@app.websocket("/ws/admin/{user_id}")
async def admin_chat_websocket(websocket: WebSocket, user_id: str):
    await websocket.accept()
    admin_chat_connections[user_id] = websocket
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)

            # Get admin_id from message (simplified - in production, validate admin token)
            admin_id = message_data.get("admin_id", "admin")

            # Encrypt message content
            encrypted_message = fernet.encrypt(message_data["message"].encode()).decode()

            # Store encrypted message in database
            db = SessionLocal()
            try:
                chat_message = models.AdminUserChat(
                    user_id=user_id,
                    admin_id=admin_id,
                    message=encrypted_message,
                    is_from_admin=message_data.get("is_from_admin", False),
                    attachments=message_data.get("attachments", [])
                )
                db.add(chat_message)
                db.commit()
                db.refresh(chat_message)

                # Generate presigned URLs for attachments
                attachments_with_urls = []
                if chat_message.attachments:
                    for file_id in chat_message.attachments:
                        file_record = db.query(models.File).filter(models.File.id == file_id).first()
                        if file_record:
                            import boto3
                            from botocore.client import Config
                            s3_client = boto3.client(
                                's3',
                                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                                region_name=os.getenv('AWS_REGION', 'us-east-1'),
                                config=Config(signature_version='s3v4')
                            )
                            presigned_url = s3_client.generate_presigned_url(
                                'get_object',
                                Params={'Bucket': os.getenv('S3_BUCKET_NAME', 'almenn-files'), 'Key': file_record.file_path},
                                ExpiresIn=3600
                            )
                            attachments_with_urls.append({
                                "file_id": str(file_record.id),
                                "filename": file_record.filename,
                                "presigned_url": presigned_url
                            })

                # Send response back to sender
                response_data = {
                    "id": str(chat_message.id),
                    "user_id": chat_message.user_id,
                    "admin_id": chat_message.admin_id,
                    "message": chat_message.message,  # Encrypted
                    "attachments": attachments_with_urls,
                    "is_from_admin": chat_message.is_from_admin,
                    "created_at": chat_message.created_at.isoformat()
                }
                await websocket.send_json(response_data)

            finally:
                db.close()
    except WebSocketDisconnect:
        if user_id in admin_chat_connections:
            del admin_chat_connections[user_id]

@app.post("/chat/download/{session_id}")
async def download_chat(session_id: UUID4, current_user: dict = Depends(get_current_user)):
    # TODO: Check authorization
    # TODO: Generate transcript
    return {"download_url": "mock_url"}

# Admin endpoints
@app.get("/admin/moderation/queue")
async def get_moderation_queue(current_user: dict = Depends(get_current_user)):
    # TODO: Check admin permission
    # TODO: Get pending moderations
    return {"queue": []}

@app.post("/admin/message/delete")
async def delete_message(request: AdminMessageDelete, current_user: dict = Depends(get_current_user)):
    # TODO: Check admin permission
    # TODO: Delete message
    return {"status": "deleted"}

@app.post("/admin/user/ban")
async def ban_user(request: AdminUserBan, current_user: dict = Depends(get_current_user)):
    # TODO: Check admin permission
    # TODO: Ban user
    return {"status": "banned"}

@app.put("/admin/user/upi")
async def update_user_upi(request: AdminUpdateUPIRequest, current_user: dict = Depends(get_current_user)):
    # Check admin permission
    if not current_user.get("is_admin", False):
        raise HTTPException(status_code=403, detail="Admin access required")

    db = SessionLocal()
    try:
        # Fetch user
        user = db.query(models.User).filter(models.User.id == request.user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Update UPI ID
        user.upi_id = request.upi_id
        db.commit()

        return {"status": "success", "message": f"UPI ID updated for user {request.user_id}"}
    finally:
        db.close()

@app.put("/admin/payment_gateway")
async def update_payment_gateway(request: AdminUpdatePaymentGatewayRequest, current_user: dict = Depends(get_current_user)):
    # Check admin permission
    if not current_user.get("is_admin", False):
        raise HTTPException(status_code=403, detail="Admin access required")

    db = SessionLocal()
    try:
        # Check if gateway settings exist, create or update
        gateway_settings = db.query(models.PaymentGatewaySettings).filter(
            models.PaymentGatewaySettings.gateway == request.gateway
        ).first()

        if gateway_settings:
            # Update existing settings
            gateway_settings.merchant_id = request.merchant_id
            gateway_settings.salt_key = request.salt_key
            gateway_settings.salt_index = request.salt_index
            gateway_settings.qr_code_url = request.qr_code_url
            gateway_settings.is_active = request.is_active
            gateway_settings.updated_by = current_user["user_id"]
        else:
            # Create new settings
            gateway_settings = models.PaymentGatewaySettings(
                gateway=request.gateway,
                merchant_id=request.merchant_id,
                salt_key=request.salt_key,
                salt_index=request.salt_index,
                qr_code_url=request.qr_code_url,
                is_active=request.is_active,
                updated_by=current_user["user_id"]
            )
            db.add(gateway_settings)

        db.commit()
        db.refresh(gateway_settings)

        return {
            "status": "success",
            "message": f"Payment gateway settings updated for {request.gateway}",
            "gateway_settings": {
                "gateway": gateway_settings.gateway,
                "merchant_id": gateway_settings.merchant_id,
                "qr_code_url": gateway_settings.qr_code_url,
                "is_active": gateway_settings.is_active
            }
        }
    finally:
        db.close()

@app.get("/admin/payment_gateway/{gateway}")
async def get_payment_gateway_settings(gateway: str, current_user: dict = Depends(get_current_user)):
    # Check admin permission
    if not current_user.get("is_admin", False):
        raise HTTPException(status_code=403, detail="Admin access required")

    db = SessionLocal()
    try:
        gateway_settings = db.query(models.PaymentGatewaySettings).filter(
            models.PaymentGatewaySettings.gateway == gateway
        ).first()

        if not gateway_settings:
            raise HTTPException(status_code=404, detail="Payment gateway settings not found")

        return {
            "gateway": gateway_settings.gateway,
            "merchant_id": gateway_settings.merchant_id,
            "qr_code_url": gateway_settings.qr_code_url,
            "is_active": gateway_settings.is_active,
            "updated_at": gateway_settings.updated_at
        }
    finally:
        db.close()

# Support endpoints
@app.post("/support/ticket", response_model=SupportTicketResponse)
async def create_support_ticket(request: SupportTicketCreate, current_user: dict = Depends(get_current_user)):
    db = SessionLocal()
    try:
        ticket = models.SupportTicket(
            user_id=current_user["user_id"],
            subject=request.subject,
            category=request.category,
            description=request.description,
            priority=request.priority
        )
        db.add(ticket)
        db.commit()
        db.refresh(ticket)

        # Log analytics event
        await analytics.log_event(
            event_type="support_ticket_created",
            user_id=current_user["user_id"],
            metadata={"category": request.category, "priority": request.priority}
        )

        return SupportTicketResponse(
            id=ticket.id,
            subject=ticket.subject,
            category=ticket.category,
            description=ticket.description,
            status=ticket.status,
            priority=ticket.priority,
            created_at=ticket.created_at,
            last_updated=ticket.last_updated
        )
    finally:
        db.close()

@app.get("/support/tickets", response_model=List[SupportTicketResponse])
async def get_support_tickets(current_user: dict = Depends(get_current_user)):
    db = SessionLocal()
    try:
        tickets = db.query(models.SupportTicket).filter(
            models.SupportTicket.user_id == current_user["user_id"]
        ).order_by(models.SupportTicket.created_at.desc()).all()

        return [SupportTicketResponse(
            id=ticket.id,
            subject=ticket.subject,
            category=ticket.category,
            description=ticket.description,
            status=ticket.status,
            priority=ticket.priority,
            created_at=ticket.created_at,
            last_updated=ticket.last_updated
        ) for ticket in tickets]
    finally:
        db.close()

@app.post("/report/abuse")
async def report_abuse(request: ReportAbuseRequest, current_user: dict = Depends(get_current_user)):
    db = SessionLocal()
    try:
        # Create support ticket for abuse report
        ticket = models.SupportTicket(
            user_id=current_user["user_id"],
            subject=f"Abuse Report: {request.content_type}",
            category="abuse",
            description=f"Content Type: {request.content_type}\nContent ID: {request.content_id}\nReason: {request.reason}\nDescription: {request.description or 'N/A'}",
            priority="high"
        )
        db.add(ticket)
        db.commit()

        # Log analytics event
        await analytics.log_event(
            event_type="abuse_reported",
            user_id=current_user["user_id"],
            metadata={"content_type": request.content_type, "reason": request.reason}
        )

        return {"message": "Abuse report submitted successfully", "ticket_id": str(ticket.id)}
    finally:
        db.close()

@app.get("/welcome")
async def welcome(request: Request):
    return {"message": "Welcome to the AlMenn API Service!"}

@app.get("/")
async def root():
    return {"message": "Welcome to AlMenn API"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
