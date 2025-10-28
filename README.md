# AlMenn - Educational AI Assistant

AlMenn is an educational AI assistant platform that provides personalized learning experiences through advanced AI technology, including RAG (Retrieval-Augmented Generation), visual generation, and agent-based systems.

## Features

- **AI-Powered Learning**: Interactive AI conversations with educational content
- **File Processing**: Upload and analyze documents, images, and other files
- **Real-time Chat**: Global chat and admin support systems
- **Coin System**: Gamified learning with virtual currency
- **Admin Panel**: Comprehensive moderation and management tools
- **Analytics**: Detailed usage tracking and insights

## Tech Stack

- **Backend**: FastAPI (Python)
- **Frontend**: HTML/CSS/JavaScript (Vanilla)
- **Database**: PostgreSQL
- **AI/ML**: PyTorch, Transformers, Sentence Transformers
- **Vector DB**: Pinecone
- **Task Queue**: Celery + Redis
- **File Storage**: AWS S3
- **Deployment**: Railway (Backend), Vercel (Frontend)

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- PostgreSQL
- Redis
- Railway account (for production deployment)
- Vercel account (for frontend deployment)

### Local Development

#### Option 1: Docker Compose (Recommended)
```bash
# Clone the repository
git clone https://github.com/yourusername/almenn.git
cd almenn

# Copy environment file and configure
cp backend/env_example.txt backend/.env
# Edit backend/.env with your configuration

# Start all services
docker-compose -f infra/docker-compose.yml up -d

# View logs
docker-compose -f infra/docker-compose.yml logs -f
```

#### Option 2: Manual Setup

1. **Database Setup**
   ```bash
   # Install PostgreSQL and Redis locally, or use Docker
   docker run -d --name postgres -p 5432:5432 -e POSTGRES_DB=almenn -e POSTGRES_USER=almenn_user -e POSTGRES_PASSWORD=almenn_password postgres:15
   docker run -d --name redis -p 6379:6379 redis:7
   ```

2. **Backend Setup**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   cp env_example.txt .env  # Configure your environment variables
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

### Production Deployment

#### Backend (Railway)
1. **Connect to Railway**
   ```bash
   npm install -g @railway/cli
   railway login
   ```

2. **Deploy Backend**
   ```bash
   railway init
   railway up
   ```

3. **Configure Environment Variables**
   Set the following in Railway dashboard:
   - `DATABASE_URL` (auto-provided by Railway)
   - `REDIS_URL` (auto-provided by Railway)
   - `SECRET_KEY`
   - `ENCRYPTION_KEY`
   - Other API keys as needed

#### Frontend (Vercel)
1. **Deploy Frontend**
   ```bash
   cd frontend
   vercel --prod
   ```

2. **Configure Environment Variables**
   Set `VITE_API_BASE_URL` to your Railway backend URL in Vercel dashboard.

#### Alternative: Kubernetes Deployment
```bash
# Deploy to Kubernetes cluster
kubectl apply -f infra/k8s/
```

## Environment Variables

### Backend
```env
DATABASE_URL=postgresql://user:pass@host:port/db
REDIS_URL=redis://host:port
SECRET_KEY=your-secret-key
ENCRYPTION_KEY=your-encryption-key
OPENAI_API_KEY=your-openai-key
PINECONE_API_KEY=your-pinecone-key
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
```

### Frontend
```env
VITE_API_BASE_URL=https://your-railway-backend-url
```

## API Documentation

Once the backend is running, visit `http://localhost:8000/docs` for interactive API documentation.

## Project Structure

```
almenn/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── models.py            # SQLAlchemy models
│   ├── ai_engine.py         # AI processing engine
│   ├── tasks.py             # Celery tasks
│   ├── requirements.txt     # Python dependencies
│   └── tests/               # Test suite
├── frontend/
│   ├── index.html           # Main HTML file
│   ├── styles.css           # CSS styles
│   ├── script.js            # JavaScript logic
│   └── package.json         # Node dependencies
├── infra/
│   ├── docker-compose.yml   # Local development
│   ├── Dockerfile.backend   # Backend container
│   └── Dockerfile.frontend  # Frontend container
├── docs/                    # Documentation
└── railway.json             # Railway deployment config
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For support, email support@almenn.ai or create an issue in this repository.
