# AlMenn Deployment Guide

This guide covers all deployment options for the AlMenn educational AI assistant platform.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Local Development](#local-development)
- [Production Deployment](#production-deployment)
  - [Railway + Vercel (Recommended)](#railway--vercel-recommended)
  - [Docker Compose](#docker-compose)
  - [Kubernetes](#kubernetes)
- [Environment Configuration](#environment-configuration)
- [Monitoring & Maintenance](#monitoring--maintenance)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements
- Python 3.11+
- Node.js 18+
- PostgreSQL 15+
- Redis 7+
- Docker & Docker Compose (for containerized deployment)

### Accounts & Services
- Railway account (backend deployment)
- Vercel account (frontend deployment)
- AWS account (file storage, optional)
- Pinecone account (vector database, optional)
- OpenAI API key (AI features)

## Local Development

### Option 1: Docker Compose (Recommended)

1. **Clone and setup**
   ```bash
   git clone https://github.com/yourusername/almenn.git
   cd almenn
   cp backend/env_example.txt backend/.env
   ```

2. **Configure environment variables**
   Edit `backend/.env` with your local configuration:
   ```env
   DATABASE_URL=postgresql://almenn_user:almenn_password@postgres:5432/almenn
   REDIS_URL=redis://redis:6379
   SECRET_KEY=your-development-secret-key
   ENCRYPTION_KEY=your-32-character-encryption-key
   ENVIRONMENT=development
   ```

3. **Start services**
   ```bash
   docker compose -f infra/docker-compose.yml up -d
   ```

4. **View logs**
   ```bash
   docker compose -f infra/docker-compose.yml logs -f
   ```

5. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

### Option 2: Manual Setup

1. **Database setup**
   ```bash
   # Using Docker
   docker run -d --name postgres -p 5432:5432 \
     -e POSTGRES_DB=almenn \
     -e POSTGRES_USER=almenn_user \
     -e POSTGRES_PASSWORD=almenn_password \
     postgres:15

   docker run -d --name redis -p 6379:6379 redis:7
   ```

2. **Backend setup**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   cp env_example.txt .env
   # Edit .env with your configuration
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Frontend setup**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

## Production Deployment

### Railway + Vercel (Recommended)

#### Backend Deployment (Railway)

1. **Install Railway CLI**
   ```bash
   npm install -g @railway/cli
   railway login
   ```

2. **Initialize project**
   ```bash
   railway init
   # Select "Empty Project" and name it "almenn-backend"
   ```

3. **Deploy**
   ```bash
   railway up
   ```

4. **Configure environment variables in Railway dashboard**
   ```
   SECRET_KEY=your-production-secret-key
   ENCRYPTION_KEY=your-32-character-encryption-key
   ENVIRONMENT=production
   # Railway auto-provides DATABASE_URL and REDIS_URL
   # Add other API keys as needed
   ```

#### Frontend Deployment (Vercel)

1. **Install Vercel CLI**
   ```bash
   npm install -g vercel
   vercel login
   ```

2. **Deploy frontend**
   ```bash
   cd frontend
   vercel --prod
   ```

3. **Configure environment variables in Vercel dashboard**
   ```
   VITE_API_BASE_URL=https://your-railway-backend-url
   ```

### Docker Compose

1. **Build and deploy**
   ```bash
   docker compose -f infra/docker-compose.yml up -d --build
   ```

2. **Scale services (optional)**
   ```bash
   docker compose -f infra/docker-compose.yml up -d --scale celery-worker=3
   ```

### Kubernetes

1. **Apply Kubernetes manifests**
   ```bash
   kubectl apply -f infra/k8s/
   ```

2. **Check deployment status**
   ```bash
   kubectl get pods
   kubectl get services
   ```

3. **Configure ingress**
   Update `infra/k8s/ingress.yaml` with your domain and SSL certificates.

## Environment Configuration

### Backend Environment Variables

#### Required
```env
DATABASE_URL=postgresql://user:pass@host:port/db
REDIS_URL=redis://host:port
SECRET_KEY=your-super-secret-key-here
ENCRYPTION_KEY=your-32-character-encryption-key
```

#### Optional (AI Features)
```env
OPENAI_API_KEY=your-openai-api-key
PINECONE_API_KEY=your-pinecone-api-key
```

#### Optional (File Storage)
```env
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_REGION=us-east-1
S3_BUCKET_NAME=your-s3-bucket-name
```

#### Optional (Email)
```env
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
FROM_EMAIL=noreply@almenn.ai
```

#### Optional (Payments)
```env
PHONEPE_MERCHANT_ID=your-phonepe-merchant-id
PHONEPE_SALT_KEY=your-phonepe-salt-key
PHONEPE_SALT_INDEX=your-phonepe-salt-index
```

### Frontend Environment Variables

```env
VITE_API_BASE_URL=https://your-backend-url
```

## Monitoring & Maintenance

### Health Checks

- Backend health: `GET /health`
- Metrics: `GET /metrics` (Prometheus format)

### Logs

```bash
# Railway logs
railway logs

# Docker logs
docker compose -f infra/docker-compose.yml logs -f backend

# Kubernetes logs
kubectl logs -f deployment/backend
```

### Database Maintenance

```bash
# Backup (Railway)
railway run pg_dump $DATABASE_URL > backup.sql

# Restore
railway run psql $DATABASE_URL < backup.sql
```

### Updates

1. **Update code**
   ```bash
   git pull origin main
   ```

2. **Rebuild containers**
   ```bash
   docker compose -f infra/docker-compose.yml up -d --build
   ```

3. **Run migrations (if any)**
   ```bash
   # Database schema is auto-created, but check for migration scripts
   ```

## Troubleshooting

### Common Issues

#### Backend won't start
- Check environment variables
- Verify database connectivity
- Check logs: `docker compose logs backend`

#### Frontend shows API errors
- Verify `VITE_API_BASE_URL` is correct
- Check CORS settings in backend
- Ensure backend is running and accessible

#### Database connection errors
- Verify DATABASE_URL format
- Check database credentials
- Ensure database is running and accessible

#### AI features not working
- Verify API keys are set
- Check AI service quotas
- Review AI engine logs

### Performance Issues

#### High CPU usage
- Scale Celery workers: `docker compose up -d --scale celery-worker=5`
- Check for memory leaks in AI processing

#### Slow responses
- Check database query performance
- Monitor Redis cache hit rates
- Review rate limiting settings

#### Out of memory
- Increase container memory limits
- Optimize AI model loading
- Implement request queuing

### Security Checklist

- [ ] SECRET_KEY is set and secure
- [ ] ENCRYPTION_KEY is 32 characters
- [ ] Database credentials are strong
- [ ] API keys are not exposed in logs
- [ ] CORS is properly configured
- [ ] Rate limiting is enabled
- [ ] HTTPS is enabled in production
- [ ] Security headers are set

## Support

For deployment issues:
1. Check this guide
2. Review logs and error messages
3. Create an issue on GitHub
4. Contact support@almenn.ai

## Deployment Checklist

- [ ] Environment variables configured
- [ ] Database accessible
- [ ] Redis accessible
- [ ] API keys set (if using AI features)
- [ ] File storage configured (if using uploads)
- [ ] Email configured (if using auth)
- [ ] Payments configured (if using coin purchases)
- [ ] Frontend built and deployed
- [ ] Backend deployed and healthy
- [ ] Domain configured
- [ ] SSL certificates installed
- [ ] Monitoring set up
- [ ] Backup strategy in place
