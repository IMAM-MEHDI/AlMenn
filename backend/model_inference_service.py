"""
Model Inference Service - Dedicated GPU-enabled service for AI model inference

This service provides:
- GPU-accelerated model inference
- Separate from main backend for better resource management
- REST API for model operations
- Health monitoring and metrics
"""

import os
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sentence_transformers
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn
import psutil
import GPUtil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Model Inference Service", version="1.0.0")

class InferenceRequest(BaseModel):
    query: str
    context: Optional[str] = None
    max_length: Optional[int] = 512
    temperature: Optional[float] = 0.7

class EmbeddingRequest(BaseModel):
    text: str

class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    gpu_memory_used: float
    gpu_memory_total: float
    cpu_usage: float
    memory_usage: float
    model_loaded: bool

class ModelInferenceService:
    """GPU-accelerated model inference service"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm_model = None
        self.llm_tokenizer = None
        self.embedding_model = None
        self.model_loaded = False

        logger.info(f"Initializing Model Inference Service on device: {self.device}")
        self._initialize_models()

    def _initialize_models(self):
        """Initialize AI models"""
        try:
            # Initialize embedding model
            logger.info("Loading embedding model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

            # Initialize LLM (use smaller model for inference service)
            logger.info("Loading LLM...")
            model_name = "microsoft/DialoGPT-medium"  # Medium model for better performance
            self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.llm_model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

            # Set pad token if not present
            if self.llm_tokenizer.pad_token is None:
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

            self.model_loaded = True
            logger.info("Models loaded successfully")

        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            self.model_loaded = False

    async def generate_response(self, query: str, context: str = "", max_length: int = 512, temperature: float = 0.7) -> str:
        """Generate AI response using LLM"""
        if not self.model_loaded or not self.llm_model or not self.llm_tokenizer:
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            # Prepare prompt
            if context:
                prompt = f"Context: {context}\n\nQuestion: {query}\n\nProvide an educational, helpful answer:"
            else:
                prompt = f"Question: {query}\n\nProvide an educational, helpful answer:"

            # Tokenize
            inputs = self.llm_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)

            # Generate response
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.llm_tokenizer.pad_token,
                    eos_token_id=self.llm_tokenizer.eos_token
                )

            response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Clean up response
            response = response.replace(prompt, "").strip()

            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise HTTPException(status_code=500, detail="Inference failed")

    async def generate_embedding(self, text: str) -> list:
        """Generate text embedding"""
        if not self.embedding_model:
            raise HTTPException(status_code=503, detail="Embedding model not loaded")

        try:
            embedding = self.embedding_model.encode(text).tolist()
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise HTTPException(status_code=500, detail="Embedding generation failed")

    def get_health_status(self) -> Dict[str, Any]:
        """Get service health status"""
        gpu_available = torch.cuda.is_available()
        gpu_memory_used = 0.0
        gpu_memory_total = 0.0

        if gpu_available:
            try:
                gpu = GPUtil.getGPUs()[0]
                gpu_memory_used = gpu.memoryUsed
                gpu_memory_total = gpu.memoryTotal
            except:
                pass

        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent

        return {
            "status": "healthy" if self.model_loaded else "unhealthy",
            "gpu_available": gpu_available,
            "gpu_memory_used": gpu_memory_used,
            "gpu_memory_total": gpu_memory_total,
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "model_loaded": self.model_loaded
        }

# Global service instance
inference_service = ModelInferenceService()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return inference_service.get_health_status()

@app.post("/generate")
async def generate_response(request: InferenceRequest):
    """Generate AI response"""
    response = await inference_service.generate_response(
        query=request.query,
        context=request.context or "",
        max_length=request.max_length,
        temperature=request.temperature
    )
    return {"response": response}

@app.post("/embed")
async def generate_embedding(request: EmbeddingRequest):
    """Generate text embedding"""
    embedding = await inference_service.generate_embedding(request.text)
    return {"embedding": embedding}

@app.get("/metrics")
async def get_metrics():
    """Get service metrics"""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "device": str(inference_service.device),
        "model_loaded": inference_service.model_loaded,
        "gpu_memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
        "gpu_memory_reserved": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    logger.info(f"Starting Model Inference Service on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
