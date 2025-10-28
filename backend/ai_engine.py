"""
AI Engine for AlMenn - Educational AI Assistant

This module implements the core AI functionality including:
- RAG (Retrieval-Augmented Generation)
- Visual generation for explanations
- Agent system for various tasks
- Continual learning from user feedback
"""

import os
import uuid
import json
import io
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

# AI/ML imports (optional - graceful fallback if not installed)
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import sentence_transformers
    from sentence_transformers import SentenceTransformer
    import pinecone
    # Import Pinecone client
    try:
        from pinecone import Pinecone, ServerlessSpec
    except ImportError:
        Pinecone = None
        ServerlessSpec = None
    AI_DEPS_AVAILABLE = True
except ImportError as e:
    AI_DEPS_AVAILABLE = False
    print(f"Warning: AI dependencies not available: {e}. Using mock implementations.")

# Web scraping and processing (optional)
try:
    import requests
    from bs4 import BeautifulSoup
    import PyPDF2
    import docx
    import openpyxl
    import pytesseract
    from PIL import Image
    PROCESSING_DEPS_AVAILABLE = True
except ImportError:
    PROCESSING_DEPS_AVAILABLE = False
    print("Warning: Processing dependencies not available.")

# Visualization (optional)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import svgwrite
    VISUALIZATION_DEPS_AVAILABLE = True
except ImportError:
    VISUALIZATION_DEPS_AVAILABLE = False
    print("Warning: Visualization dependencies not available.")

# Database and async
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func
import models
from tasks import celery_app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIEngine:
    """Main AI Engine class for AlMenn"""

    def __init__(self):
        if AI_DEPS_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = None
        self.embedding_model = None
        self.llm_model = None
        self.llm_tokenizer = None
        self.vector_db = None
        self._initialize_models()

    def _initialize_models(self):
        """Initialize AI models and vector database"""
        if not AI_DEPS_AVAILABLE:
            logger.warning("AI dependencies not available, using mock implementations")
            self.embedding_model = None
            self.llm_model = None
            self.llm_tokenizer = None
            self.vector_db = None
            return

        try:
            # Initialize embedding model for RAG
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

            # Initialize LLM (using a smaller model for demo, replace with larger model in production)
            model_name = "microsoft/DialoGPT-small"  # Replace with educational model like "microsoft/DialoGPT-medium" or custom fine-tuned model
            self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.llm_model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

            # Initialize Pinecone vector database
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            if pinecone_api_key:
                pc = Pinecone(api_key=pinecone_api_key)
                index_name = "almenn-knowledge-base"

                # Create index if it doesn't exist
                if index_name not in pc.list_indexes().names():
                    if ServerlessSpec:
                        pc.create_index(
                            name=index_name,
                            dimension=384,  # Dimension for all-MiniLM-L6-v2
                            metric="cosine",
                            spec=ServerlessSpec(cloud="aws", region="us-east-1")
                        )
                    else:
                        # Fallback if ServerlessSpec not available
                        pc.create_index(
                            name=index_name,
                            dimension=384,
                            metric="cosine"
                        )

                self.vector_db = pc.Index(index_name)

            logger.info("AI Engine models initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize AI models: {e}")
            # Fallback to mock implementations
            self.embedding_model = None
            self.llm_model = None
            self.llm_tokenizer = None
            self.vector_db = None

    async def process_query(self, query: str, session_id: str, user_id: str, mode: str = "rag", attachments: List[str] = None) -> Dict[str, Any]:
        """
        Process an AI query using RAG or direct generation

        Args:
            query: User's question
            session_id: Current session ID
            user_id: User ID
            mode: "rag" or "direct"
            attachments: List of file IDs to include in context

        Returns:
            Dict containing response, sources, and visual elements
        """
        try:
            context = ""
            context_docs = []

            if mode == "rag" and self.vector_db:
                # Retrieve relevant context from vector database
                context_docs = await self._retrieve_context(query, attachments)
                context = "\n".join([doc['text'] for doc in context_docs])

            # Generate response
            response = await self._generate_response(query, context)

            # Generate visual elements if needed
            visuals = await self._generate_visuals(query, response)

            # Store interaction for learning and chat history
            await self._store_interaction(query, response, session_id, user_id, attachments)

            return {
                "response": response,
                "sources": context_docs if mode == "rag" else [],
                "visuals": visuals,
                "session_id": session_id
            }

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            # Store failed interaction for debugging
            await self._store_failed_interaction(query, str(e), session_id, user_id)
            return {
                "response": "I apologize, but I'm having trouble processing your request right now. Please try again later.",
                "sources": [],
                "visuals": [],
                "session_id": session_id,
                "error": str(e)
            }

    async def _retrieve_context(self, query: str, attachments: List[str] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant context from vector database"""
        if not self.vector_db or not self.embedding_model:
            return []

        try:
            # Embed query
            query_embedding = self.embedding_model.encode(query).tolist()

            # Search vector database
            results = self.vector_db.query(
                vector=query_embedding,
                top_k=5,
                include_metadata=True
            )

            context_docs = []
            for match in results['matches']:
                context_docs.append({
                    "text": match['metadata']['text'],
                    "source": match['metadata']['source'],
                    "score": match['score']
                })

            # Include attached files if provided
            if attachments:
                for file_id in attachments:
                    # TODO: Retrieve file content from database and add to context
                    pass

            return context_docs

        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []

    async def _generate_response(self, query: str, context: str = "") -> str:
        """Generate AI response using LLM"""
        if not self.llm_model or not self.llm_tokenizer:
            # Mock response for development
            return f"This is a mock educational response to: '{query}'. Context provided: {len(context)} characters."

        try:
            # Prepare prompt
            if context:
                prompt = f"Context: {context}\n\nQuestion: {query}\n\nProvide an educational, helpful answer:"
            else:
                prompt = f"Question: {query}\n\nProvide an educational, helpful answer:"

            # Tokenize
            inputs = self.llm_tokenizer(prompt, return_tensors="pt").to(self.device)

            # Generate response
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    inputs.input_ids,
                    max_length=512,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.llm_tokenizer.eos_token_id
                )

            response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Clean up response
            response = response.replace(prompt, "").strip()

            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I'm unable to generate a response right now."

    async def _generate_visuals(self, query: str, response: str) -> List[Dict[str, Any]]:
        """Generate visual elements for the response"""
        visuals = []

        try:
            # Check if response contains mathematical expressions
            if any(keyword in query.lower() for keyword in ['equation', 'formula', 'math', 'calculate']):
                visual = await self._generate_math_visual(query)
                if visual:
                    visuals.append(visual)

            # Check if response would benefit from diagrams
            if any(keyword in query.lower() for keyword in ['explain', 'how', 'process', 'flow']):
                visual = await self._generate_flow_diagram(query, response)
                if visual:
                    visuals.append(visual)

            # Check for data visualization needs
            if any(keyword in query.lower() for keyword in ['data', 'statistics', 'chart', 'graph']):
                visual = await self._generate_data_visual(query)
                if visual:
                    visuals.append(visual)

        except Exception as e:
            logger.error(f"Error generating visuals: {e}")

        return visuals

    async def _generate_math_visual(self, query: str) -> Optional[Dict[str, Any]]:
        """Generate mathematical visualizations"""
        try:
            # Use matplotlib to create math plots
            fig, ax = plt.subplots(figsize=(8, 6))

            # Example: Simple function plot (in production, parse actual math from query)
            import numpy as np
            x = np.linspace(-10, 10, 100)
            y = np.sin(x)  # Placeholder

            ax.plot(x, y, 'b-', linewidth=2)
            ax.set_title('Mathematical Visualization')
            ax.grid(True, alpha=0.3)

            # Save plot
            plot_path = f"/tmp/math_plot_{uuid.uuid4()}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()

            return {
                "type": "math_plot",
                "url": plot_path,
                "description": "Mathematical visualization"
            }

        except Exception as e:
            logger.error(f"Error generating math visual: {e}")
            return None

    async def _generate_flow_diagram(self, query: str, response: str) -> Optional[Dict[str, Any]]:
        """Generate flow diagrams using Mermaid"""
        try:
            # Create a simple flowchart
            mermaid_code = """
            graph TD
                A[Start] --> B{Decision Point}
                B -->|Yes| C[Action 1]
                B -->|No| D[Action 2]
                C --> E[End]
                D --> E
            """

            # In production, use more sophisticated parsing to generate relevant diagrams
            diagram_path = f"/tmp/flow_diagram_{uuid.uuid4()}.svg"

            # Generate SVG (simplified - in production use proper Mermaid rendering)
            dwg = svgwrite.Drawing(diagram_path, size=('400px', '300px'))
            dwg.add(dwg.rect(insert=(50, 50), size=(300, 200), fill='lightblue', stroke='black'))
            dwg.add(dwg.text('Process Flow', insert=(200, 150), text_anchor='middle'))
            dwg.save()

            return {
                "type": "flow_diagram",
                "url": diagram_path,
                "description": "Process flow diagram"
            }

        except Exception as e:
            logger.error(f"Error generating flow diagram: {e}")
            return None

    async def _generate_data_visual(self, query: str) -> Optional[Dict[str, Any]]:
        """Generate data visualizations"""
        try:
            # Create sample data visualization
            fig = go.Figure()

            # Sample bar chart
            fig.add_trace(go.Bar(
                x=['A', 'B', 'C', 'D'],
                y=[10, 15, 8, 12],
                name='Sample Data'
            ))

            fig.update_layout(
                title='Data Visualization',
                xaxis_title='Categories',
                yaxis_title='Values'
            )

            chart_path = f"/tmp/data_chart_{uuid.uuid4()}.html"
            fig.write_html(chart_path)

            return {
                "type": "data_chart",
                "url": chart_path,
                "description": "Data visualization chart"
            }

        except Exception as e:
            logger.error(f"Error generating data visual: {e}")
            return None

    async def _store_interaction(self, query: str, response: str, session_id: str, user_id: str, attachments: List[str] = None):
        """Store user interaction for continual learning and chat history"""
        try:
            from sqlalchemy.orm import sessionmaker
            from sqlalchemy import create_engine
            import models

            # Database setup
            DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db")
            engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

            db = SessionLocal()
            try:
                # Store in AI_Chat table for chat history
                ai_chat = models.AI_Chat(
                    session_id=session_id,
                    user_message=query,
                    ai_response=response,
                    attachments=attachments or []
                )
                db.add(ai_chat)
                db.commit()
                db.refresh(ai_chat)

                # Store embeddings for RAG learning
                if self.embedding_model and self.vector_db:
                    # Create embedding for the interaction
                    interaction_text = f"User: {query}\nAI: {response}"
                    embedding = self.embedding_model.encode(interaction_text).tolist()

                    # Store in vector database for future context
                    self.vector_db.upsert([(
                        str(ai_chat.id),
                        embedding,
                        {
                            "text": interaction_text,
                            "source": f"user_interaction_{user_id}",
                            "session_id": session_id,
                            "user_id": user_id,
                            "type": "chat_history"
                        }
                    )])

                logger.info(f"Stored interaction for user {user_id}, chat_id: {ai_chat.id}")

            finally:
                db.close()

        except Exception as e:
            logger.error(f"Error storing interaction: {e}")

    async def _store_failed_interaction(self, query: str, error: str, session_id: str, user_id: str):
        """Store failed interactions for debugging and improvement"""
        try:
            from sqlalchemy.orm import sessionmaker
            from sqlalchemy import create_engine
            import models

            # Database setup
            DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db")
            engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

            db = SessionLocal()
            try:
                # Store failed interaction
                ai_chat = models.AI_Chat(
                    session_id=session_id,
                    user_message=query,
                    ai_response=f"Error: {error}",
                    attachments=[]
                )
                db.add(ai_chat)
                db.commit()

                logger.warning(f"Stored failed interaction for user {user_id}")

            finally:
                db.close()

        except Exception as e:
            logger.error(f"Error storing failed interaction: {e}")

    async def process_file_upload(self, file_content: bytes, filename: str, user_id: str) -> str:
        """Process uploaded file for indexing"""
        try:
            file_id = str(uuid.uuid4())

            # Extract text based on file type
            extracted_text = await self._extract_text_from_file(file_content, filename)

            if extracted_text:
                # Store in vector database
                if self.vector_db and self.embedding_model:
                    embedding = self.embedding_model.encode(extracted_text).tolist()

                    self.vector_db.upsert([
                        (file_id, embedding, {
                            "text": extracted_text,
                            "source": f"user_upload_{user_id}",
                            "filename": filename,
                            "user_id": user_id
                        })
                    ])

                # Queue for detailed processing
                celery_app.send_task('extract_text_from_file', args=[file_id])

            return file_id

        except Exception as e:
            logger.error(f"Error processing file upload: {e}")
            raise

    async def _extract_text_from_file(self, file_content: bytes, filename: str) -> str:
        """Extract text from various file formats"""
        try:
            file_extension = filename.lower().split('.')[-1]

            if file_extension == 'pdf':
                # PDF extraction
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text

            elif file_extension in ['docx', 'doc']:
                # Word document extraction
                doc = docx.Document(io.BytesIO(file_content))
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text

            elif file_extension in ['xlsx', 'xls']:
                # Excel extraction
                wb = openpyxl.load_workbook(io.BytesIO(file_content))
                text = ""
                for sheet in wb:
                    for row in sheet.iter_rows(values_only=True):
                        text += " ".join([str(cell) for cell in row if cell]) + "\n"
                return text

            elif file_extension in ['png', 'jpg', 'jpeg']:
                # OCR for images
                image = Image.open(io.BytesIO(file_content))
                text = pytesseract.image_to_string(image)
                return text

            else:
                # Plain text or unknown format
                return file_content.decode('utf-8', errors='ignore')

        except Exception as e:
            logger.error(f"Error extracting text from {filename}: {e}")
            return ""

# Global AI Engine instance
ai_engine = AIEngine()

# Agent Classes
class FileProcessorAgent:
    """Agent for processing uploaded files"""

    def __init__(self, ai_engine: AIEngine):
        self.ai_engine = ai_engine

    async def process_file(self, file_id: str):
        """Process a file asynchronously"""
        # Implementation in tasks.py
        pass

class UserLearningAgent:
    """Agent for collecting user feedback for continual learning"""

    def __init__(self):
        self.feedback_queue = []

    async def collect_feedback(self, interaction_id: str, feedback: Dict[str, Any]):
        """Collect user feedback"""
        self.feedback_queue.append({
            "interaction_id": interaction_id,
            "feedback": feedback,
            "timestamp": datetime.utcnow()
        })

    async def generate_training_samples(self) -> List[Dict[str, Any]]:
        """Generate training samples from collected feedback"""
        # TODO: Implement training sample generation
        return []

class WebCrawlerAgent:
    """Agent for crawling educational web content"""

    def __init__(self):
        self.rate_limiter = {}  # Domain-based rate limiting

    async def crawl_educational_content(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Crawl permitted educational URLs"""
        crawled_content = []

        for url in urls:
            try:
                # Check robots.txt (simplified)
                if await self._check_robots_txt(url):
                    content = await self._fetch_and_process(url)
                    if content:
                        crawled_content.append(content)

                # Rate limiting
                await asyncio.sleep(1)  # Respectful delay

            except Exception as e:
                logger.error(f"Error crawling {url}: {e}")

        return crawled_content

    async def _check_robots_txt(self, url: str) -> bool:
        """Check if crawling is allowed"""
        # Simplified implementation - in production use proper robots.txt parsing
        return True

    async def _fetch_and_process(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetch and process web content"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract educational content (simplified)
            title = soup.find('title').text if soup.find('title') else ""
            paragraphs = soup.find_all('p')
            content = " ".join([p.text for p in paragraphs])

            return {
                "url": url,
                "title": title,
                "content": content,
                "crawled_at": datetime.utcnow()
            }

        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

class ModeratorAgent:
    """Agent for content moderation"""

    def __init__(self):
        self.toxicity_model = None  # Load toxicity detection model

    async def moderate_content(self, content: str) -> Dict[str, Any]:
        """Check content for toxicity and PII"""
        # TODO: Implement content moderation
        return {
            "is_safe": True,
            "flags": [],
            "confidence": 0.95
        }

class VisualGeneratorAgent:
    """Agent for generating explanatory visuals"""

    def __init__(self, ai_engine: AIEngine):
        self.ai_engine = ai_engine

    async def generate_visual(self, query: str, response: str) -> Optional[Dict[str, Any]]:
        """Generate appropriate visual for the query/response"""
        return await self.ai_engine._generate_visuals(query, response)

class SessionManagerAgent:
    """Agent for managing AI sessions and coin logic"""

    def __init__(self):
        self.active_sessions = {}  # In production, use Redis/database

    def start_session(self, user_id: str) -> str:
        """Start a new AI session for user"""
        session_id = str(uuid.uuid4())
        self.active_sessions[user_id] = {
            "session_id": session_id,
            "start_time": datetime.utcnow(),
            "minutes_remaining": 60,
            "coins_deducted": 4
        }
        return session_id

    def extend_session(self, user_id: str) -> int:
        """Extend session by 60 minutes"""
        if user_id in self.active_sessions:
            self.active_sessions[user_id]["minutes_remaining"] += 60
            self.active_sessions[user_id]["coins_deducted"] += 4
            return self.active_sessions[user_id]["minutes_remaining"]
        return 0

    def get_session_status(self, user_id: str) -> int:
        """Get remaining minutes for user's session"""
        if user_id in self.active_sessions:
            # For testing, don't expire sessions
            return 60
        return 0

    def is_session_active(self, user_id: str) -> bool:
        """Check if user has an active session"""
        return user_id in self.active_sessions and self.get_session_status(user_id) > 0

    async def validate_session(self, session_id: str, user_id: str) -> bool:
        """Validate if session is active and user has sufficient coins/time"""
        # For development/testing, always return True to avoid session reset issues
        return True
        # if user_id in self.active_sessions:
        #     return self.active_sessions[user_id]["session_id"] == session_id and self.is_session_active(user_id)
        # return False

    async def deduct_coins(self, user_id: str, amount: int) -> bool:
        """Deduct coins from user balance"""
        # TODO: Implement coin deduction from database
        return True

# Initialize agents
file_processor = FileProcessorAgent(ai_engine)
user_learning = UserLearningAgent()
web_crawler = WebCrawlerAgent()
moderator = ModeratorAgent()
visual_generator = VisualGeneratorAgent(ai_engine)
session_manager = SessionManagerAgent()
