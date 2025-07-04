# semantic_qa/vector_management.py

import os
import logging
import numpy as np
from typing import List, Optional, Dict, Any
from django.conf import settings
from django.db import transaction
from django.utils import timezone
from django.db import models  # Add this import
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document as LangchainDocument

from .models import QAEntry, TextChunk, Document, ProcessingJob, SystemConfig  # Fixed import

logger = logging.getLogger('semantic_qa')

class VectorManagementService:
    """Service for managing vector embeddings and stores"""
    
    def __init__(self):
        self.embeddings = None
        self.vector_store_path = os.path.join(settings.BASE_DIR, 'vector_stores')
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize embedding model"""
        try:
            embedding_model = SystemConfig.get_config(
                'embedding_model', 
                'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
            )
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            logger.info(f"✅ Embeddings initialized with model: {embedding_model}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize embeddings: {str(e)}")
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"❌ Failed to generate embedding: {str(e)}")
            return []
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        try:
            embeddings = self.embeddings.embed_documents(texts)
            return embeddings
        except Exception as e:
            logger.error(f"❌ Failed to generate batch embeddings: {str(e)}")
            return []
    
    def rebuild_qa_embeddings(self, force: bool = False, batch_size: int = 50, job: Optional[ProcessingJob] = None):
        """Rebuild embeddings for all QA entries"""
        try:
            # Get QA entries that need embedding updates
            if force:
                qa_entries = QAEntry.objects.all()
            else:
                qa_entries = QAEntry.objects.filter(
                    models.Q(embedding__isnull=True) | models.Q(embedding=[])
                )
            
            total_count = qa_entries.count()
            if total_count == 0:
                logger.info("ℹ️ No QA entries need embedding updates")
                return
            
            logger.info(f"🔄 Processing {total_count} QA entries...")
            
            processed = 0
            
            # Process in batches
            for i in range(0, total_count, batch_size):
                batch = qa_entries[i:i + batch_size]
                texts = []
                entries = []
                
                for entry in batch:
                    # Create comprehensive text for embedding
                    text_parts = []
                    
                    if entry.sku:
                        text_parts.append(f"SKU: {entry.sku}")
                    if entry.question:
                        text_parts.append(f"Question: {entry.question}")
                    if entry.answer:
                        text_parts.append(f"Answer: {entry.answer}")
                    if entry.category:
                        text_parts.append(f"Category: {entry.category}")
                    if hasattr(entry, 'keywords') and entry.keywords:
                        text_parts.append(f"Keywords: {entry.keywords}")
                    
                    text = " | ".join(text_parts)
                    texts.append(text)
                    entries.append(entry)
                
                # Generate embeddings for batch
                embeddings = self.generate_embeddings_batch(texts)
                
                # Update database with embeddings
                with transaction.atomic():
                    for entry, embedding in zip(entries, embeddings):
                        if hasattr(entry, 'embedding'):
                            entry.embedding = embedding
                        if hasattr(entry, 'embedding_model'):
                            entry.embedding_model = getattr(self.embeddings, 'model_name', 'unknown')
                        entry.save()
                
                processed += len(batch)
                
                # Update job progress
                if job:
                    progress = int((processed / total_count) * 50)  # 50% for QA embeddings
                    job.update_progress(progress, f"Processed {processed}/{total_count} QA entries")
                
                logger.info(f"✅ Processed QA batch {i//batch_size + 1}: {processed}/{total_count}")
            
            logger.info(f"✅ QA embeddings rebuild completed: {processed} entries processed")
            
        except Exception as e:
            logger.error(f"❌ Failed to rebuild QA embeddings: {str(e)}")
            raise
    
    def rebuild_document_embeddings(self, force: bool = False, batch_size: int = 50, job: Optional[ProcessingJob] = None):
        """Rebuild embeddings for all document chunks"""
        try:
            # Get document chunks that need embedding updates
            if force:
                chunks = TextChunk.objects.all()
            else:
                chunks = TextChunk.objects.filter(
                    models.Q(embedding__isnull=True) | models.Q(embedding=[])
                )
            
            total_count = chunks.count()
            if total_count == 0:
                logger.info("ℹ️ No document chunks need embedding updates")
                return
            
            logger.info(f"🔄 Processing {total_count} document chunks...")
            
            processed = 0
            
            # Process in batches
            for i in range(0, total_count, batch_size):
                batch = chunks[i:i + batch_size]
                texts = []
                chunk_objects = []
                
                for chunk in batch:
                    # Create comprehensive text for embedding
                    text_parts = [chunk.text]
                    
                    if hasattr(chunk, 'context_before') and chunk.context_before:
                        text_parts.insert(0, chunk.context_before)
                    if hasattr(chunk, 'context_after') and chunk.context_after:
                        text_parts.append(chunk.context_after)
                    if hasattr(chunk, 'section_title') and chunk.section_title:
                        text_parts.insert(0, f"Section: {chunk.section_title}")
                    
                    text = " ".join(text_parts)
                    texts.append(text)
                    chunk_objects.append(chunk)
                
                # Generate embeddings for batch
                embeddings = self.generate_embeddings_batch(texts)
                
                # Update database with embeddings
                with transaction.atomic():
                    for chunk, embedding in zip(chunk_objects, embeddings):
                        if hasattr(chunk, 'embedding'):
                            chunk.embedding = embedding
                        if hasattr(chunk, 'embedding_model'):
                            chunk.embedding_model = getattr(self.embeddings, 'model_name', 'unknown')
                        chunk.save()
                
                processed += len(batch)
                
                # Update job progress
                if job:
                    progress = 50 + int((processed / total_count) * 40)  # 40% for document embeddings
                    job.update_progress(progress, f"Processed {processed}/{total_count} document chunks")
                
                logger.info(f"✅ Processed document batch {i//batch_size + 1}: {processed}/{total_count}")
            
            logger.info(f"✅ Document embeddings rebuild completed: {processed} chunks processed")
            
        except Exception as e:
            logger.error(f"❌ Failed to rebuild document embeddings: {str(e)}")
            raise
    
    def rebuild_vector_stores(self):
        """Rebuild FAISS vector stores from database embeddings"""
        try:
            os.makedirs(self.vector_store_path, exist_ok=True)
            logger.info("✅ Vector stores rebuild completed (placeholder)")
            
        except Exception as e:
            logger.error(f"❌ Failed to rebuild vector stores: {str(e)}")
            raise