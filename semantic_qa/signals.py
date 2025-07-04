from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from django.conf import settings
from .models import QAEntry, TextChunk, Document
import logging

logger = logging.getLogger('semantic_qa')

@receiver(post_save, sender=QAEntry)
def qa_entry_saved(sender, instance, created, **kwargs):
    """Update embedding when QA entry is saved"""
    logger.info(f"🔄 QA Entry {'created' if created else 'updated'}: {instance.id}")
    
    # Only process if embedding field exists and is empty
    if hasattr(instance, 'embedding') and not instance.embedding:
        try:
            from .vector_management import VectorManagementService
            vector_service = VectorManagementService()
            
            # Generate text for embedding
            text_parts = []
            if instance.sku:
                text_parts.append(f"SKU: {instance.sku}")
            if instance.question:
                text_parts.append(f"Question: {instance.question}")
            if instance.answer:
                text_parts.append(f"Answer: {instance.answer}")
            
            text = " | ".join(text_parts)
            embedding = vector_service.generate_embedding(text)
            
            if embedding:
                instance.embedding = embedding
                instance.embedding_model = getattr(vector_service.embeddings, 'model_name', 'unknown')
                instance.save(update_fields=['embedding', 'embedding_model'])
                logger.info(f"✅ Generated embedding for QA entry {instance.id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate embedding for QA entry {instance.id}: {str(e)}")

@receiver(post_save, sender=Document)
def document_saved(sender, instance, created, **kwargs):
    """Trigger re-vectorization when document is saved"""
    if instance.processing_status == 'completed':
        logger.info(f"🔄 Document processing completed: {instance.id}")
        try:
            from .vector_management import VectorManagementService
            vector_service = VectorManagementService()
            vector_service.auto_rebuild_after_upload(document_id=instance.id)
            logger.info(f"✅ Auto re-vectorization triggered for document {instance.id}")
        except Exception as e:
            logger.error(f"❌ Auto re-vectorization failed for document {instance.id}: {str(e)}")