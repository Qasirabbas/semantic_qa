from django.core.management.base import BaseCommand
from django.db import transaction
from semantic_qa.models import QAEntry, TextChunk, ProcessingJob
from semantic_qa.vector_management import VectorManagementService  # Fixed import
import logging

logger = logging.getLogger('semantic_qa')

class Command(BaseCommand):
    help = 'Rebuild all vector embeddings for QA entries and documents'

    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force rebuild even if embeddings exist'
        )
        parser.add_argument(
            '--type',
            choices=['qa', 'documents', 'all'],
            default='all',
            help='What to rebuild'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=50,
            help='Batch size for processing'
        )

    def handle(self, *args, **options):
        force = options['force']
        rebuild_type = options['type']
        batch_size = options['batch_size']
        
        self.stdout.write('🚀 Starting vector rebuilding process...')
        
        # Create processing job
        job = ProcessingJob.objects.create(
            job_type='embedding_generation',
            status='running',
            input_data={
                'force': force,
                'type': rebuild_type,
                'batch_size': batch_size
            }
        )
        
        try:
            service = VectorManagementService()
            
            if rebuild_type in ['qa', 'all']:
                self.stdout.write('📝 Rebuilding QA embeddings...')
                service.rebuild_qa_embeddings(force=force, batch_size=batch_size, job=job)
            
            if rebuild_type in ['documents', 'all']:
                self.stdout.write('📄 Rebuilding document embeddings...')
                service.rebuild_document_embeddings(force=force, batch_size=batch_size, job=job)
            
            # Rebuild vector stores
            self.stdout.write('🔄 Rebuilding vector stores...')
            service.rebuild_vector_stores()
            
            job.status = 'completed'
            job.progress_percent = 100
            job.save()
            
            self.stdout.write(self.style.SUCCESS('✅ Vector rebuilding completed successfully!'))
            
        except Exception as e:
            job.status = 'failed'
            job.error_details = str(e)
            job.save()
            
            self.stdout.write(self.style.ERROR(f'❌ Vector rebuilding failed: {str(e)}'))
            raise