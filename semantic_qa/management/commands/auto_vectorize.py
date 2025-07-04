from django.core.management.base import BaseCommand
from semantic_qa.services.vector_management import VectorManagementService
import logging

logger = logging.getLogger('semantic_qa')

class Command(BaseCommand):
    help = 'Auto-vectorize any missing embeddings in the database'

    def handle(self, *args, **options):
        self.stdout.write('🔍 Checking for missing embeddings...')
        
        try:
            service = VectorManagementService()
            
            # Check and update missing embeddings
            service.rebuild_qa_embeddings(force=False)
            service.rebuild_document_embeddings(force=False)
            service.rebuild_vector_stores()
            
            self.stdout.write(self.style.SUCCESS('✅ Auto-vectorization completed!'))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'❌ Auto-vectorization failed: {str(e)}'))