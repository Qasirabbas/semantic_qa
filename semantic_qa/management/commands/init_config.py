from django.core.management.base import BaseCommand
from semantic_qa.models import SystemConfig


class Command(BaseCommand):
    help = 'Initialize default system configuration'

    def handle(self, *args, **options):
        configs = [
            ('ollama_base_url', 'http://localhost:11434', 'Base URL for Ollama API'),
            ('ollama_model', 'qwen2.5', 'Ollama model for chat and processing'),
            ('embedding_model', 'nomic-embed-text', 'Ollama model for embeddings'),
            ('similarity_threshold', '0.3', 'Minimum similarity score for semantic search'),
            ('max_results', '10', 'Maximum number of search results to return'),
            ('supported_languages', 'en,zh,es,fr,de,ja', 'Comma-separated language codes'),
            ('enable_translation', 'True', 'Enable translation features'),
            ('default_language', 'en', 'Default language for the system'),
        ]
        
        created_count = 0
        for key, value, description in configs:
            config, created = SystemConfig.objects.get_or_create(
                key=key,
                defaults={'value': value, 'description': description}
            )
            if created:
                created_count += 1
                self.stdout.write(
                    self.style.SUCCESS(f'Created config: {key} = {value}')
                )
            else:
                self.stdout.write(f'Config already exists: {key} = {config.value}')
        
        self.stdout.write(
            self.style.SUCCESS(f'Configuration initialized. Created {created_count} new entries.')
        )