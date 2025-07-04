# semantic_qa/management/commands/import_excel.py
import os
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from semantic_qa.models import QAEntry
from semantic_qa.utils import parse_excel_file

class Command(BaseCommand):
    help = 'Import QA data from Excel file'

    def add_arguments(self, parser):
        parser.add_argument('file_path', type=str, help='Path to Excel file')
        parser.add_argument(
            '--overwrite',
            action='store_true',
            help='Overwrite existing entries with same SKU+Question',
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be imported without actually importing',
        )

    def handle(self, *args, **options):
        file_path = options['file_path']
        overwrite = options['overwrite']
        dry_run = options['dry_run']

        if not os.path.exists(file_path):
            raise CommandError(f'File "{file_path}" does not exist.')

        if not file_path.lower().endswith(('.xlsx', '.xls')):
            raise CommandError('File must be an Excel file (.xlsx or .xls)')

        self.stdout.write(f'Processing file: {file_path}')

        try:
            # Parse Excel file
            with open(file_path, 'rb') as f:
                from django.core.files.uploadedfile import SimpleUploadedFile
                uploaded_file = SimpleUploadedFile(
                    name=os.path.basename(file_path),
                    content=f.read()
                )
                qa_entries = parse_excel_file(uploaded_file)

            if not qa_entries:
                self.stdout.write(
                    self.style.WARNING('No valid data found in Excel file')
                )
                return

            self.stdout.write(f'Found {len(qa_entries)} entries to process')

            if dry_run:
                self.stdout.write(self.style.WARNING('DRY RUN MODE - No data will be saved'))
                for i, entry in enumerate(qa_entries[:5], 1):  # Show first 5 entries
                    self.stdout.write(f'{i}. SKU: {entry["sku"]}, Question: {entry["question"][:50]}...')
                if len(qa_entries) > 5:
                    self.stdout.write(f'... and {len(qa_entries) - 5} more entries')
                return

            # Save to database
            created_count = 0
            updated_count = 0
            error_count = 0

            for entry_data in qa_entries:
                try:
                    if overwrite:
                        entry, created = QAEntry.objects.update_or_create(
                            sku=entry_data['sku'],
                            question=entry_data['question'],
                            defaults={
                                'answer': entry_data['answer'],
                                'image_link': entry_data.get('image_link', ''),
                                'category': entry_data.get('category', ''),
                                'keywords': entry_data.get('keywords', '')
                            }
                        )
                    else:
                        entry, created = QAEntry.objects.get_or_create(
                            sku=entry_data['sku'],
                            question=entry_data['question'],
                            defaults={
                                'answer': entry_data['answer'],
                                'image_link': entry_data.get('image_link', ''),
                                'category': entry_data.get('category', ''),
                                'keywords': entry_data.get('keywords', '')
                            }
                        )

                    if created:
                        created_count += 1
                        self.stdout.write(f'Created: {entry.sku} - {entry.question[:50]}...')
                    else:
                        if overwrite:
                            updated_count += 1
                            self.stdout.write(f'Updated: {entry.sku} - {entry.question[:50]}...')
                        else:
                            self.stdout.write(
                                self.style.WARNING(f'Skipped (exists): {entry.sku} - {entry.question[:50]}...')
                            )

                except Exception as e:
                    error_count += 1
                    self.stdout.write(
                        self.style.ERROR(f'Error processing entry: {str(e)}')
                    )

            # Summary
            self.stdout.write(
                self.style.SUCCESS(
                    f'\nImport completed!\n'
                    f'Created: {created_count}\n'
                    f'Updated: {updated_count}\n'
                    f'Errors: {error_count}\n'
                    f'Total processed: {len(qa_entries)}'
                )
            )

        except Exception as e:
            raise CommandError(f'Error processing file: {str(e)}')

# semantic_qa/management/commands/setup_ollama.py
import requests
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from semantic_qa.models import SystemConfig

class Command(BaseCommand):
    help = 'Setup and verify Ollama connection and models'

    def add_arguments(self, parser):
        parser.add_argument(
            '--url',
            type=str,
            default='http://localhost:11434',
            help='Ollama server URL (default: http://localhost:11434)',
        )
        parser.add_argument(
            '--model',
            type=str,
            default='qwen2.5',
            help='LLM model name (default: qwen2.5)',
        )
        parser.add_argument(
            '--embedding-model',
            type=str,
            default='nomic-embed-text',
            help='Embedding model name (default: nomic-embed-text)',
        )
        parser.add_argument(
            '--pull-models',
            action='store_true',
            help='Pull models if they are not available',
        )

    def handle(self, *args, **options):
        ollama_url = options['url']
        model_name = options['model']
        embedding_model = options['embedding_model']
        pull_models = options['pull_models']

        self.stdout.write(f'Checking Ollama server at: {ollama_url}')

        # Test connection
        try:
            response = requests.get(f'{ollama_url}/api/tags', timeout=10)
            response.raise_for_status()
            self.stdout.write(self.style.SUCCESS('✓ Ollama server is running'))
        except requests.exceptions.RequestException as e:
            raise CommandError(f'Cannot connect to Ollama server: {str(e)}')

        # Get available models
        try:
            models_response = requests.get(f'{ollama_url}/api/tags', timeout=10)
            models_data = models_response.json()
            available_models = [model['name'] for model in models_data.get('models', [])]
            
            self.stdout.write(f'Available models: {", ".join(available_models)}')
            
        except Exception as e:
            self.stdout.write(self.style.WARNING(f'Could not fetch model list: {str(e)}'))
            available_models = []

        # Check LLM model
        if model_name in available_models:
            self.stdout.write(self.style.SUCCESS(f'✓ LLM model "{model_name}" is available'))
        else:
            if pull_models:
                self.stdout.write(f'Pulling LLM model: {model_name}')
                self._pull_model(ollama_url, model_name)
            else:
                self.stdout.write(
                    self.style.WARNING(f'⚠ LLM model "{model_name}" not found. Use --pull-models to download.')
                )

        # Check embedding model
        if embedding_model in available_models:
            self.stdout.write(self.style.SUCCESS(f'✓ Embedding model "{embedding_model}" is available'))
        else:
            if pull_models:
                self.stdout.write(f'Pulling embedding model: {embedding_model}')
                self._pull_model(ollama_url, embedding_model)
            else:
                self.stdout.write(
                    self.style.WARNING(f'⚠ Embedding model "{embedding_model}" not found. Use --pull-models to download.')
                )

        # Test LLM
        self.stdout.write('Testing LLM generation...')
        try:
            test_response = requests.post(
                f'{ollama_url}/api/generate',
                json={
                    'model': model_name,
                    'prompt': 'Hello, respond with just "OK"',
                    'stream': False
                },
                timeout=30
            )
            test_response.raise_for_status()
            response_text = test_response.json().get('response', '').strip()
            self.stdout.write(self.style.SUCCESS(f'✓ LLM test successful: {response_text}'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'✗ LLM test failed: {str(e)}'))

        # Test embeddings
        self.stdout.write('Testing embedding generation...')
        try:
            embed_response = requests.post(
                f'{ollama_url}/api/embeddings',
                json={
                    'model': embedding_model,
                    'prompt': 'test embedding'
                },
                timeout=30
            )
            embed_response.raise_for_status()
            embedding = embed_response.json().get('embedding', [])
            if embedding:
                self.stdout.write(self.style.SUCCESS(f'✓ Embedding test successful (dimension: {len(embedding)})'))
            else:
                self.stdout.write(self.style.ERROR('✗ Embedding test failed: no embedding returned'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'✗ Embedding test failed: {str(e)}'))

        # Update system configuration
        SystemConfig.set_config('ollama_base_url', ollama_url)
        SystemConfig.set_config('ollama_model', model_name)
        SystemConfig.set_config('embedding_model', embedding_model)

        self.stdout.write(self.style.SUCCESS('\nOllama setup completed!'))
        self.stdout.write('Configuration saved to database.')

    def _pull_model(self, ollama_url, model_name):
        """Pull a model from Ollama"""
        try:
            self.stdout.write(f'Pulling model {model_name}... (this may take a while)')
            
            response = requests.post(
                f'{ollama_url}/api/pull',
                json={'name': model_name},
                stream=True,
                timeout=300  # 5 minutes timeout
            )
            response.raise_for_status()
            
            # Stream the response to show progress
            for line in response.iter_lines():
                if line:
                    try:
                        import json
                        data = json.loads(line.decode('utf-8'))
                        if 'status' in data:
                            self.stdout.write(f'  {data["status"]}', ending='\r')
                    except:
                        pass
            
            self.stdout.write(self.style.SUCCESS(f'\n✓ Successfully pulled model: {model_name}'))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'\n✗ Failed to pull model {model_name}: {str(e)}'))

# semantic_qa/management/commands/test_search.py
from django.core.management.base import BaseCommand
from semantic_qa.services import SemanticSearchService

class Command(BaseCommand):
    help = 'Test semantic search functionality'

    def add_arguments(self, parser):
        parser.add_argument('query', type=str, help='Search query to test')
        parser.add_argument(
            '--language',
            type=str,
            default='en',
            help='Target language for results (default: en)',
        )

    def handle(self, *args, **options):
        query = options['query']
        language = options['language']

        self.stdout.write(f'Testing search for: "{query}"')
        self.stdout.write(f'Language: {language}')
        self.stdout.write('-' * 50)

        semantic_service = SemanticSearchService()
        
        try:
            results = semantic_service.search_qa_entries(query, '127.0.0.1', 'Test CLI')
            
            self.stdout.write(f'Query type: {results["query_type"]}')
            self.stdout.write(f'Response time: {results["response_time"]:.3f}s')
            self.stdout.write(f'Total results: {results["total_results"]}')
            self.stdout.write('-' * 50)
            
            if results['results']:
                for i, result in enumerate(results['results'], 1):
                    entry = result['entry']
                    score = result['score']
                    match_type = result['match_type']
                    
                    self.stdout.write(f'\n{i}. SKU: {entry.sku}')
                    self.stdout.write(f'   Question: {entry.question}')
                    self.stdout.write(f'   Answer: {entry.answer[:100]}...')
                    self.stdout.write(f'   Score: {score:.3f} ({match_type})')
                    if entry.image_link:
                        self.stdout.write(f'   Image: {entry.image_link}')
                    self.stdout.write(f'   Category: {entry.category}')
            else:
                self.stdout.write('No results found.')
                
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Search failed: {str(e)}'))