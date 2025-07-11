# Generated by Django 4.2.23 on 2025-06-17 04:02

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Document',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(help_text='Document title', max_length=255)),
                ('document_type', models.CharField(choices=[('pdf', 'PDF Document'), ('image', 'Image File'), ('link', 'Web Link'), ('excel', 'Excel File')], max_length=20)),
                ('original_file', models.FileField(blank=True, null=True, upload_to='documents/')),
                ('source_url', models.URLField(blank=True, help_text='For web links', null=True)),
                ('extracted_text', models.TextField(blank=True, help_text='Full extracted text')),
                ('processing_status', models.CharField(choices=[('pending', 'Pending'), ('processing', 'Processing'), ('completed', 'Completed'), ('failed', 'Failed')], default='pending', max_length=20)),
                ('processing_log', models.TextField(blank=True, help_text='Processing logs and errors')),
                ('ocr_confidence', models.FloatField(blank=True, help_text='OCR confidence score', null=True)),
                ('file_size', models.BigIntegerField(blank=True, null=True)),
                ('page_count', models.IntegerField(blank=True, null=True)),
                ('language_detected', models.CharField(blank=True, max_length=10)),
                ('metadata', models.JSONField(default=dict, help_text='Additional metadata')),
                ('category', models.CharField(blank=True, max_length=100)),
                ('tags', models.TextField(blank=True, help_text='Auto-generated tags')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
            options={
                'db_table': 'documents',
            },
        ),
        migrations.CreateModel(
            name='ProcessingJob',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('job_type', models.CharField(choices=[('document_processing', 'Document Processing'), ('text_extraction', 'Text Extraction'), ('embedding_generation', 'Embedding Generation'), ('batch_upload', 'Batch Upload')], max_length=50)),
                ('status', models.CharField(choices=[('pending', 'Pending'), ('running', 'Running'), ('completed', 'Completed'), ('failed', 'Failed'), ('cancelled', 'Cancelled')], default='pending', max_length=20)),
                ('input_data', models.JSONField(default=dict)),
                ('output_data', models.JSONField(default=dict)),
                ('progress_percent', models.IntegerField(default=0)),
                ('current_step', models.CharField(blank=True, max_length=255)),
                ('total_steps', models.IntegerField(default=1)),
                ('success_count', models.IntegerField(default=0)),
                ('error_count', models.IntegerField(default=0)),
                ('error_details', models.TextField(blank=True)),
                ('started_at', models.DateTimeField(blank=True, null=True)),
                ('completed_at', models.DateTimeField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
            options={
                'db_table': 'processing_jobs',
            },
        ),
        migrations.CreateModel(
            name='QAEntry',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('sku', models.CharField(db_index=True, help_text='Product SKU', max_length=200)),
                ('question', models.TextField(help_text='Original question')),
                ('answer', models.TextField(help_text='Answer to the question')),
                ('image_link', models.URLField(blank=True, help_text='Image URL', null=True)),
                ('category', models.CharField(blank=True, help_text='Question category', max_length=100)),
                ('keywords', models.TextField(blank=True, help_text='Extracted keywords for search')),
                ('question_hash', models.CharField(help_text='SHA256 hash of SKU+question for uniqueness', max_length=64)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
            options={
                'db_table': 'qa_entries',
            },
        ),
        migrations.CreateModel(
            name='SystemConfig',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('key', models.CharField(max_length=100, unique=True)),
                ('value', models.TextField()),
                ('description', models.TextField(blank=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
            options={
                'db_table': 'system_config',
            },
        ),
        migrations.CreateModel(
            name='Translation',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('source_text_hash', models.CharField(help_text='SHA256 hash of source text', max_length=64)),
                ('source_text', models.TextField(help_text='Original text')),
                ('translated_text', models.TextField(help_text='Translated text')),
                ('source_language', models.CharField(default='en', max_length=10)),
                ('target_language', models.CharField(max_length=10)),
                ('translation_service', models.CharField(default='chatglm', max_length=50)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
            options={
                'db_table': 'translations',
                'indexes': [models.Index(fields=['source_language', 'target_language'], name='translation_source__8f4214_idx'), models.Index(fields=['source_text_hash'], name='translation_source__9bdd08_idx')],
                'unique_together': {('source_text_hash', 'source_language', 'target_language')},
            },
        ),
        migrations.CreateModel(
            name='TextChunk',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('text', models.TextField(help_text='Chunk text content')),
                ('chunk_index', models.IntegerField(help_text='Order of chunk in document')),
                ('chunk_size', models.IntegerField(help_text='Character count')),
                ('page_number', models.IntegerField(blank=True, null=True)),
                ('section_title', models.CharField(blank=True, max_length=255)),
                ('context_before', models.TextField(blank=True, max_length=200)),
                ('context_after', models.TextField(blank=True, max_length=200)),
                ('embedding', models.JSONField(blank=True, help_text='Vector embedding', null=True)),
                ('embedding_model', models.CharField(blank=True, max_length=100)),
                ('keywords', models.TextField(blank=True)),
                ('relevance_score', models.FloatField(default=0.0)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('document', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='chunks', to='semantic_qa.document')),
                ('qa_entry', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='chunks', to='semantic_qa.qaentry')),
            ],
            options={
                'db_table': 'text_chunks',
            },
        ),
        migrations.CreateModel(
            name='SemanticQuery',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('query_text', models.TextField(help_text='Original user query')),
                ('processed_query', models.TextField(help_text='Processed/cleaned query')),
                ('query_type', models.CharField(choices=[('exact_sku', 'Exact SKU Match'), ('partial_sku', 'Partial SKU Match'), ('semantic', 'Semantic Match'), ('category', 'Category Match'), ('keyword', 'Keyword Match'), ('document', 'Document Search'), ('rag', 'RAG Generation'), ('hybrid', 'Hybrid Search')], max_length=50)),
                ('document_types_searched', models.JSONField(default=list, help_text='Document types included in search')),
                ('use_rag', models.BooleanField(default=False, help_text='Whether RAG was used')),
                ('total_results', models.IntegerField(default=0)),
                ('rag_context_used', models.TextField(blank=True, help_text='Context provided to RAG')),
                ('generated_answer', models.TextField(blank=True, help_text='RAG generated answer')),
                ('user_ip', models.GenericIPAddressField(blank=True, null=True)),
                ('user_agent', models.TextField(blank=True)),
                ('response_time', models.FloatField(help_text='Response time in seconds')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
            options={
                'db_table': 'semantic_queries',
                'indexes': [models.Index(fields=['created_at'], name='semantic_qu_created_8c250d_idx'), models.Index(fields=['query_type'], name='semantic_qu_query_t_c83fc0_idx'), models.Index(fields=['use_rag'], name='semantic_qu_use_rag_a57276_idx')],
            },
        ),
        migrations.CreateModel(
            name='QueryMatch',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('relevance_score', models.FloatField(help_text='Semantic similarity score (0-1)')),
                ('match_reason', models.CharField(help_text='Why this entry matched', max_length=100)),
                ('rank_position', models.IntegerField(help_text='Position in search results')),
                ('qa_entry', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='semantic_qa.qaentry')),
                ('query', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='semantic_qa.semanticquery')),
                ('text_chunk', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='semantic_qa.textchunk')),
            ],
            options={
                'db_table': 'query_matches',
            },
        ),
        migrations.AddIndex(
            model_name='qaentry',
            index=models.Index(fields=['sku'], name='qa_entries_sku_fe85ad_idx'),
        ),
        migrations.AddIndex(
            model_name='qaentry',
            index=models.Index(fields=['category'], name='qa_entries_categor_13d46c_idx'),
        ),
        migrations.AddIndex(
            model_name='qaentry',
            index=models.Index(fields=['created_at'], name='qa_entries_created_338380_idx'),
        ),
        migrations.AddIndex(
            model_name='qaentry',
            index=models.Index(fields=['question_hash'], name='qa_entries_questio_8a6378_idx'),
        ),
        migrations.AlterUniqueTogether(
            name='qaentry',
            unique_together={('question_hash',)},
        ),
        migrations.AddIndex(
            model_name='processingjob',
            index=models.Index(fields=['status'], name='processing__status_96bb49_idx'),
        ),
        migrations.AddIndex(
            model_name='processingjob',
            index=models.Index(fields=['job_type'], name='processing__job_typ_d1b57f_idx'),
        ),
        migrations.AddIndex(
            model_name='processingjob',
            index=models.Index(fields=['created_at'], name='processing__created_7e276e_idx'),
        ),
        migrations.AddIndex(
            model_name='document',
            index=models.Index(fields=['document_type'], name='documents_documen_fc21d0_idx'),
        ),
        migrations.AddIndex(
            model_name='document',
            index=models.Index(fields=['processing_status'], name='documents_process_e1e618_idx'),
        ),
        migrations.AddIndex(
            model_name='document',
            index=models.Index(fields=['category'], name='documents_categor_f8ad6f_idx'),
        ),
        migrations.AddIndex(
            model_name='document',
            index=models.Index(fields=['created_at'], name='documents_created_3c6eaa_idx'),
        ),
        migrations.AddIndex(
            model_name='textchunk',
            index=models.Index(fields=['document', 'chunk_index'], name='text_chunks_documen_f82e39_idx'),
        ),
        migrations.AddIndex(
            model_name='textchunk',
            index=models.Index(fields=['qa_entry'], name='text_chunks_qa_entr_bc8713_idx'),
        ),
        migrations.AddIndex(
            model_name='textchunk',
            index=models.Index(fields=['page_number'], name='text_chunks_page_nu_f80a15_idx'),
        ),
        migrations.AddIndex(
            model_name='textchunk',
            index=models.Index(fields=['relevance_score'], name='text_chunks_relevan_990219_idx'),
        ),
        migrations.AlterUniqueTogether(
            name='querymatch',
            unique_together={('query', 'text_chunk'), ('query', 'qa_entry')},
        ),
    ]
