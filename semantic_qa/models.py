# semantic_qa/models.py
from django.db import models
from django.utils import timezone
import hashlib
import json

class QAEntry(models.Model):
    """Model to store questions, answers, and image links from Excel"""
    sku = models.CharField(max_length=200, db_index=True, help_text="Product SKU")
    question = models.TextField(help_text="Original question")
    answer = models.TextField(help_text="Answer to the question")
    image_link = models.URLField(blank=True, null=True, help_text="Image URL")
    category = models.CharField(max_length=100, blank=True, help_text="Question category")
    keywords = models.TextField(blank=True, help_text="Extracted keywords for search")
    question_hash = models.CharField(max_length=64, help_text="SHA256 hash of SKU+question for uniqueness")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'qa_entries'
        indexes = [
            models.Index(fields=['sku']),
            models.Index(fields=['category']),
            models.Index(fields=['created_at']),
            models.Index(fields=['question_hash']),
        ]
        unique_together = [['question_hash']]
    
    def save(self, *args, **kwargs):
        if not self.question_hash:
            unique_string = f"{self.sku}|{self.question}"
            self.question_hash = hashlib.sha256(unique_string.encode('utf-8')).hexdigest()
        super().save(*args, **kwargs)
    
    def __str__(self):
        return f"{self.sku} - {self.question[:50]}..."

class Document(models.Model):
    """Model to store uploaded documents (PDF, images, links)"""
    DOCUMENT_TYPES = [
        ('pdf', 'PDF Document'),
        ('image', 'Image File'),
        ('link', 'Web Link'),
        ('excel', 'Excel File'),
    ]
    
    PROCESSING_STATUS = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    title = models.CharField(max_length=255, help_text="Document title")
    document_type = models.CharField(max_length=20, choices=DOCUMENT_TYPES)
    original_file = models.FileField(upload_to='documents/', blank=True, null=True)
    source_url = models.URLField(blank=True, null=True, help_text="For web links")
    extracted_text = models.TextField(blank=True, help_text="Full extracted text")
    
    # Processing metadata
    processing_status = models.CharField(max_length=20, choices=PROCESSING_STATUS, default='pending')
    processing_log = models.TextField(blank=True, help_text="Processing logs and errors")
    ocr_confidence = models.FloatField(null=True, blank=True, help_text="OCR confidence score")
    
    # Document metadata
    file_size = models.BigIntegerField(null=True, blank=True)
    page_count = models.IntegerField(null=True, blank=True)
    language_detected = models.CharField(max_length=10, blank=True)
    metadata = models.JSONField(default=dict, help_text="Additional metadata")
    
    # Categorization
    category = models.CharField(max_length=100, blank=True)
    tags = models.TextField(blank=True, help_text="Auto-generated tags")
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'documents'
        indexes = [
            models.Index(fields=['document_type']),
            models.Index(fields=['processing_status']),
            models.Index(fields=['category']),
            models.Index(fields=['created_at']),
        ]
    
    def __str__(self):
        return f"{self.title} ({self.document_type})"
    
    def get_file_extension(self):
        if self.original_file:
            return self.original_file.name.split('.')[-1].lower()
        return None

class TextChunk(models.Model):
    """Model to store text chunks for RAG retrieval"""
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='chunks')
    qa_entry = models.ForeignKey(QAEntry, on_delete=models.CASCADE, blank=True, null=True, related_name='chunks')
    
    # Chunk content
    text = models.TextField(help_text="Chunk text content")
    chunk_index = models.IntegerField(help_text="Order of chunk in document")
    chunk_size = models.IntegerField(help_text="Character count")
    
    # Context information
    page_number = models.IntegerField(null=True, blank=True)
    section_title = models.CharField(max_length=255, blank=True)
    context_before = models.TextField(blank=True, max_length=200)
    context_after = models.TextField(blank=True, max_length=200)
    
    # Vector embedding (stored as JSON array)
    embedding = models.JSONField(null=True, blank=True, help_text="Vector embedding")
    embedding_model = models.CharField(max_length=100, blank=True)
    
    # Search optimization
    keywords = models.TextField(blank=True)
    relevance_score = models.FloatField(default=0.0)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'text_chunks'
        indexes = [
            models.Index(fields=['document', 'chunk_index']),
            models.Index(fields=['qa_entry']),
            models.Index(fields=['page_number']),
            models.Index(fields=['relevance_score']),
        ]
    
    def __str__(self):
        try:
            source = self.document.title if self.document else f"QA: {self.qa_entry.sku if self.qa_entry else 'Unknown'}"
            return f"Chunk {self.chunk_index} from {source}"
        except:
            return f"Chunk {self.chunk_index} (ID: {self.id})"

class SemanticQuery(models.Model):
    """Model to log user queries and semantic matching results"""
    query_text = models.TextField(help_text="Original user query")
    processed_query = models.TextField(help_text="Processed/cleaned query")
    query_type = models.CharField(max_length=50, choices=[
        ('exact_sku', 'Exact SKU Match'),
        ('partial_sku', 'Partial SKU Match'),
        ('semantic', 'Semantic Match'),
        ('category', 'Category Match'),
        ('keyword', 'Keyword Match'),
        ('document', 'Document Search'),
        ('rag', 'RAG Generation'),
        ('hybrid', 'Hybrid Search'),
    ])
    
    # Search filters
    document_types_searched = models.JSONField(default=list, help_text="Document types included in search")
    use_rag = models.BooleanField(default=False, help_text="Whether RAG was used")
    
    # Results metadata
    total_results = models.IntegerField(default=0)
    rag_context_used = models.TextField(blank=True, help_text="Context provided to RAG")
    generated_answer = models.TextField(blank=True, help_text="RAG generated answer")
    
    user_ip = models.GenericIPAddressField(blank=True, null=True)
    user_agent = models.TextField(blank=True)
    response_time = models.FloatField(help_text="Response time in seconds")
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'semantic_queries'
        indexes = [
            models.Index(fields=['created_at']),
            models.Index(fields=['query_type']),
            models.Index(fields=['use_rag']),
        ]
    
    def __str__(self):
        return f"Query: {self.query_text[:50]}... ({self.query_type})"

class QueryMatch(models.Model):
    """Through model to track which QA entries/chunks matched a query"""
    query = models.ForeignKey(SemanticQuery, on_delete=models.CASCADE)
    qa_entry = models.ForeignKey(QAEntry, on_delete=models.CASCADE, blank=True, null=True)
    text_chunk = models.ForeignKey(TextChunk, on_delete=models.CASCADE, blank=True, null=True)
    
    relevance_score = models.FloatField(help_text="Semantic similarity score (0-1)")
    match_reason = models.CharField(max_length=100, help_text="Why this entry matched")
    rank_position = models.IntegerField(help_text="Position in search results")
    
    class Meta:
        db_table = 'query_matches'
        unique_together = [['query', 'qa_entry'], ['query', 'text_chunk']]

class Translation(models.Model):
    """Model to store translations for different languages"""
    source_text_hash = models.CharField(max_length=64, help_text="SHA256 hash of source text")
    source_text = models.TextField(help_text="Original text")
    translated_text = models.TextField(help_text="Translated text")
    source_language = models.CharField(max_length=10, default='en')
    target_language = models.CharField(max_length=10)
    translation_service = models.CharField(max_length=50, default='chatglm')
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'translations'
        unique_together = ['source_text_hash', 'source_language', 'target_language']
        indexes = [
            models.Index(fields=['source_language', 'target_language']),
            models.Index(fields=['source_text_hash']),
        ]
    
    def save(self, *args, **kwargs):
        if not self.source_text_hash:
            self.source_text_hash = hashlib.sha256(self.source_text.encode('utf-8')).hexdigest()
        super().save(*args, **kwargs)
    
    def __str__(self):
        return f"{self.source_language} -> {self.target_language}: {self.source_text[:50]}..."

class SystemConfig(models.Model):
    """Model to store system configuration"""
    key = models.CharField(max_length=100, unique=True)
    value = models.TextField()
    description = models.TextField(blank=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'system_config'
    
    def __str__(self):
        return f"{self.key}: {self.value}"
    
    @classmethod
    def get_config(cls, key, default=None):
        try:
            return cls.objects.get(key=key).value
        except cls.DoesNotExist:
            return default
    
    @classmethod
    def set_config(cls, key, value, description=""):
        config, created = cls.objects.get_or_create(
            key=key,
            defaults={'value': value, 'description': description}
        )
        if not created:
            config.value = value
            config.description = description
            config.save()
        return config

class ProcessingJob(models.Model):
    """Model to track background processing jobs"""
    JOB_TYPES = [
        ('document_processing', 'Document Processing'),
        ('text_extraction', 'Text Extraction'),
        ('embedding_generation', 'Embedding Generation'),
        ('batch_upload', 'Batch Upload'),
    ]
    
    JOB_STATUS = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('cancelled', 'Cancelled'),
    ]
    
    job_type = models.CharField(max_length=50, choices=JOB_TYPES)
    status = models.CharField(max_length=20, choices=JOB_STATUS, default='pending')
    
    # Job parameters
    input_data = models.JSONField(default=dict)
    output_data = models.JSONField(default=dict)
    
    # Progress tracking
    progress_percent = models.IntegerField(default=0)
    current_step = models.CharField(max_length=255, blank=True)
    total_steps = models.IntegerField(default=1)
    
    # Results
    success_count = models.IntegerField(default=0)
    error_count = models.IntegerField(default=0)
    error_details = models.TextField(blank=True)
    
    # Timing
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'processing_jobs'
        indexes = [
            models.Index(fields=['status']),
            models.Index(fields=['job_type']),
            models.Index(fields=['created_at']),
        ]
    
    def __str__(self):
        return f"{self.job_type} - {self.status}"
    
    def update_progress(self, percent, step="", save=True):
        self.progress_percent = percent
        if step:
            self.current_step = step
        if save:
            self.save()