from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from django.utils.translation import gettext_lazy as _
from .models import QAEntry, SemanticQuery, QueryMatch, Translation, SystemConfig

@admin.register(QAEntry)
class QAEntryAdmin(admin.ModelAdmin):
    list_display = ['sku', 'question_preview', 'category', 'has_image', 'created_at']
    list_filter = ['category', 'created_at', 'updated_at']
    search_fields = ['sku', 'question', 'answer', 'keywords']
    readonly_fields = ['created_at', 'updated_at']
    
    fieldsets = (
        (_('Basic Information'), {
            'fields': ('sku', 'category')
        }),
        (_('Content'), {
            'fields': ('question', 'answer', 'keywords')
        }),
        (_('Media'), {
            'fields': ('image_link',)
        }),
        (_('Timestamps'), {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def question_preview(self, obj):
        return obj.question[:100] + "..." if len(obj.question) > 100 else obj.question
    question_preview.short_description = _('Question')
    
    def has_image(self, obj):
        if obj.image_link:
            return format_html(
                '<a href="{}" target="_blank"><i class="fas fa-image text-success"></i></a>',
                obj.image_link
            )
        return format_html('<i class="fas fa-times text-muted"></i>')
    has_image.short_description = _('Image')
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related()

class QueryMatchInline(admin.TabularInline):
    model = QueryMatch
    extra = 0
    readonly_fields = ['qa_entry', 'relevance_score', 'match_reason']
    
    def has_add_permission(self, request, obj=None):
        return False

@admin.register(SemanticQuery)
class SemanticQueryAdmin(admin.ModelAdmin):
    list_display = ['query_preview', 'query_type', 'user_ip', 'response_time', 'created_at']
    list_filter = ['query_type', 'created_at']
    search_fields = ['query_text', 'processed_query', 'user_ip']
    readonly_fields = ['created_at']
    inlines = [QueryMatchInline]
    
    fieldsets = (
        (_('Query Information'), {
            'fields': ('query_text', 'processed_query', 'query_type')
        }),
        (_('User Information'), {
            'fields': ('user_ip', 'user_agent')
        }),
        (_('Performance'), {
            'fields': ('response_time',)
        }),
        (_('Timestamp'), {
            'fields': ('created_at',)
        }),
    )
    
    def query_preview(self, obj):
        return obj.query_text[:100] + "..." if len(obj.query_text) > 100 else obj.query_text
    query_preview.short_description = _('Query')
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related()

@admin.register(QueryMatch)
class QueryMatchAdmin(admin.ModelAdmin):
    list_display = ['query_preview', 'qa_entry_preview', 'relevance_score', 'match_reason']
    list_filter = ['match_reason', 'relevance_score']
    search_fields = ['query__query_text', 'qa_entry__sku', 'qa_entry__question']
    readonly_fields = ['query', 'qa_entry', 'relevance_score', 'match_reason']
    
    def query_preview(self, obj):
        return obj.query.query_text[:50] + "..." if len(obj.query.query_text) > 50 else obj.query.query_text
    query_preview.short_description = _('Query')
    
    def qa_entry_preview(self, obj):
        return f"{obj.qa_entry.sku} - {obj.qa_entry.question[:50]}..."
    qa_entry_preview.short_description = _('QA Entry')
    
    def has_add_permission(self, request):
        return False
    
    def has_change_permission(self, request, obj=None):
        return False

@admin.register(Translation)
class TranslationAdmin(admin.ModelAdmin):
    list_display = ['source_preview', 'source_language', 'target_language', 'translation_service', 'created_at']
    list_filter = ['source_language', 'target_language', 'translation_service', 'created_at']
    search_fields = ['source_text', 'translated_text']
    readonly_fields = ['created_at']
    
    fieldsets = (
        (_('Translation'), {
            'fields': ('source_text', 'translated_text')
        }),
        (_('Languages'), {
            'fields': ('source_language', 'target_language')
        }),
        (_('Service'), {
            'fields': ('translation_service',)
        }),
        (_('Timestamp'), {
            'fields': ('created_at',)
        }),
    )
    
    def source_preview(self, obj):
        return obj.source_text[:100] + "..." if len(obj.source_text) > 100 else obj.source_text
    source_preview.short_description = _('Source Text')

@admin.register(SystemConfig)
class SystemConfigAdmin(admin.ModelAdmin):
    list_display = ['key', 'value_preview', 'updated_at']
    search_fields = ['key', 'value', 'description']
    readonly_fields = ['updated_at']
    
    fieldsets = (
        (_('Configuration'), {
            'fields': ('key', 'value', 'description')
        }),
        (_('Timestamp'), {
            'fields': ('updated_at',)
        }),
    )
    
    def value_preview(self, obj):
        return obj.value[:100] + "..." if len(obj.value) > 100 else obj.value
    value_preview.short_description = _('Value')

# Customize admin site
admin.site.site_header = _('Semantic QA System Administration')
admin.site.site_title = _('Semantic QA Admin')
admin.site.index_title = _('Welcome to Semantic QA Administration')