# semantic_qa/urls.py
from django.urls import path
from . import views

app_name = 'semantic_qa'

urlpatterns = [
    # Main interfaces
    path('', views.index, name='index'),
    
    # Enhanced search endpoints
    path('search/', views.enhanced_search_results, name='enhanced_search_results'),
    path('search/legacy/', views.search_results, name='search_results'),  # Legacy redirect
    
    # API endpoints
    path('api/search/', views.enhanced_search_api, name='enhanced_search_api'),
    path('api/search/legacy/', views.search_api, name='search_api'),  # Legacy compatibility
    path('api/translate/', views.translate_api, name='translate_api'),
    
    # Document management
    path('api/upload-document/', views.upload_document, name='upload_document'),
    path('api/process-document/<int:document_id>/', views.process_document_api, name='process_document_api'),
    path('api/processing-status/<int:job_id>/', views.processing_status_api, name='processing_status_api'),
    
    # Excel upload (existing)
    path('api/upload/', views.upload_excel, name='upload_excel'),
    path('api/download-template/', views.download_template, name='download_template'),
    
    # Document management pages
    path('upload-documents/', views.upload_document_page, name='upload_document_page'),
    path('manage-documents/', views.manage_documents, name='manage_documents'),
    path('document/<int:document_id>/', views.document_detail, name='document_detail'),
    path('document/<int:document_id>/delete/', views.delete_document, name='delete_document'),
    
    # Admin interfaces (enhanced)
    path('admin-dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('manage-entries/', views.manage_entries, name='manage_entries'),
    path('manage-entries/<int:entry_id>/', views.edit_entry, name='edit_entry'),
    path('manage-entries/<int:entry_id>/delete/', views.delete_entry, name='delete_entry'),
    path('upload-excel/', views.upload_excel_page, name='upload_excel_page'),
    
    # Analytics and reporting
    path('analytics/', views.analytics_dashboard, name='analytics_dashboard'),
    path('query-logs/', views.query_logs, name='query_logs'),
    path('export-data/', views.export_data, name='export_data'),
    
    # Image handling
    path('image-proxy/<path:image_url>/', views.image_proxy, name='image_proxy'),
    
    # Language switching
    path('set-language/', views.set_language, name='set_language'),
]