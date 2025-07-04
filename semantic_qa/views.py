# semantic_qa/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse, Http404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.paginator import Paginator
from django.contrib import messages
from django.db.models import Q, Count, Avg
from django.urls import reverse
from django.utils import translation, timezone
from django.utils.translation import gettext as _
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import json
import time
import requests
from io import BytesIO
import logging
import os
import tempfile
from datetime import datetime, timedelta
import pandas as pd

from .models import (QAEntry, SemanticQuery, Translation, SystemConfig, 
                    QueryMatch, Document, TextChunk, ProcessingJob)
from .rag_service import RAGService
from .document_processor import DocumentProcessor
from .translations_service import TranslationService
from .utils import (get_client_ip, parse_excel_file, format_enhanced_search_results, 
                   format_enhanced_search_results, generate_excel_template, log_user_activity)
from .forms import (QAEntryForm, ExcelUploadForm, DocumentUploadForm, 
                   BatchDocumentUploadForm, EnhancedSearchForm, FilterForm, SystemConfigForm)
from .vector_management import VectorManagementService
logger = logging.getLogger('semantic_qa')

# Initialize services
rag_service = RAGService()
doc_processor = DocumentProcessor()
translation_service = TranslationService()

def index(request):
    """Main search interface with enhanced features"""
    supported_languages = SystemConfig.get_config('supported_languages', 'en,zh,es,fr,de,ja').split(',')
    default_language = SystemConfig.get_config('default_language', 'en')
    
    # Get comprehensive statistics
    total_qa_entries = QAEntry.objects.count()
    total_documents = Document.objects.filter(processing_status='completed').count()
    total_queries = SemanticQuery.objects.count()
    total_chunks = TextChunk.objects.count()
    
    # Get recent popular searches
    popular_queries = SemanticQuery.objects.values('processed_query')\
        .annotate(count=Count('id'))\
        .order_by('-count')[:5]
    
    # Get document type statistics
    doc_stats = Document.objects.filter(processing_status='completed')\
        .values('document_type')\
        .annotate(count=Count('id'))
    
    # Get recent successful searches with RAG
    recent_rag_queries = SemanticQuery.objects.filter(
        use_rag=False,
        generated_answer__isnull=False
    ).exclude(generated_answer='').order_by('-created_at')[:3]
    
    context = {
        'supported_languages': supported_languages,
        'default_language': default_language,
        'popular_queries': popular_queries,
        'total_qa_entries': total_qa_entries,
        'total_documents': total_documents,
        'total_queries': total_queries,
        'total_chunks': total_chunks,
        'doc_stats': doc_stats,
        'recent_rag_queries': recent_rag_queries,
        'rag_enabled': SystemConfig.get_config('enable_rag', 'True').lower() == 'true',
        'page_title': _('智能语义搜索系统'),
    }
    return render(request, 'semantic_qa/index.html', context)

@csrf_exempt
@require_http_methods(["POST"])
def enhanced_search_api(request):
    """Enhanced API endpoint for semantic search with RAG"""
    try:
        data = json.loads(request.body)
        query = data.get('query', '').strip()
        language = data.get('language', 'en')
        document_types = data.get('document_types', ['pdf', 'image', 'link'])
        search_qa_entries = data.get('search_qa_entries', True)
        search_documents = data.get('search_documents', True)
        use_rag = data.get('use_rag', False)
        max_results = min(data.get('max_results', 20), 50)  # Cap at 50
        
        if not query:
            return JsonResponse({'error': _('Query is required')}, status=400)
        
        # Get client information
        user_ip = get_client_ip(request)
        user_agent = request.META.get('HTTP_USER_AGENT', '')
        
        # Log activity
        log_user_activity(user_ip, 'enhanced_search', 
                         f"Query: {query}, RAG: {use_rag}, Types: {document_types}")
        
        # Adjust document types based on search preferences
        if not search_documents:
            document_types = []
        
        # Perform enhanced search
        results = rag_service.enhanced_search(
            query=query,
            document_types=document_types if search_documents else [],
            use_rag=use_rag,
            max_results=max_results,
            user_ip=user_ip,
            user_agent=user_agent
        )
        
        # Translate results if needed
        if language != 'en' and results['success']:
            results = translation_service.translate_qa_result(results, language)
        
        # Format response for API
        if results['success']:
            formatted_results = format_enhanced_search_results(results['results'])
            
            response_data = {
                'success': True,
                'query': results['query'],
                'processed_query': results['processed_query'],
                'results': formatted_results,
                'query_type': results['query_type'],
                'response_time': results['response_time'],
                'total_results': results['total_results'],
                'generated_answer': results.get('generated_answer', ''),
                'rag_context_used': results.get('rag_context_used', False),
                'document_types_searched': results.get('document_types_searched', []),
                'message': _('Search completed successfully')
            }
        else:
            response_data = {
                'success': False,
                'error': results.get('error', 'Unknown error occurred'),
                'query': query,
                'results': [],
                'total_results': 0
            }
        
        return JsonResponse(response_data)
        
    except json.JSONDecodeError:
        return JsonResponse({'error': _('Invalid JSON data')}, status=400)
    except Exception as e:
        logger.error(f"Enhanced search API error: {str(e)}")
        return JsonResponse({'error': _('An error occurred during search')}, status=500)

def enhanced_search_results(request):
    """Enhanced web interface for search results"""
    query = request.GET.get('q', '').strip()
    language = request.GET.get('lang', 'en')
    document_types = request.GET.getlist('doc_types') or ['pdf', 'image', 'link']
    search_qa_entries = request.GET.get('search_qa', 'true').lower() == 'true'
    search_documents = request.GET.get('search_docs', 'true').lower() == 'true'
    use_rag = request.GET.get('use_rag', 'false').lower() == 'true'
    
    if not query:
        messages.warning(request, _('Please enter a search query'))
        return redirect('semantic_qa:index')
    
    # Get client information
    user_ip = get_client_ip(request)
    user_agent = request.META.get('HTTP_USER_AGENT', '')
    
    try:
        # Perform enhanced search
        results = rag_service.enhanced_search(
            query=query,
            document_types=document_types if search_documents else [],
            use_rag=use_rag,
            max_results=20,
            user_ip=user_ip,
            user_agent=user_agent
        )
        
        # for res in results.get('results', []):
        #     entry = res.get('entry')
        #     if entry and getattr(entry, 'image_link', None):
        #         # 追加提示到 answer 字段
        #         tip = f"\n\nPlease copy the link address to the address bar of your computer browser and then open it. {entry.image_link}"
        #         # 注意：entry.answer 可能是模型对象字段，建议用 setattr
        #         if hasattr(entry, 'answer'):
        #             entry.answer = (entry.answer or '') + tip
        #         # 如果 answer 也在 res 里（比如 res['answer']），也可以同步加
        #         if 'answer' in res:
        #             res['answer'] = (res['answer'] or '') + tip
        # Translate results if needed
        if language != 'en' and results['success']:
            results = translation_service.translate_qa_result(results, language)
        
        # Format results for display
        formatted_results = format_enhanced_search_results(results['results']) if results['success'] else []
        
        # Pagination
        paginator = Paginator(formatted_results, 5)
        page_number = request.GET.get('page')
        page_obj = paginator.get_page(page_number)
        
        # Get search form for filters
        search_form = EnhancedSearchForm(initial={
            'query': query,
            'language': language,
            'search_qa_entries': search_qa_entries,
            'search_documents': search_documents,
            'document_types': document_types,
            'use_rag': use_rag
        })
        
        context = {
            'query': query,
            'results': results,
            'formatted_results': formatted_results,
            'page_obj': page_obj,
            'language': language,
            'document_types': document_types,
            'search_qa_entries': search_qa_entries,
            'search_documents': search_documents,
            'use_rag': use_rag,
            'search_form': search_form,
            'supported_languages': SystemConfig.get_config('supported_languages', 'en,zh,es,fr,de,ja').split(','),
            'page_title': _('Search Results for "{query}"').format(query=query),
        }
        
        if not results['success']:
            context['error'] = results.get('error', 'Search failed')
            messages.error(request, _('An error occurred during search'))
        
    except Exception as e:
        logger.error(f"Enhanced search error: {str(e)}")
        messages.error(request, _('An error occurred during search'))
        context = {
            'query': query,
            'results': {'results': [], 'generated_answer': ''},
            'formatted_results': [],
            'error': str(e),
            'search_form': EnhancedSearchForm()
        }
    
    return render(request, 'semantic_qa/enhanced_search_results.html', context)

@csrf_exempt
@require_http_methods(["POST"])
def upload_document(request):
    """Upload and process documents (PDF, images, links) - Complete Enhanced Version"""
    try:
        # Check if it's a batch upload
        if 'files' in request.FILES and len(request.FILES.getlist('files')) > 1:
            return handle_batch_document_upload(request)
        
        # Get form data
        form_data = request.POST.copy()
        files_data = request.FILES.copy()
        
        # Log the incoming request
        user_ip = get_client_ip(request)
        logger.info(f"📤 Document upload request from {user_ip}")
        logger.info(f"📋 Form data: {dict(form_data)}")
        logger.info(f"📁 Files: {list(files_data.keys())}")
        
        # Validate form data
        form = DocumentUploadForm(form_data, files_data)
        
        if not form.is_valid():
            logger.error(f"❌ Form validation failed: {form.errors}")
            return JsonResponse({
                'success': False,
                'error': 'Form validation failed',
                'errors': dict(form.errors)
            }, status=400)
        
        # Create document record
        document = form.save(commit=False)
        
        # Enhanced document type detection and validation
        if document.document_type == 'link':
            # For web links, ensure we have a URL
            if not document.source_url:
                return JsonResponse({
                    'success': False,
                    'error': 'URL is required for link documents'
                }, status=400)
            
            # Validate URL format
            try:
                from urllib.parse import urlparse
                parsed_url = urlparse(document.source_url)
                if not parsed_url.scheme or not parsed_url.netloc:
                    return JsonResponse({
                        'success': False,
                        'error': 'Invalid URL format'
                    }, status=400)
            except Exception as e:
                return JsonResponse({
                    'success': False,
                    'error': f'URL validation failed: {str(e)}'
                }, status=400)
                
        else:
            # For file uploads, validate file and detect correct document type
            if not document.original_file:
                return JsonResponse({
                    'success': False,
                    'error': 'File upload is required for this document type'
                }, status=400)
            
            # Get file extension and validate
            try:
                file_extension = document.original_file.name.split('.')[-1].lower()
                logger.info(f"📄 File extension detected: .{file_extension}")
                
                # Map extensions to document types - COMPREHENSIVE MAPPING
                extension_map = {
                    # PDF files
                    'pdf': 'pdf',
                    
                    # Image files - COMPREHENSIVE LIST
                    'jpg': 'image', 'jpeg': 'image', 'png': 'image', 
                    'gif': 'image', 'bmp': 'image', 'tiff': 'image', 
                    'webp': 'image', 'tif': 'image', 'svg': 'image'
                }
                
                detected_type = extension_map.get(file_extension)
                logger.info(f"🔍 Detected document type: {detected_type} for extension .{file_extension}")
                
                if not detected_type:
                    return JsonResponse({
                        'success': False,
                        'error': f'Unsupported file type: .{file_extension}. Supported types: PDF, JPG, JPEG, PNG, GIF, BMP, TIFF, WEBP, SVG'
                    }, status=400)
                
                # FORCE the correct document type based on file extension
                # This prevents manual selection from overriding automatic detection
                document.document_type = detected_type
                logger.info(f"✅ Document type set to: {document.document_type}")
                
                # Validate file content matches expected type
                if detected_type == 'image':
                    try:
                        from PIL import Image
                        with Image.open(document.original_file) as img:
                            # Get image info
                            image_format = img.format
                            image_size = img.size
                            logger.info(f"🖼️ Image validated: {image_size[0]}x{image_size[1]} pixels, {image_format} format")
                    except Exception as img_error:
                        logger.error(f"❌ Image validation failed: {str(img_error)}")
                        return JsonResponse({
                            'success': False,
                            'error': f'Invalid image file: {str(img_error)}'
                        }, status=400)
                
                elif detected_type == 'pdf':
                    try:
                        import fitz  # PyMuPDF
                        with fitz.open(stream=document.original_file.read(), filetype="pdf") as pdf_doc:
                            page_count = len(pdf_doc)
                            logger.info(f"📄 PDF validated: {page_count} pages")
                            # Reset file pointer
                            document.original_file.seek(0)
                    except Exception as pdf_error:
                        logger.error(f"❌ PDF validation failed: {str(pdf_error)}")
                        return JsonResponse({
                            'success': False,
                            'error': f'Invalid PDF file: {str(pdf_error)}'
                        }, status=400)

            except Exception as e:
                logger.error(f"❌ File type detection failed: {str(e)}")
                return JsonResponse({
                    'success': False,
                    'error': f'File type detection failed: {str(e)}'
                }, status=500)
        
        # Auto-generate title if not provided
        if not document.title.strip():
            if document.original_file:
                document.title = os.path.splitext(document.original_file.name)[0]
            elif document.source_url:
                from urllib.parse import urlparse
                parsed = urlparse(document.source_url)
                document.title = parsed.path.split('/')[-1] or parsed.netloc
            else:
                document.title = f"Document_{timezone.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Ensure title is not too long
        if len(document.title) > 255:
            document.title = document.title[:252] + "..."
        
        # Set initial processing status
        document.processing_status = 'pending'
        
        # Save document
        try:
            document.save()
            logger.info(f"💾 Document saved with ID: {document.id}")
        except Exception as save_error:
            logger.error(f"❌ Failed to save document: {str(save_error)}")
            return JsonResponse({
                'success': False,
                'error': f'Failed to save document: {str(save_error)}'
            }, status=500)
        
        # Create processing job
        job = ProcessingJob.objects.create(
            job_type='document_processing',
            input_data={
                'document_id': document.id,
                'document_type': document.document_type,
                'title': document.title,
                'file_size': document.file_size or 0
            },
            status='pending',
            total_steps=1
        )
        
        logger.info(f"📝 Processing job created with ID: {job.id}")
        
        # Start processing immediately
        try:
            job.status = 'running'
            job.started_at = timezone.now()
            job.save()
            
            logger.info(f"🚀 Starting document processing for: {document.title}")
            
            # Process document
            result = doc_processor.process_document(document, job)
            
            # Update job based on result
            if result['success']:
                job.status = 'completed'
                job.success_count = 1
                job.progress_percent = 100
                job.current_step = 'Processing completed successfully'
                job.output_data = {
                    'chunks_created': result.get('chunks_created', 0),
                    'text_length': len(result.get('extracted_text', '')),
                    'metadata': result.get('metadata', {})
                }
                
                logger.info(f"✅ Document processing completed successfully")
                logger.info(f"📊 Text extracted: {len(result.get('extracted_text', ''))} characters")
                logger.info(f"🧩 Chunks created: {result.get('chunks_created', 0)}")
                
            else:
                job.status = 'failed'
                job.error_count = 1
                job.error_details = result.get('error', 'Unknown error occurred')
                
                logger.error(f"❌ Document processing failed: {job.error_details}")
            
            job.completed_at = timezone.now()
            job.save()
            
            # Refresh document from database to get updated data
            document.refresh_from_db()
            
        except Exception as processing_error:
            logger.error(f"❌ Document processing exception: {str(processing_error)}")
            
            # Update job with error
            job.status = 'failed'
            job.error_count = 1
            job.error_details = str(processing_error)
            job.completed_at = timezone.now()
            job.save()
            
            # Update document status
            document.processing_status = 'failed'
            document.processing_log = str(processing_error)
            document.save()
        
        # Log activity
        log_user_activity(user_ip, 'document_upload', 
                         f"Type: {document.document_type}, Title: {document.title}, Status: {document.processing_status}")
        
        # Prepare response data
        response_data = {
            'success': True,
            'message': 'Document uploaded and processed successfully' if job.status == 'completed' else 'Document uploaded, processing completed with issues',
            'document_id': document.id,
            'processing_status': document.processing_status,
            'job_id': job.id,
            'document_type': document.document_type,
            'title': document.title
        }
        
        # Add processing results if successful
        if job.status == 'completed' and job.output_data:
            vector_service = VectorManagementService()
            vector_service.auto_rebuild_after_upload(document_id=document.id)
            logger.info(f"✅ Auto re-vectorization triggered for document {document.id}")

            response_data.update({
                'extracted_text_length': job.output_data.get('text_length', 0),
                'chunks_created': job.output_data.get('chunks_created', 0),
                'metadata': job.output_data.get('metadata', {})
            })
        elif job.status == 'failed':
            response_data.update({
                'success': False,
                'error': job.error_details,
                'processing_failed': True
            })
        
        logger.info(f"📤 Sending response: {response_data}")
        
        return JsonResponse(response_data)
        
    except json.JSONDecodeError as json_error:
        logger.error(f"❌ JSON decode error: {str(json_error)}")
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON data in request'
        }, status=400)
        
    except Exception as e:
        logger.error(f"❌ Unexpected error in upload_document: {str(e)}")
        logger.error(f"❌ Error type: {type(e).__name__}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        
        return JsonResponse({
            'success': False,
            'error': f'An unexpected error occurred: {str(e)}',
            'error_type': type(e).__name__
        }, status=500)

def handle_batch_document_upload(request):
    """Handle batch upload of multiple documents"""
    try:
        files = request.FILES.getlist('files')
        default_category = request.POST.get('default_category', '')
        auto_process = request.POST.get('auto_process', 'true').lower() == 'true'
        
        if not files:
            return JsonResponse({'error': _('No files provided')}, status=400)
        
        # Validate file count
        if len(files) > 20:  # Limit to 20 files per batch
            return JsonResponse({'error': _('Too many files. Maximum 20 files per batch.')}, status=400)
        
        # Create batch processing job
        batch_job = ProcessingJob.objects.create(
            job_type='batch_upload',
            input_data={
                'file_count': len(files),
                'default_category': default_category,
                'auto_process': auto_process
            },
            total_steps=len(files),
            status='running',
            started_at=timezone.now()
        )
        
        documents_created = []
        success_count = 0
        error_count = 0
        error_details = []
        
        for i, file in enumerate(files):
            try:
                # Update progress
                progress = int((i / len(files)) * 90)  # Reserve 10% for final steps
                batch_job.update_progress(
                    progress,
                    f"Processing file {i+1}/{len(files)}: {file.name}"
                )
                
                # Validate file
                if file.size > 50 * 1024 * 1024:  # 50MB limit
                    error_details.append(f"{file.name}: File too large (>50MB)")
                    error_count += 1
                    continue
                
                # Determine document type
                file_extension = file.name.split('.')[-1].lower()
                if file_extension == 'pdf':
                    doc_type = 'pdf'
                elif file_extension in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp']:
                    doc_type = 'image'
                else:
                    error_details.append(f"Unsupported file type: {file.name}")
                    error_count += 1
                    continue
                
                # Create document
                document = Document.objects.create(
                    title=os.path.splitext(file.name)[0],
                    document_type=doc_type,
                    original_file=file,
                    file_size=file.size,
                    category=default_category,
                    processing_status='pending'
                )
                
                documents_created.append(document.id)
                
                # Process if auto_process is enabled
                if auto_process:
                    try:
                        result = doc_processor.process_document(document)
                        if result['success']:
                            success_count += 1
                        else:
                            error_count += 1
                            error_details.append(f"{file.name}: {result.get('error', 'Processing failed')}")
                    except Exception as process_error:
                        error_count += 1
                        error_details.append(f"{file.name}: Processing error - {str(process_error)}")
                else:
                    success_count += 1
                
            except Exception as e:
                error_count += 1
                error_details.append(f"{file.name}: Upload error - {str(e)}")
                logger.error(f"Batch upload error for {file.name}: {str(e)}")
        
        # Update batch job
        batch_job.update_progress(100, "Batch upload completed")
        batch_job.status = 'completed'
        batch_job.success_count = success_count
        batch_job.error_count = error_count
        batch_job.error_details = '\n'.join(error_details)
        batch_job.completed_at = timezone.now()
        batch_job.output_data = {
            'documents_created': documents_created,
            'processed_immediately': auto_process
        }
        batch_job.save()
        
        # Log activity
        user_ip = get_client_ip(request)
        log_user_activity(user_ip, 'batch_upload', 
                         f"Files: {len(files)}, Success: {success_count}, Errors: {error_count}")
        
        return JsonResponse({
            'success': True,
            'message': _('Batch upload completed'),
            'total_files': len(files),
            'success_count': success_count,
            'error_count': error_count,
            'error_details': error_details[:10] if error_details else [],
            'documents_created': len(documents_created),
            'job_id': batch_job.id,
            'has_more_errors': len(error_details) > 10
        })
        
    except Exception as e:
        logger.error(f"Batch upload error: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

def upload_document_page(request):
    """Document upload page"""
    if request.method == 'POST':
        # Handle form submissions (redirect to API)
        return redirect('semantic_qa:upload_document')
    
    # Get recent documents
    recent_documents = Document.objects.order_by('-created_at')[:5]
    
    # Get processing statistics
    processing_stats = Document.objects.values('processing_status').annotate(count=Count('id'))
    processing_stats_dict = {stat['processing_status']: stat['count'] for stat in processing_stats}
    
    # Get document type statistics
    type_stats = Document.objects.values('document_type').annotate(count=Count('id'))
    
    # Get recent processing jobs
    recent_jobs = ProcessingJob.objects.order_by('-created_at')[:5]
    
    context = {
        'recent_documents': recent_documents,
        'processing_stats': processing_stats_dict,
        'type_stats': type_stats,
        'recent_jobs': recent_jobs,
        'max_file_size': SystemConfig.get_config('max_file_size_mb', '50'),
        'page_title': _('Upload Documents')
    }
    return render(request, 'semantic_qa/upload_document.html', context)

@csrf_exempt
@require_http_methods(["POST"])
def process_document_api(request, document_id):
    """API to process a specific document"""
    try:
        document = get_object_or_404(Document, id=document_id)
        
        if document.processing_status == 'completed':
            return JsonResponse({
                'success': True,
                'message': _('Document already processed'),
                'status': 'completed'
            })
        
        if document.processing_status == 'processing':
            return JsonResponse({
                'success': True,
                'message': _('Document is currently being processed'),
                'status': 'processing'
            })
        
        # Create processing job
        job = ProcessingJob.objects.create(
            job_type='document_processing',
            input_data={'document_id': document.id},
            status='running',
            started_at=timezone.now()
        )
        
        # Process document
        result = doc_processor.process_document(document, job)
        
        # Update job
        if result['success']:
            job.status = 'completed'
            job.success_count = 1
            job.progress_percent = 100
            job.output_data = result
        else:
            job.status = 'failed'
            job.error_count = 1
            job.error_details = result.get('error', 'Processing failed')
        
        job.completed_at = timezone.now()
        job.save()
        
        return JsonResponse({
            'success': result['success'],
            'message': _('Document processing completed') if result['success'] else _('Document processing failed'),
            'status': document.processing_status,
            'job_id': job.id,
            'chunks_created': result.get('chunks_created', 0) if result['success'] else 0,
            'error': result.get('error') if not result['success'] else None
        })
        
    except Exception as e:
        logger.error(f"Document processing API error: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

def manage_documents(request):
    """Manage uploaded documents with comprehensive filtering"""
    # Get filter parameters
    search_query = request.GET.get('search', '')
    document_type = request.GET.get('document_type', '')
    processing_status = request.GET.get('processing_status', '')
    category_filter = request.GET.get('category', '')
    date_from = request.GET.get('date_from', '')
    date_to = request.GET.get('date_to', '')
    
    # Start with all documents
    documents = Document.objects.all()
    
    # Apply filters
    if search_query:
        documents = documents.filter(
            Q(title__icontains=search_query) |
            Q(extracted_text__icontains=search_query) |
            Q(tags__icontains=search_query) |
            Q(category__icontains=search_query)
        )
    
    if document_type:
        documents = documents.filter(document_type=document_type)
    
    if processing_status:
        documents = documents.filter(processing_status=processing_status)
    
    if category_filter:
        documents = documents.filter(category__icontains=category_filter)
    
    if date_from:
        try:
            from_date = datetime.strptime(date_from, '%Y-%m-%d').date()
            documents = documents.filter(created_at__date__gte=from_date)
        except ValueError:
            pass
    
    if date_to:
        try:
            to_date = datetime.strptime(date_to, '%Y-%m-%d').date()
            documents = documents.filter(created_at__date__lte=to_date)
        except ValueError:
            pass
    
    # Order by creation date
    documents = documents.order_by('-created_at')
    
    # Pagination
    paginator = Paginator(documents, 20)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    # Statistics
    total_documents = documents.count()
    processing_stats = Document.objects.values('processing_status').annotate(count=Count('id'))
    type_stats = Document.objects.values('document_type').annotate(count=Count('id'))
    
    # Get all categories for filter dropdown
    all_categories = Document.objects.exclude(category='').values_list('category', flat=True).distinct()
    
    # Filter form
    filter_form = FilterForm(initial={
        'search': search_query,
        'document_type': document_type,
        'processing_status': processing_status,
        'category': category_filter,
        'date_from': date_from,
        'date_to': date_to
    })
    
    context = {
        'page_obj': page_obj,
        'filter_form': filter_form,
        'search_query': search_query,
        'document_type': document_type,
        'processing_status': processing_status,
        'category_filter': category_filter,
        'total_documents': total_documents,
        'processing_stats': processing_stats,
        'type_stats': type_stats,
        'all_categories': all_categories,
        'page_title': _('Manage Documents')
    }
    
    return render(request, 'semantic_qa/manage_documents.html', context)

def document_detail(request, document_id):
    """View document details and chunks"""
    document = get_object_or_404(Document, id=document_id)
    
    # Get text chunks
    chunks = TextChunk.objects.filter(document=document).order_by('chunk_index')
    
    # Pagination for chunks
    paginator = Paginator(chunks, 10)
    page_number = request.GET.get('page')
    chunks_page = paginator.get_page(page_number)
    
    # Get related queries that matched this document
    related_queries = SemanticQuery.objects.filter(
        querymatch__text_chunk__document=document
    ).distinct().order_by('-created_at')[:5]
    
    # Processing job information
    processing_jobs = ProcessingJob.objects.filter(
        input_data__document_id=document.id
    ).order_by('-created_at')[:3]
    
    context = {
        'document': document,
        'chunks': chunks_page,
        'total_chunks': chunks.count(),
        'related_queries': related_queries,
        'processing_jobs': processing_jobs,
        'page_title': f'Document: {document.title}'
    }
    
    return render(request, 'semantic_qa/document_detail.html', context)

def delete_document(request, document_id):
    """Delete a document and its chunks"""
    document = get_object_or_404(Document, id=document_id)
    
    if request.method == 'POST':
        try:
            # Delete file from storage
            if document.original_file:
                try:
                    if os.path.exists(document.original_file.path):
                        os.remove(document.original_file.path)
                except Exception as e:
                    logger.warning(f"Could not delete file {document.original_file.path}: {str(e)}")
            
            # Log activity
            user_ip = get_client_ip(request)
            log_user_activity(user_ip, 'document_delete', f"ID: {document.id}, Title: {document.title}")
            
            # Delete document (chunks will be deleted via cascade)
            document_title = document.title
            document.delete()
            
            messages.success(request, _('Document "{title}" deleted successfully').format(title=document_title))
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            messages.error(request, _('Error deleting document'))
        
        return redirect('semantic_qa:manage_documents')
    
    context = {
        'document': document,
        'chunk_count': TextChunk.objects.filter(document=document).count(),
        'page_title': _('Delete Document')
    }
    
    return render(request, 'semantic_qa/confirm_delete_document.html', context)

@csrf_exempt
@require_http_methods(["GET"])
def processing_status_api(request, job_id):
    """API to check processing job status"""
    try:
        job = get_object_or_404(ProcessingJob, id=job_id)
        
        # Calculate elapsed time
        elapsed_time = None
        if job.started_at:
            if job.completed_at:
                elapsed_time = (job.completed_at - job.started_at).total_seconds()
            else:
                elapsed_time = (timezone.now() - job.started_at).total_seconds()
        
        return JsonResponse({
            'success': True,
            'job_id': job.id,
            'job_type': job.job_type,
            'status': job.status,
            'progress_percent': job.progress_percent,
            'current_step': job.current_step,
            'total_steps': job.total_steps,
            'success_count': job.success_count,
            'error_count': job.error_count,
            'error_details': job.error_details,
            'input_data': job.input_data,
            'output_data': job.output_data,
            'started_at': job.started_at.isoformat() if job.started_at else None,
            'completed_at': job.completed_at.isoformat() if job.completed_at else None,
            'elapsed_time': elapsed_time
        })
        
    except Exception as e:
        logger.error(f"Processing status API error: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

# Legacy API endpoints (maintained for compatibility)
@csrf_exempt
@require_http_methods(["POST"])
def search_api(request):
    """Legacy API endpoint - redirects to enhanced search"""
    try:
        data = json.loads(request.body)
        query = data.get('query', '').strip()
        language = data.get('language', 'en')
        
        # Convert to enhanced search format
        enhanced_data = {
            'query': query,
            'language': language,
            'document_types': ['pdf', 'image', 'link'],
            'search_qa_entries': True,
            'search_documents': True,
            'use_rag': True,
            'max_results': 10
        }
        
        # Temporarily replace request body
        original_body = request.body
        request._body = json.dumps(enhanced_data).encode('utf-8')
        
        # Call enhanced search
        response = enhanced_search_api(request)
        
        # Restore original body
        request._body = original_body
        
        return response
        
    except json.JSONDecodeError:
        return JsonResponse({'error': _('Invalid JSON data')}, status=400)
    except Exception as e:
        logger.error(f"Legacy search API error: {str(e)}")
        return JsonResponse({'error': _('An error occurred during search')}, status=500)

def search_results(request):
    """Legacy search results - redirects to enhanced search"""
    query = request.GET.get('q', '').strip()
    language = request.GET.get('lang', 'en')
    
    # Redirect to enhanced search with default parameters
    from django.http import HttpResponseRedirect
    from urllib.parse import urlencode
    
    params = {
        'q': query,
        'lang': language,
        'doc_types': ['pdf', 'image', 'link'],
        'search_qa': 'true',
        'search_docs': 'true',
        'use_rag': 'true'
    }
    
    url = reverse('semantic_qa:enhanced_search_results') + '?' + urlencode(params, doseq=True)
    return HttpResponseRedirect(url)

@csrf_exempt
@require_http_methods(["POST"])
def translate_api(request):
    """API endpoint for text translation"""
    try:
        data = json.loads(request.body)
        text = data.get('text', '').strip()
        target_language = data.get('target_language', 'en')
        source_language = data.get('source_language', 'en')
        
        if not text:
            return JsonResponse({'error': _('Text is required')}, status=400)
        
        translated_text = translation_service.translate_text(text, target_language, source_language)
        
        return JsonResponse({
            'success': True,
            'original_text': text,
            'translated_text': translated_text,
            'source_language': source_language,
            'target_language': target_language
        })
        
    except json.JSONDecodeError:
        return JsonResponse({'error': _('Invalid JSON data')}, status=400)
    except Exception as e:
        logger.error(f"Translation API error: {str(e)}")
        return JsonResponse({'error': _('Translation failed')}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def upload_excel(request):
    """Upload and process Excel file with QA data"""
    if 'excel_file' not in request.FILES:
        return JsonResponse({'error': _('No file uploaded')}, status=400)
    
    file = request.FILES['excel_file']
    
    if not file.name.endswith(('.xlsx', '.xls')):
        return JsonResponse({'error': _('Please upload an Excel file (.xlsx or .xls)')}, status=400)
    
    try:
        # Log file details for debugging
        logger.info(f"Processing Excel file: {file.name}, size: {file.size} bytes")
        
        # Parse Excel file
        qa_entries = parse_excel_file(file)
        
        if not qa_entries:
            return JsonResponse({'error': _('No valid data found in Excel file')}, status=400)
        
        # Save to database
        created_count = 0
        updated_count = 0
        error_count = 0
        error_details = []
        
        for i, entry_data in enumerate(qa_entries):
            try:
                # Create a hash for checking uniqueness
                import hashlib
                unique_string = f"{entry_data['sku']}|{entry_data['question']}"
                question_hash = hashlib.sha256(unique_string.encode('utf-8')).hexdigest()
                
                entry, created = QAEntry.objects.update_or_create(
                    question_hash=question_hash,
                    defaults={
                        'sku': entry_data['sku'],
                        'question': entry_data['question'],
                        'answer': entry_data['answer'],
                        'image_link': entry_data.get('image_link', ''),
                        'category': entry_data.get('category', ''),
                        'keywords': entry_data.get('keywords', '')
                    }
                )
                
                if created:
                    created_count += 1
                else:
                    updated_count += 1
                    
            except Exception as e:
                error_count += 1
                error_msg = f"Row {i+3}: {str(e)}"  # +3 because of header and description rows
                error_details.append(error_msg)
                logger.error(f"Error saving Excel entry {i+1}: {str(e)}")
        
        # Create text chunks for QA entries (for RAG)
        try:
            chunks_created = 0
            for entry_data in qa_entries:
                # Find the created/updated entry
                unique_string = f"{entry_data['sku']}|{entry_data['question']}"
                question_hash = hashlib.sha256(unique_string.encode('utf-8')).hexdigest()
                
                try:
                    qa_entry = QAEntry.objects.get(question_hash=question_hash)
                    
                    # Create text chunk for RAG
                    chunk_text = f"SKU: {qa_entry.sku}\nQuestion: {qa_entry.question}\nAnswer: {qa_entry.answer}"
                    if qa_entry.category:
                        chunk_text += f"\nCategory: {qa_entry.category}"
                    
                    # Delete existing chunks for this QA entry
                    TextChunk.objects.filter(qa_entry=qa_entry).delete()
                    
                    # Create new chunk
                    TextChunk.objects.create(
                        qa_entry=qa_entry,
                        text=chunk_text,
                        chunk_index=0,
                        chunk_size=len(chunk_text),
                        keywords=qa_entry.keywords or ''
                    )
                    chunks_created += 1
                    
                except QAEntry.DoesNotExist:
                    continue
                    
        except Exception as e:
            logger.warning(f"Error creating text chunks: {str(e)}")
        
        # Log activity
        user_ip = get_client_ip(request)
        log_user_activity(user_ip, 'excel_upload', 
                         f"File: {file.name}, Created: {created_count}, Updated: {updated_count}")
        
        response_data = {
            'success': True,
            'message': _('Successfully processed Excel file'),
            'total_entries': len(qa_entries),
            'created': created_count,
            'updated': updated_count,
            'errors': error_count,
            'chunks_created': chunks_created if 'chunks_created' in locals() else 0
        }
        if error_details:
            response_data['error_details'] = error_details[:10]  # Limit to first 10 errors
            if len(error_details) > 10:
                response_data['additional_errors'] = len(error_details) - 10
        
        return JsonResponse(response_data)
        
    except ValueError as ve:
        # This handles parsing errors with column detection
        logger.error(f"Excel parsing error: {str(ve)}")
        return JsonResponse({
            'error': str(ve),
            'suggestion': _('Please check that your Excel file has the required columns: SKU, Question, Answer. Column names are case-insensitive.')
        }, status=400)
        
    except Exception as e:
        logger.error(f"Excel upload error: {str(e)}")
        return JsonResponse({
            'error': f'{_("Error processing file")}: {str(e)}',
            'suggestion': _('Please try downloading and using our Excel template.')
        }, status=500)

def upload_excel_page(request):
    """Excel upload page with enhanced error handling"""
    if request.method == 'POST':
        form = ExcelUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Process the uploaded file
            file = form.cleaned_data['excel_file']
            try:
                logger.info(f"Processing uploaded Excel file: {file.name}")
                
                qa_entries = parse_excel_file(file)
                
                if not qa_entries:
                    messages.error(request, _('No valid data found in the Excel file. Please check the file format and content.'))
                    return render(request, 'semantic_qa/upload_excel.html', {'form': form})
                
                created_count = 0
                updated_count = 0
                error_count = 0
                
                for entry_data in qa_entries:
                    try:
                        # Create a hash for checking uniqueness
                        import hashlib
                        unique_string = f"{entry_data['sku']}|{entry_data['question']}"
                        question_hash = hashlib.sha256(unique_string.encode('utf-8')).hexdigest()
                        
                        entry, created = QAEntry.objects.update_or_create(
                            question_hash=question_hash,
                            defaults={
                                'sku': entry_data['sku'],
                                'question': entry_data['question'],
                                'answer': entry_data['answer'],
                                'image_link': entry_data.get('image_link', ''),
                                'category': entry_data.get('category', ''),
                                'keywords': entry_data.get('keywords', '')
                            }
                        )
                        
                        if created:
                            created_count += 1
                        else:
                            updated_count += 1
                            
                    except Exception as e:
                        error_count += 1
                        logger.error(f"Error saving Excel entry: {str(e)}")
                
                if error_count > 0:
                    messages.warning(request, 
                        _('Processed {total} entries ({created} created, {updated} updated) with {errors} errors. Check logs for details.').format(
                            total=len(qa_entries), created=created_count, updated=updated_count, errors=error_count
                        ))
                else:
                    messages.success(request, 
                        _('Successfully uploaded {total} entries ({created} created, {updated} updated)').format(
                            total=len(qa_entries), created=created_count, updated=updated_count
                        ))
                
                return redirect('semantic_qa:manage_entries')
                
            except ValueError as ve:
                # Handle column detection errors
                logger.error(f"Column detection error: {str(ve)}")
                messages.error(request, f'{str(ve)} Please download our template to see the correct format.')
                
            except Exception as e:
                logger.error(f"File processing error: {str(e)}")
                messages.error(request, f'{_("Error processing file")}: {str(e)}')
    else:
        form = ExcelUploadForm()
    
    # Get some sample data for display
    sample_entries = QAEntry.objects.all()[:3]
    
    context = {
        'form': form,
        'sample_entries': sample_entries,
        'page_title': _('Upload Excel File')
    }
    return render(request, 'semantic_qa/upload_excel.html', context)

def download_template(request):
    """Download Excel template"""
    try:
        excel_buffer = generate_excel_template()
        
        response = HttpResponse(
            excel_buffer.getvalue(),
            content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        response['Content-Disposition'] = 'attachment; filename="qa_template.xlsx"'
        
        return response
        
    except Exception as e:
        logger.error(f"Template download error: {str(e)}")
        messages.error(request, _('Error generating template'))
        return redirect('semantic_qa:upload_excel_page')

def admin_dashboard(request):
    """Enhanced admin dashboard with document and RAG statistics"""
    # Get comprehensive statistics
    total_qa_entries = QAEntry.objects.count()
    total_documents = Document.objects.count()
    completed_documents = Document.objects.filter(processing_status='completed').count()
    total_queries = SemanticQuery.objects.count()
    total_chunks = TextChunk.objects.count()
    rag_queries = SemanticQuery.objects.filter(use_rag=True).count()
    
    # Recent queries
    recent_queries = SemanticQuery.objects.select_related().order_by('-created_at')[:10]
    
    # Popular SKUs from QA entries
    popular_skus = QAEntry.objects.values('sku')\
        .annotate(query_count=Count('querymatch__query'))\
        .order_by('-query_count')[:10]
    
    # Categories distribution (QA entries)
    qa_categories = QAEntry.objects.exclude(category='')\
        .values('category')\
        .annotate(count=Count('id'))\
        .order_by('-count')[:10]
    
    # Document categories distribution
    doc_categories = Document.objects.filter(processing_status='completed')\
        .exclude(category='')\
        .values('category')\
        .annotate(count=Count('id'))\
        .order_by('-count')[:10]
    
    # Query types distribution
    query_types = SemanticQuery.objects.values('query_type')\
        .annotate(count=Count('id'))\
        .order_by('-count')
    
    # Document type distribution
    document_types = Document.objects.values('document_type')\
        .annotate(count=Count('id'))\
        .order_by('-count')
    
    # Processing status distribution
    processing_status = Document.objects.values('processing_status')\
        .annotate(count=Count('id'))
    
    # Recent processing jobs
    recent_jobs = ProcessingJob.objects.order_by('-created_at')[:5]
    
    # Performance metrics
    avg_response_time = SemanticQuery.objects.aggregate(
        avg_time=Avg('response_time')
    )['avg_time'] or 0
    
    # Daily query trends (last 7 days)
    from datetime import timedelta
    seven_days_ago = timezone.now() - timedelta(days=7)
    daily_queries = SemanticQuery.objects.filter(created_at__gte=seven_days_ago)\
        .extra(select={'day': 'DATE(created_at)'})\
        .values('day')\
        .annotate(count=Count('id'))\
        .order_by('day')
    
    # RAG usage statistics
    rag_success_rate = 0
    if rag_queries > 0:
        successful_rag = SemanticQuery.objects.filter(
            use_rag=True,
            generated_answer__isnull=False
        ).exclude(generated_answer='').count()
        rag_success_rate = (successful_rag / rag_queries) * 100
    
    context = {
        'total_qa_entries': total_qa_entries,
        'total_documents': total_documents,
        'completed_documents': completed_documents,
        'total_queries': total_queries,
        'total_chunks': total_chunks,
        'rag_queries': rag_queries,
        'rag_success_rate': round(rag_success_rate, 1),
        'avg_response_time': round(avg_response_time, 3),
        'recent_queries': recent_queries,
        'popular_skus': popular_skus,
        'qa_categories': qa_categories,
        'doc_categories': doc_categories,
        'query_types': query_types,
        'document_types': document_types,
        'processing_status': processing_status,
        'recent_jobs': recent_jobs,
        'daily_queries': list(daily_queries),
        'page_title': _('Admin Dashboard')
    }
    
    return render(request, 'semantic_qa/enhanced_admin_dashboard.html', context)

def manage_entries(request):
    """Manage QA entries with enhanced filtering"""
    # Get filter parameters
    search_query = request.GET.get('search', '')
    category_filter = request.GET.get('category', '')
    sku_filter = request.GET.get('sku', '')
    date_from = request.GET.get('date_from', '')
    date_to = request.GET.get('date_to', '')
    
    entries = QAEntry.objects.all()
    
    # Apply filters
    if search_query:
        entries = entries.filter(
            Q(sku__icontains=search_query) |
            Q(question__icontains=search_query) |
            Q(answer__icontains=search_query) |
            Q(keywords__icontains=search_query)
        )
    
    if category_filter:
        entries = entries.filter(category__icontains=category_filter)
    
    if sku_filter:
        entries = entries.filter(sku__icontains=sku_filter)
    
    if date_from:
        try:
            from_date = datetime.strptime(date_from, '%Y-%m-%d').date()
            entries = entries.filter(created_at__date__gte=from_date)
        except ValueError:
            pass
    
    if date_to:
        try:
            to_date = datetime.strptime(date_to, '%Y-%m-%d').date()
            entries = entries.filter(created_at__date__lte=to_date)
        except ValueError:
            pass
    
    entries = entries.order_by('-created_at')
    
    # Pagination
    paginator = Paginator(entries, 20)  # 20 entries per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    # Get all categories for filter dropdown
    all_categories = QAEntry.objects.exclude(category='')\
        .values_list('category', flat=True)\
        .distinct().order_by('category')
    
    # Filter form
    filter_form = FilterForm(initial={
        'search': search_query,
        'category': category_filter,
        'sku': sku_filter,
        'date_from': date_from,
        'date_to': date_to,
        'content_type': 'qa_entry'
    })
    
    context = {
        'page_obj': page_obj,
        'filter_form': filter_form,
        'search_query': search_query,
        'category_filter': category_filter,
        'sku_filter': sku_filter,
        'all_categories': all_categories,
        'total_entries': entries.count(),
        'page_title': _('Manage QA Entries')
    }
    
    return render(request, 'semantic_qa/manage_entries.html', context)

def edit_entry(request, entry_id):
    """Edit a QA entry""" 
    entry = get_object_or_404(QAEntry, id=entry_id)
    
    if request.method == 'POST':
        form = QAEntryForm(request.POST, instance=entry)
        if form.is_valid():
            updated_entry = form.save()
            
            # Update associated text chunk for RAG
            try:
                chunk_text = f"SKU: {updated_entry.sku}\nQuestion: {updated_entry.question}\nAnswer: {updated_entry.answer}"
                if updated_entry.category:
                    chunk_text += f"\nCategory: {updated_entry.category}"
                
                # Update or create chunk
                chunk, created = TextChunk.objects.update_or_create(
                    qa_entry=updated_entry,
                    defaults={
                        'text': chunk_text,
                        'chunk_index': 0,
                        'chunk_size': len(chunk_text),
                        'keywords': updated_entry.keywords or ''
                    }
                )
                
            except Exception as e:
                logger.warning(f"Error updating text chunk for QA entry {entry_id}: {str(e)}")
            
            messages.success(request, _('QA entry updated successfully'))
            return redirect('semantic_qa:manage_entries')
    else:
        form = QAEntryForm(instance=entry)
    
    context = {
        'form': form,
        'entry': entry,
        'page_title': _('Edit QA Entry')
    }
    
    return render(request, 'semantic_qa/edit_entry.html', context)

def delete_entry(request, entry_id):
    """Delete a QA entry"""
    entry = get_object_or_404(QAEntry, id=entry_id)
    
    if request.method == 'POST':
        # Log activity
        user_ip = get_client_ip(request)
        log_user_activity(user_ip, 'qa_entry_delete', f"ID: {entry.id}, SKU: {entry.sku}")
        
        entry_sku = entry.sku
        entry.delete()  # Text chunks will be deleted via cascade
        
        messages.success(request, _('QA entry "{sku}" deleted successfully').format(sku=entry_sku))
        return redirect('semantic_qa:manage_entries')
    
    context = {
        'entry': entry,
        'page_title': _('Delete QA Entry')
    }
    
    return render(request, 'semantic_qa/confirm_delete.html', context)

def analytics_dashboard(request):
    """Enhanced analytics dashboard with document and RAG metrics"""
    from django.db.models import Avg, Max, Min
    from datetime import datetime, timedelta
    
    # Date range filter
    days = int(request.GET.get('days', 30))
    start_date = timezone.now() - timedelta(days=days)
    
    # Query statistics
    queries_in_period = SemanticQuery.objects.filter(created_at__gte=start_date)
    
    # Basic metrics
    total_queries = queries_in_period.count()
    avg_response_time = queries_in_period.aggregate(avg_time=Avg('response_time'))['avg_time'] or 0
    
    # RAG metrics
    rag_queries = queries_in_period.filter(use_rag=True)
    rag_successful = rag_queries.exclude(generated_answer='').count()
    rag_success_rate = (rag_successful / rag_queries.count() * 100) if rag_queries.count() > 0 else 0
    
    # Query types distribution
    query_type_stats = queries_in_period.values('query_type')\
        .annotate(count=Count('id'))\
        .order_by('-count')
    
    # Document type usage in queries
    doc_type_usage = queries_in_period.exclude(document_types_searched=[])\
        .values('document_types_searched')\
        .annotate(count=Count('id'))
    
    # Daily statistics
    daily_stats = queries_in_period.extra(
        select={'day': 'DATE(created_at)'}
    ).values('day').annotate(
        total_queries=Count('id'),
        avg_response_time=Avg('response_time'),
        rag_queries=Count('id', filter=Q(use_rag=True))
    ).order_by('day')
    
    # Most searched terms
    popular_terms = queries_in_period.values('processed_query')\
        .annotate(count=Count('id'))\
        .order_by('-count')[:20]
    
    # Performance metrics by query type
    performance_by_type = queries_in_period.values('query_type')\
        .annotate(
            count=Count('id'),
            avg_time=Avg('response_time'),
            max_time=Max('response_time'),
            min_time=Min('response_time')
        ).order_by('-count')
    
    # Document processing statistics
    doc_processing_stats = ProcessingJob.objects.filter(
        job_type='document_processing',
        created_at__gte=start_date
    ).values('status').annotate(count=Count('id'))
    
    # Content type distribution in results
    qa_matches = QueryMatch.objects.filter(
        query__created_at__gte=start_date,
        qa_entry__isnull=False
    ).count()
    
    doc_matches = QueryMatch.objects.filter(
        query__created_at__gte=start_date,
        text_chunk__isnull=False
    ).count()
    
    context = {
        'days': days,
        'total_queries': total_queries,
        'avg_response_time': round(avg_response_time, 3),
        'rag_queries_count': rag_queries.count(),
        'rag_success_rate': round(rag_success_rate, 1),
        'query_type_stats': query_type_stats,
        'doc_type_usage': doc_type_usage,
        'daily_stats': list(daily_stats),
        'popular_terms': popular_terms,
        'performance_by_type': performance_by_type,
        'doc_processing_stats': doc_processing_stats,
        'qa_matches': qa_matches,
        'doc_matches': doc_matches,
        'total_documents': Document.objects.count(),
        'completed_documents': Document.objects.filter(processing_status='completed').count(),
        'page_title': _('Analytics Dashboard')
    }
    
    return render(request, 'semantic_qa/analytics_dashboard.html', context)

def query_logs(request):
    """View detailed query logs with filtering"""
    queries = SemanticQuery.objects.select_related().order_by('-created_at')
    
    # Filters
    query_type = request.GET.get('type', '')
    use_rag = request.GET.get('rag', '')
    date_from = request.GET.get('date_from', '')
    date_to = request.GET.get('date_to', '')
    search_term = request.GET.get('search', '')
    
    if query_type:
        queries = queries.filter(query_type=query_type)
    
    if use_rag:
        if use_rag == 'true':
            queries = queries.filter(use_rag=True)
        elif use_rag == 'false':
            queries = queries.filter(use_rag=False)
    
    if date_from:
        try:
            from_date = datetime.strptime(date_from, '%Y-%m-%d').date()
            queries = queries.filter(created_at__date__gte=from_date)
        except ValueError:
            pass
    
    if date_to:
        try:
            to_date = datetime.strptime(date_to, '%Y-%m-%d').date()
            queries = queries.filter(created_at__date__lte=to_date)
        except ValueError:
            pass
    
    if search_term:
        queries = queries.filter(
            Q(query_text__icontains=search_term) |
            Q(processed_query__icontains=search_term)
        )
    
    # Pagination
    paginator = Paginator(queries, 50)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'query_type': query_type,
        'use_rag': use_rag,
        'date_from': date_from,
        'date_to': date_to,
        'search_term': search_term,
        'total_queries': queries.count(),
        'page_title': _('Query Logs')
    }
    
    return render(request, 'semantic_qa/query_logs.html', context)

def export_data(request):
    """Export QA data and documents to Excel"""
    try:
        from django.http import HttpResponse
        import pandas as pd
        from io import BytesIO
        
        # Create Excel writer
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Export QA entries
            qa_entries = QAEntry.objects.all().values(
                'sku', 'question', 'answer', 'image_link', 
                'category', 'keywords', 'created_at', 'updated_at'
            )
            
            if qa_entries:
                qa_df = pd.DataFrame(list(qa_entries))
                qa_df.to_excel(writer, sheet_name='QA_Entries', index=False)
            
            # Export documents
            documents = Document.objects.filter(processing_status='completed').values(
                'title', 'document_type', 'category', 'file_size',
                'processing_status', 'language_detected', 'created_at'
            )
            
            if documents:
                doc_df = pd.DataFrame(list(documents))
                doc_df.to_excel(writer, sheet_name='Documents', index=False)
            
            # Export query statistics
            queries = SemanticQuery.objects.values(
                'query_text', 'query_type', 'use_rag', 'total_results',
                'response_time', 'created_at'
            )
            
            if queries:
                query_df = pd.DataFrame(list(queries))
                query_df.to_excel(writer, sheet_name='Query_Logs', index=False)
        
        output.seek(0)
        
        response = HttpResponse(
            output.getvalue(),
            content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
        timestamp = timezone.now().strftime('%Y%m%d_%H%M%S')
        response['Content-Disposition'] = f'attachment; filename="semantic_qa_export_{timestamp}.xlsx"'
        
        return response
        
    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        messages.error(request, _('Error exporting data'))
        return redirect('semantic_qa:admin_dashboard')

def clean_image_url(url: str) -> str:
    """Enhanced image URL cleaning function to fix malformed URLs"""
    if not url:
        return ""
    
    url = url.strip()
    
    # Fix common URL formatting issues that cause the SSLError
    
    # Case 1: Missing protocol separators - Fixed patterns from logs
    url_fixes = [
        ('ae01.alicdn.comkf', 'ae01.alicdn.com/kf/'),
        ('ae02.alicdn.comkf', 'ae02.alicdn.com/kf/'),
        ('ae03.alicdn.comkf', 'ae03.alicdn.com/kf/'),
        ('httpsae01', 'https://ae01'),
        ('httpsae02', 'https://ae02'),
        ('httpsae03', 'https://ae03'),
        ('httpae01', 'http://ae01'),
        ('httpae02', 'http://ae02'),
        ('httpae03', 'http://ae03'),
        ('httpswww', 'https://www'),
        ('httpwww', 'http://www'),
        ('rsnavwiki.comimages', 'rsnavwiki.com/images/'),
    ]
    
    for old_pattern, new_pattern in url_fixes:
        if old_pattern in url:
            url = url.replace(old_pattern, new_pattern)
    
    # Case 2: Missing :// after protocol
    if url.startswith('https') and '://' not in url:
        url = url.replace('https', 'https://', 1)
    elif url.startswith('http') and '://' not in url:
        url = url.replace('http', 'http://', 1)
    
    # Case 3: URL doesn't start with protocol at all
    elif not url.startswith(('http://', 'https://')):
        if any(x in url for x in ['alicdn.com', 'ae01', 'ae02', 'ae03']):
            url = 'https://' + url
        else:
            url = 'https://' + url
    
    # Case 4: Fix case sensitivity issues in URLs
    # Some URLs have mixed case that causes issues
    if 'alicdn.com' in url:
        # Keep the base domain lowercase, but preserve the path case
        parts = url.split('alicdn.com')
        if len(parts) == 2:
            base = parts[0] + 'alicdn.com'
            path = parts[1]
            url = base.lower() + path
    
    return url

def image_proxy(request, image_url):
    """Enhanced image proxy with better URL handling and error recovery"""
    try:
        import urllib.parse
        
        # Decode URL (handle double encoding)
        if '%253A' in image_url:
            image_url = urllib.parse.unquote(image_url)
        image_url = urllib.parse.unquote(image_url)
        
        # Clean the URL with enhanced function
        image_url = clean_image_url(image_url)
        
        logger.info(f"🖼️ Enhanced image proxy request for: {image_url}")
        
        # Validate URL format
        if not image_url.startswith(('http://', 'https://')):
            logger.error(f"❌ Invalid URL format: {image_url}")
            return HttpResponse("Invalid URL", status=400)
        
        # Enhanced headers to mimic real browser more closely
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://www.aliexpress.com/',
            'Sec-Fetch-Dest': 'image',
            'Sec-Fetch-Mode': 'no-cors',
            'Sec-Fetch-Site': 'cross-site',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        # Multiple retry strategy
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Fetch image with different configurations per attempt
                if attempt == 0:
                    # First attempt: Normal request with SSL verification
                    response = requests.get(
                        image_url, 
                        headers=headers, 
                        timeout=15, 
                        stream=True,
                        verify=True,
                        allow_redirects=True
                    )
                elif attempt == 1:
                    # Second attempt: Without SSL verification
                    response = requests.get(
                        image_url, 
                        headers=headers, 
                        timeout=20, 
                        stream=True,
                        verify=False,
                        allow_redirects=True
                    )
                else:
                    # Third attempt: Minimal headers
                    minimal_headers = {
                        'User-Agent': 'Mozilla/5.0 (compatible; ImageBot/1.0)',
                        'Accept': 'image/*,*/*;q=0.8'
                    }
                    response = requests.get(
                        image_url, 
                        headers=minimal_headers, 
                        timeout=25, 
                        stream=True,
                        verify=False,
                        allow_redirects=True
                    )
                
                if response.status_code == 200:
                    content_type = response.headers.get('content-type', 'image/jpeg')
                    
                    # Create Django response
                    django_response = HttpResponse(response.content, content_type=content_type)
                    django_response['Cache-Control'] = 'public, max-age=3600'
                    django_response['Access-Control-Allow-Origin'] = '*'
                    
                    logger.info(f"✅ Successfully proxied image on attempt {attempt + 1}: {image_url}")
                    return django_response
                else:
                    logger.warning(f"⚠️ HTTP {response.status_code} on attempt {attempt + 1} for: {image_url}")
                    if attempt == max_retries - 1:
                        return HttpResponse(f"Image not found (HTTP {response.status_code})", status=404)
                    
            except requests.exceptions.SSLError as e:
                logger.warning(f"🔒 SSL error on attempt {attempt + 1} for {image_url}: {str(e)}")
                if attempt == max_retries - 1:
                    return HttpResponse("SSL error accessing image", status=404)
                continue
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"🌐 Request error on attempt {attempt + 1} for {image_url}: {str(e)}")
                if attempt == max_retries - 1:
                    return HttpResponse("Image fetch failed", status=404)
                continue
        
        return HttpResponse("All retry attempts failed", status=404)
        
    except Exception as e:
        logger.error(f"❌ General error for {image_url}: {str(e)}")
        return HttpResponse("Internal error", status=500)

def set_language(request):
    """Set user language preference"""
    next_url = request.POST.get('next', request.GET.get('next', '/'))
    language = request.POST.get('language', request.GET.get('language', 'en'))
    
    # Activate the language
    translation.activate(language)
    
    # Set language in session
    request.session['django_language'] = language
    
    response = redirect(next_url)
    response.set_cookie(settings.LANGUAGE_COOKIE_NAME, language)
    
    return response

def system_config(request):
    """System configuration page"""
    if request.method == 'POST':
        form = SystemConfigForm(request.POST)
        if form.is_valid():
            try:
                form.save()
                messages.success(request, _('System configuration updated successfully'))
                
                # Reinitialize services with new config
                try:
                    global rag_service, doc_processor, translation_service
                    rag_service = RAGService()
                    doc_processor = DocumentProcessor()
                    translation_service = TranslationService()
                    messages.info(request, _('Services reinitialized with new configuration'))
                except Exception as e:
                    logger.warning(f"Service reinitialization warning: {str(e)}")
                    messages.warning(request, _('Configuration saved but some services may need restart'))
                
                return redirect('semantic_qa:system_config')
                
            except Exception as e:
                logger.error(f"Error saving system config: {str(e)}")
                messages.error(request, _('Error saving configuration'))
    else:
        form = SystemConfigForm()
    
    # Get current service status
    service_status = {
        'rag_service': rag_service is not None and rag_service.embeddings is not None,
        'doc_processor': doc_processor is not None and doc_processor.ocr_reader is not None,
        'translation_service': translation_service is not None
    }
    
    # Get system statistics
    system_stats = {
        'total_qa_entries': QAEntry.objects.count(),
        'total_documents': Document.objects.count(),
        'total_chunks': TextChunk.objects.count(),
        'total_queries': SemanticQuery.objects.count(),
        'processing_jobs': ProcessingJob.objects.filter(status='running').count()
    }
    
    context = {
        'form': form,
        'service_status': service_status,
        'system_stats': system_stats,
        'page_title': _('System Configuration')
    }
    
    return render(request, 'semantic_qa/system_config.html', context)

@csrf_exempt
@require_http_methods(["POST"])
def test_service_api(request):
    """API to test service connectivity"""
    try:
        data = json.loads(request.body)
        service_type = data.get('service_type', '')
        
        results = {}
        
        if service_type == 'rag' or not service_type:
            # Test RAG service
            try:
                if rag_service and rag_service.embeddings:
                    test_embedding = rag_service.embeddings.embed_query("test")
                    results['rag'] = {
                        'status': 'success',
                        'message': f'RAG service working. Embedding dimension: {len(test_embedding)}',
                        'embedding_model': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
                    }
                else:
                    results['rag'] = {
                        'status': 'error',
                        'message': 'RAG service not initialized'
                    }
            except Exception as e:
                results['rag'] = {
                    'status': 'error',
                    'message': f'RAG service error: {str(e)}'
                }
        
        if service_type == 'chatglm' or not service_type:
            # Test ChatGLM
            try:
                if rag_service and rag_service.chatglm_client:
                    test_response = rag_service.chatglm_client.chat.completions.create(
                        model="glm-4-flash",
                        messages=[{"role": "user", "content": "Reply with 'OK'"}],
                        max_tokens=10
                    )
                    results['chatglm'] = {
                        'status': 'success',
                        'message': 'ChatGLM service working',
                        'response': test_response.choices[0].message.content
                    }
                else:
                    results['chatglm'] = {
                        'status': 'error',
                        'message': 'ChatGLM client not initialized'
                    }
            except Exception as e:
                results['chatglm'] = {
                    'status': 'error',
                    'message': f'ChatGLM error: {str(e)}'
                }
        
        if service_type == 'ocr' or not service_type:
            # Test OCR
            try:
                if doc_processor and doc_processor.ocr_reader:
                    results['ocr'] = {
                        'status': 'success',
                        'message': 'OCR service initialized',
                        'languages': ['ch_sim', 'en']
                    }
                else:
                    results['ocr'] = {
                        'status': 'error',
                        'message': 'OCR service not initialized'
                    }
            except Exception as e:
                results['ocr'] = {
                    'status': 'error',
                    'message': f'OCR error: {str(e)}'
                }
        
        return JsonResponse({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Service test error: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

def processing_jobs(request):
    """View processing jobs with filtering"""
    jobs = ProcessingJob.objects.order_by('-created_at')
    
    # Filters
    job_type = request.GET.get('type', '')
    status = request.GET.get('status', '')
    date_from = request.GET.get('date_from', '')
    date_to = request.GET.get('date_to', '')
    
    if job_type:
        jobs = jobs.filter(job_type=job_type)
    
    if status:
        jobs = jobs.filter(status=status)
    
    if date_from:
        try:
            from_date = datetime.strptime(date_from, '%Y-%m-%d').date()
            jobs = jobs.filter(created_at__date__gte=from_date)
        except ValueError:
            pass
    
    if date_to:
        try:
            to_date = datetime.strptime(date_to, '%Y-%m-%d').date()
            jobs = jobs.filter(created_at__date__lte=to_date)
        except ValueError:
            pass
    
    # Pagination
    paginator = Paginator(jobs, 50)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    # Statistics
    job_stats = ProcessingJob.objects.values('status').annotate(count=Count('id'))
    type_stats = ProcessingJob.objects.values('job_type').annotate(count=Count('id'))
    
    context = {
        'page_obj': page_obj,
        'job_type': job_type,
        'status': status,
        'date_from': date_from,
        'date_to': date_to,
        'job_stats': job_stats,
        'type_stats': type_stats,
        'total_jobs': jobs.count(),
        'page_title': _('Processing Jobs')
    }
    
    return render(request, 'semantic_qa/processing_jobs.html', context)

@csrf_exempt
@require_http_methods(["POST"])
def retry_processing_job(request, job_id):
    """Retry a failed processing job"""
    try:
        job = get_object_or_404(ProcessingJob, id=job_id)
        
        if job.status not in ['failed', 'cancelled']:
            return JsonResponse({
                'success': False,
                'error': 'Job is not in a retriable state'
            }, status=400)
        
        # Reset job status
        job.status = 'pending'
        job.progress_percent = 0
        job.current_step = ''
        job.error_details = ''
        job.started_at = None
        job.completed_at = None
        job.save()
        
        # Restart processing based on job type
        if job.job_type == 'document_processing':
            document_id = job.input_data.get('document_id')
            if document_id:
                try:
                    document = Document.objects.get(id=document_id)
                    document.processing_status = 'pending'
                    document.save()
                    
                    # Start processing
                    job.status = 'running'
                    job.started_at = timezone.now()
                    job.save()
                    
                    result = doc_processor.process_document(document, job)
                    
                    if result['success']:
                        job.status = 'completed'
                        job.success_count = 1
                        job.progress_percent = 100
                    else:
                        job.status = 'failed'
                        job.error_count = 1
                        job.error_details = result.get('error', 'Processing failed')
                    
                    job.completed_at = timezone.now()
                    job.save()
                    
                except Document.DoesNotExist:
                    job.status = 'failed'
                    job.error_details = 'Document not found'
                    job.save()
        
        return JsonResponse({
            'success': True,
            'message': _('Job restarted successfully'),
            'new_status': job.status
        })
        
    except Exception as e:
        logger.error(f"Job retry error: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

def bulk_actions(request):
    """Handle bulk actions on documents or QA entries"""
    if request.method == 'POST':
        action = request.POST.get('action')
        content_type = request.POST.get('content_type')  # 'document' or 'qa_entry'
        selected_ids = request.POST.getlist('selected_ids')
        
        if not action or not selected_ids:
            messages.error(request, _('No action or items selected'))
            return redirect(request.META.get('HTTP_REFERER', '/'))
        
        try:
            if content_type == 'document':
                documents = Document.objects.filter(id__in=selected_ids)
                
                if action == 'delete':
                    count = documents.count()
                    for doc in documents:
                        # Delete file
                        if doc.original_file and os.path.exists(doc.original_file.path):
                            try:
                                os.remove(doc.original_file.path)
                            except:
                                pass
                    documents.delete()
                    messages.success(request, _('Deleted {count} documents').format(count=count))
                
                elif action == 'reprocess':
                    count = 0
                    for doc in documents:
                        if doc.processing_status != 'processing':
                            try:
                                doc.processing_status = 'pending'
                                doc.save()
                                result = doc_processor.process_document(doc)
                                if result['success']:
                                    count += 1
                            except:
                                pass
                    messages.success(request, _('Reprocessed {count} documents').format(count=count))
                
                elif action == 'update_category':
                    new_category = request.POST.get('new_category', '')
                    count = documents.update(category=new_category)
                    messages.success(request, _('Updated category for {count} documents').format(count=count))
            
            elif content_type == 'qa_entry':
                entries = QAEntry.objects.filter(id__in=selected_ids)
                
                if action == 'delete':
                    count = entries.count()
                    entries.delete()
                    messages.success(request, _('Deleted {count} QA entries').format(count=count))
                
                elif action == 'update_category':
                    new_category = request.POST.get('new_category', '')
                    count = entries.update(category=new_category)
                    messages.success(request, _('Updated category for {count} QA entries').format(count=count))
                
                elif action == 'regenerate_chunks':
                    count = 0
                    for entry in entries:
                        try:
                            # Regenerate text chunk
                            chunk_text = f"SKU: {entry.sku}\nQuestion: {entry.question}\nAnswer: {entry.answer}"
                            if entry.category:
                                chunk_text += f"\nCategory: {entry.category}"
                            
                            TextChunk.objects.update_or_create(
                                qa_entry=entry,
                                defaults={
                                    'text': chunk_text,
                                    'chunk_index': 0,
                                    'chunk_size': len(chunk_text),
                                    'keywords': entry.keywords or ''
                                }
                            )
                            count += 1
                        except:
                            pass
                    messages.success(request, _('Regenerated chunks for {count} QA entries').format(count=count))
            
        except Exception as e:
            logger.error(f"Bulk action error: {str(e)}")
            messages.error(request, _('Error performing bulk action: {error}').format(error=str(e)))
    
    return redirect(request.META.get('HTTP_REFERER', '/'))

# Helper functions
def format_enhanced_search_results(results: list) -> list:
    """Format enhanced search results for display"""
    formatted_results = []
    
    for result in results:
        if result['type'] == 'qa_entry':
            entry = result['entry']
            formatted_result = {
                'id': entry.id,
                'type': 'qa_entry',
                'sku': entry.sku,
                'question': entry.question,
                'answer': entry.answer,
                'image_link': entry.image_link,
                'category': entry.category or 'general',
                'relevance_score': result['score'],
                'match_type': result['match_type'],
                'match_reason': result['match_reason'],
                'has_image': bool(entry.image_link),
                'display_score': f"{result['score']:.2%}" if result['score'] <= 1 else f"{result['score']:.2f}",
                'source_info': result['source_info'],
                'created_at': entry.created_at,
                'updated_at': entry.updated_at
            }
        
        elif result['type'] == 'document_chunk':
            chunk = result['chunk']
            document = result['document']
            formatted_result = {
                'id': f"chunk_{chunk.id}",
                'type': 'document_chunk',
                'document_id': document.id,
                'document_title': document.title,
                'document_type': document.document_type,
                'text_content': chunk.text,
                'page_number': chunk.page_number,
                'chunk_index': chunk.chunk_index,
                'relevance_score': result['score'],
                'match_type': result['match_type'],
                'match_reason': result['match_reason'],
                'display_score': f"{result['score']:.2%}" if result['score'] <= 1 else f"{result['score']:.2f}",
                'source_info': result['source_info'],
                'context_before': chunk.context_before,
                'context_after': chunk.context_after,
                'category': document.category or 'general',
                'created_at': document.created_at
            }
        
        else:
            continue  # Skip unknown types
        
        # Add formatted category for display
        formatted_result['category_display'] = formatted_result['category'].replace('_', ' ').title()
        
        # Add relevance level
        score = formatted_result['relevance_score']
        if score >= 0.7:
            formatted_result['relevance_level'] = 'high'
        elif score >= 0.3:
            formatted_result['relevance_level'] = 'medium'
        else:
            formatted_result['relevance_level'] = 'low'
        
        formatted_results.append(formatted_result)
    
    return formatted_results

def get_search_suggestions(request):
    """Get search suggestions based on query history"""
    query = request.GET.get('q', '').strip()
    
    if len(query) < 2:
        return JsonResponse({'suggestions': []})
    
    try:
        # Get recent queries that match
        recent_queries = SemanticQuery.objects.filter(
            Q(query_text__icontains=query) | Q(processed_query__icontains=query)
        ).values('processed_query').annotate(
            count=Count('id')
        ).order_by('-count')[:10]
        
        # Get popular SKUs that match
        skus = QAEntry.objects.filter(
            sku__icontains=query
        ).values_list('sku', flat=True).distinct()[:5]
        
        suggestions = []
        
        # Add query suggestions
        for q in recent_queries:
            suggestions.append({
                'text': q['processed_query'],
                'type': 'query',
                'count': q['count']
            })
        
        # Add SKU suggestions
        for sku in skus:
            suggestions.append({
                'text': sku,
                'type': 'sku',
                'count': 0
            })
        
        return JsonResponse({'suggestions': suggestions})
        
    except Exception as e:
        logger.error(f"Search suggestions error: {str(e)}")
        return JsonResponse({'suggestions': []})

# Error handlers
def custom_404(request, exception):
    """Custom 404 error handler"""
    return render(request, 'semantic_qa/errors/404.html', status=404)

def custom_500(request):
    """Custom 500 error handler"""
    return render(request, 'semantic_qa/errors/500.html', status=500)