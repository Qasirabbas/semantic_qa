# semantic_qa/forms.py
from django import forms
from django.utils.translation import gettext_lazy as _
from django.core.validators import URLValidator, FileExtensionValidator
from django.core.exceptions import ValidationError
from django.conf import settings
from .models import QAEntry, SystemConfig, Document, TextChunk
import re
import os
from urllib.parse import urlparse

class QAEntryForm(forms.ModelForm):
    """Form for creating/editing QA entries"""
    
    class Meta:
        model = QAEntry
        fields = ['sku', 'question', 'answer', 'image_link', 'category', 'keywords']
        widgets = {
            'sku': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': _('e.g., ABC123'),
                'maxlength': 200
            }),
            'question': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': _('Enter the question...'),
                'maxlength': 2000
            }),
            'answer': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 5,
                'placeholder': _('Enter the answer...'),
                'maxlength': 5000
            }),
            'image_link': forms.URLInput(attrs={
                'class': 'form-control',
                'placeholder': _('https://example.com/image.jpg')
            }),
            'category': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': _('e.g., installation, troubleshooting'),
                'maxlength': 100
            }),
            'keywords': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 2,
                'placeholder': _('space-separated keywords for better search'),
                'maxlength': 1000
            })
        }
        labels = {
            'sku': _('SKU'),
            'question': _('Question'),
            'answer': _('Answer'),
            'image_link': _('Image Link'),
            'category': _('Category'),
            'keywords': _('Keywords')
        }
        help_texts = {
            'sku': _('Product SKU or identifier (required)'),
            'question': _('The question being asked (required)'),
            'answer': _('The answer to the question (required)'),
            'image_link': _('URL to an image that helps answer the question (optional)'),
            'category': _('Category for organizing questions (auto-detected if empty)'),
            'keywords': _('Keywords to improve search results (auto-generated if empty)')
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add Bootstrap classes and validation
        for field_name, field in self.fields.items():
            if not field.widget.attrs.get('class'):
                field.widget.attrs['class'] = 'form-control'
            
            # Add data attributes for client-side validation
            if field.required:
                field.widget.attrs['required'] = True
                field.widget.attrs['data-required'] = 'true'
        
        # Make some fields optional
        self.fields['image_link'].required = False
        self.fields['category'].required = False
        self.fields['keywords'].required = False
        
        # Add autocomplete for category
        existing_categories = QAEntry.objects.exclude(category='').values_list('category', flat=True).distinct()
        if existing_categories:
            category_list = '|'.join(existing_categories)
            self.fields['category'].widget.attrs['data-categories'] = category_list

    def clean_sku(self):
        sku = self.cleaned_data.get('sku')
        if sku:
            sku = sku.strip().upper()
            # Validate SKU format (alphanumeric with common separators)
            if not re.match(r'^[A-Z0-9\-_.]+$', sku):
                raise forms.ValidationError(_('SKU can only contain letters, numbers, hyphens, underscores, and dots'))
        return sku

    def clean_question(self):
        question = self.cleaned_data.get('question')
        if question:
            question = question.strip()
            if len(question) < 10:
                raise forms.ValidationError(_('Question must be at least 10 characters long'))
        return question

    def clean_answer(self):
        answer = self.cleaned_data.get('answer')
        if answer:
            answer = answer.strip()
            if len(answer) < 10:
                raise forms.ValidationError(_('Answer must be at least 10 characters long'))
        return answer

    def clean_image_link(self):
        image_link = self.cleaned_data.get('image_link')
        if image_link:
            # Clean and fix the URL
            image_link = clean_image_url(image_link.strip())
            
            # Basic URL validation
            if not image_link.startswith(('http://', 'https://')):
                raise forms.ValidationError(_('Image link must start with http:// or https://'))
            
            # Validate URL format
            try:
                validator = URLValidator()
                validator(image_link)
            except ValidationError:
                raise forms.ValidationError(_('Please enter a valid image URL'))
            
            # Check if URL looks like an image
            parsed_url = urlparse(image_link)
            path = parsed_url.path.lower()
            image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg', '.tiff']
            
            if not any(path.endswith(ext) for ext in image_extensions) and 'image' not in image_link.lower():
                raise forms.ValidationError(_('URL should point to an image file'))
        
        return image_link

    def clean_keywords(self):
        keywords = self.cleaned_data.get('keywords')
        if keywords:
            keywords = keywords.strip()
            # Basic keyword validation - remove excessive whitespace
            keywords = re.sub(r'\s+', ' ', keywords)
        return keywords

class DocumentUploadForm(forms.ModelForm):
    """Form for uploading documents (PDF, images, links)"""
    
    class Meta:
        model = Document
        fields = ['title', 'document_type', 'original_file', 'source_url', 'category']
        widgets = {
            'title': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': _('Enter document title'),
                'maxlength': 255
            }),
            'document_type': forms.Select(attrs={
                'class': 'form-control'
            }),
            'original_file': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.pdf,.jpg,.jpeg,.png,.gif,.bmp,.tiff,.webp'
            }),
            'source_url': forms.URLInput(attrs={
                'class': 'form-control',
                'placeholder': _('https://example.com/page')
            }),
            'category': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': _('Document category (optional)'),
                'maxlength': 100
            })
        }
        help_texts = {
            'title': _('Descriptive title for the document (auto-generated if empty)'),
            'document_type': _('Type of document being uploaded'),
            'original_file': _('Upload PDF or image file (max 50MB)'),
            'source_url': _('URL for web links (leave file empty if using URL)'),
            'category': _('Optional category for organization')
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Make fields conditionally required
        self.fields['original_file'].required = False
        self.fields['source_url'].required = False
        self.fields['category'].required = False
        self.fields['title'].required = False
        
        # Add data attributes for dynamic form behavior
        self.fields['document_type'].widget.attrs['data-toggle'] = 'document-type-select'
        
        # Get existing categories for autocomplete
        existing_categories = Document.objects.exclude(category='').values_list('category', flat=True).distinct()
        if existing_categories:
            category_list = '|'.join(existing_categories)
            self.fields['category'].widget.attrs['data-categories'] = category_list

    def clean(self):
        cleaned_data = super().clean()
        document_type = cleaned_data.get('document_type')
        original_file = cleaned_data.get('original_file')
        source_url = cleaned_data.get('source_url')

        # Validation based on document type
        if document_type == 'link':
            if not source_url:
                raise forms.ValidationError(_('URL is required for link documents'))
            if original_file:
                raise forms.ValidationError(_('File upload not needed for link documents'))
        else:
            if not original_file:
                raise forms.ValidationError(_('File upload is required for this document type'))
            if source_url:
                raise forms.ValidationError(_('URL not needed for file uploads'))

        return cleaned_data

    def clean_original_file(self):
        file = self.cleaned_data.get('original_file')
        
        if file:
            # Check file size (get from config or default to 50MB)
            try:
                max_size = int(SystemConfig.get_config('max_file_size_mb', '50')) * 1024 * 1024
            except:
                max_size = 50 * 1024 * 1024
                
            if file.size > max_size:
                raise forms.ValidationError(
                    _('File size must be less than {size}MB').format(size=max_size // (1024 * 1024))
                )
            
            # Check file type based on document type
            document_type = self.data.get('document_type')
            file_extension = '.' + file.name.split('.')[-1].lower()
            
            allowed_extensions = {
                'pdf': ['.pdf'],
                'image': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
            }
            
            if document_type in allowed_extensions:
                if file_extension not in allowed_extensions[document_type]:
                    raise forms.ValidationError(
                        _('Invalid file type for {document_type}. Allowed: {extensions}').format(
                            document_type=document_type,
                            extensions=', '.join(allowed_extensions[document_type])
                        )
                    )
            
            # Additional validation for images
            if document_type == 'image':
                try:
                    from PIL import Image
                    import io
                    
                    # Try to open image to validate it's not corrupted
                    image_data = file.read()
                    image = Image.open(io.BytesIO(image_data))
                    image.verify()
                    
                    # Reset file pointer
                    file.seek(0)
                    
                    # Check image dimensions (reasonable limits)
                    if image.size[0] > 10000 or image.size[1] > 10000:
                        raise forms.ValidationError(_('Image dimensions too large (max 10000x10000)'))
                    
                except Exception as e:
                    raise forms.ValidationError(_('Invalid or corrupted image file'))
        
        return file

    def clean_source_url(self):
        url = self.cleaned_data.get('source_url')
        
        if url:
            url = url.strip()
            
            # Validate URL format
            try:
                validator = URLValidator()
                validator(url)
            except ValidationError:
                raise forms.ValidationError(_('Please enter a valid URL'))
            
            # Check URL accessibility (basic check)
            parsed_url = urlparse(url)
            if not parsed_url.scheme in ['http', 'https']:
                raise forms.ValidationError(_('URL must use HTTP or HTTPS'))
            
            if not parsed_url.netloc:
                raise forms.ValidationError(_('URL must include a domain name'))
        
        return url

class BatchDocumentUploadForm(forms.Form):
    """Simplified form for batch uploading - handles multiple files in view"""
    
    default_category = forms.CharField(
        required=False,
        max_length=100,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': _('Default category for all uploads')
        }),
        help_text=_('Optional default category applied to all uploaded documents')
    )
    
    auto_process = forms.BooleanField(
        initial=True,
        required=False,
        widget=forms.CheckboxInput(attrs={
            'class': 'form-check-input'
        }),
        help_text=_('Automatically start processing after upload')
    )
    
    overwrite_duplicates = forms.BooleanField(
        initial=False,
        required=False,
        widget=forms.CheckboxInput(attrs={
            'class': 'form-check-input'
        }),
        help_text=_('Overwrite existing documents with same name')
    )
    
    # Note: Files are handled directly in the view using request.FILES.getlist('files')
    # This avoids Django's widget limitations with multiple file uploads

    def clean_files(self):
        # This method will be called, but we'll handle validation in the view
        # since we're getting files from request.FILES.getlist('files')
        return self.cleaned_data.get('files')

class ExcelUploadForm(forms.Form):
    """Form for uploading Excel files"""
    
    excel_file = forms.FileField(
        label=_('Excel File'),
        help_text=_('Upload an Excel file (.xlsx or .xls) with QA data'),
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': '.xlsx,.xls'
        }),
        validators=[FileExtensionValidator(allowed_extensions=['xlsx', 'xls'])]
    )
    
    overwrite_existing = forms.BooleanField(
        label=_('Overwrite Existing Entries'),
        help_text=_('If checked, existing entries with same SKU+Question will be updated'),
        required=False,
        widget=forms.CheckboxInput(attrs={
            'class': 'form-check-input'
        })
    )
    
    create_chunks = forms.BooleanField(
        label=_('Create Text Chunks for RAG'),
        help_text=_('Generate text chunks for enhanced search capabilities'),
        initial=True,
        required=False,
        widget=forms.CheckboxInput(attrs={
            'class': 'form-check-input'
        })
    )

    def clean_excel_file(self):
        file = self.cleaned_data.get('excel_file')
        
        if file:
            # Check file extension
            if not file.name.lower().endswith(('.xlsx', '.xls')):
                raise forms.ValidationError(_('Please upload an Excel file (.xlsx or .xls)'))
            
            # Check file size (max 10MB for Excel)
            if file.size > 10 * 1024 * 1024:
                raise forms.ValidationError(_('File size must be less than 10MB'))
            
            # Try to validate Excel file structure
            try:
                import pandas as pd
                df = pd.read_excel(file, nrows=1)  # Read just first row to validate
                if df.empty:
                    raise forms.ValidationError(_('Excel file appears to be empty'))
                file.seek(0)  # Reset file pointer
            except Exception as e:
                raise forms.ValidationError(_('Invalid Excel file format'))
        
        return file

class EnhancedSearchForm(forms.Form):
    """Enhanced search form with document type filtering and RAG options"""
    
    query = forms.CharField(
        label=_('Search Query'),
        max_length=500,
        widget=forms.TextInput(attrs={
            'class': 'form-control form-control-lg',
            'placeholder': _('Enter SKU, question, keywords, or describe what you need...'),
            'autocomplete': 'off',
            'data-toggle': 'search-suggestions'
        })
    )
    
    language = forms.ChoiceField(
        label=_('Language'),
        choices=[],  # Will be populated in __init__
        required=False,
        widget=forms.Select(attrs={
            'class': 'form-control'
        })
    )
    
    # Content type filters
    search_qa_entries = forms.BooleanField(
        label=_('Search QA Database'),
        initial=True,
        required=False,
        widget=forms.CheckboxInput(attrs={
            'class': 'form-check-input',
            'data-toggle': 'content-type-filter'
        })
    )
    
    search_documents = forms.BooleanField(
        label=_('Search Documents'),
        initial=True,
        required=False,
        widget=forms.CheckboxInput(attrs={
            'class': 'form-check-input',
            'data-toggle': 'content-type-filter'
        })
    )
    
    document_types = forms.MultipleChoiceField(
        label=_('Document Types'),
        choices=[
            ('pdf', _('PDF Documents')),
            ('image', _('Images')),
            ('link', _('Web Pages')),
        ],
        initial=['pdf', 'image', 'link'],
        required=False,
        widget=forms.CheckboxSelectMultiple(attrs={
            'class': 'form-check-input'
        })
    )
    
    # RAG options
    use_rag = forms.BooleanField(
        label=_('Generate AI Answer'),
        initial=True,
        required=False,
        widget=forms.CheckboxInput(attrs={
            'class': 'form-check-input',
            'data-toggle': 'rag-option'
        }),
        help_text=_('Use AI to generate comprehensive answers from search results')
    )
    
    # Advanced options
    max_results = forms.IntegerField(
        label=_('Maximum Results'),
        initial=20,
        min_value=5,
        max_value=50,
        required=False,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'min': '5',
            'max': '50'
        })
    )
    
    similarity_threshold = forms.FloatField(
        label=_('Similarity Threshold'),
        initial=0.1,
        min_value=0.0,
        max_value=1.0,
        required=False,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'min': '0.0',
            'max': '1.0',
            'step': '0.1'
        }),
        help_text=_('Minimum similarity score for results (0.0 = very loose, 1.0 = exact match)')
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Get supported languages from config
        try:
            supported_languages = SystemConfig.get_config('supported_languages', 'en,zh,es,fr,de,ja').split(',')
        except:
            supported_languages = ['en', 'zh', 'es', 'fr', 'de', 'ja']
        
        language_choices = [
            ('', _('Auto-detect')),
            ('en', _('English')),
            ('zh', _('Chinese')),
            ('es', _('Spanish')),
            ('fr', _('French')),
            ('de', _('German')),
            ('ja', _('Japanese')),
        ]
        
        # Filter choices based on supported languages
        self.fields['language'].choices = [
            (code, name) for code, name in language_choices 
            if not code or code in supported_languages
        ]
        
        # Get current RAG setting
        try:
            rag_enabled = SystemConfig.get_config('enable_rag', 'True').lower() == 'true'
        except:
            rag_enabled = True
            
        if not rag_enabled:
            self.fields['use_rag'].initial = False
            self.fields['use_rag'].widget.attrs['disabled'] = True
            self.fields['use_rag'].help_text = _('RAG is currently disabled in system settings')

    def clean_query(self):
        query = self.cleaned_data.get('query')
        if query:
            query = query.strip()
            if len(query) < 2:
                raise forms.ValidationError(_('Search query must be at least 2 characters long'))
            if len(query) > 500:
                raise forms.ValidationError(_('Search query is too long (max 500 characters)'))
        return query

    def clean(self):
        cleaned_data = super().clean()
        search_qa_entries = cleaned_data.get('search_qa_entries')
        search_documents = cleaned_data.get('search_documents')
        
        # At least one content type must be selected
        if not search_qa_entries and not search_documents:
            raise forms.ValidationError(_('Please select at least one content type to search'))
        
        return cleaned_data

class SystemConfigForm(forms.Form):
    """Form for system configuration"""
    
    # ChatGLM Configuration
    chatglm_api_key = forms.CharField(
        label=_('ChatGLM API Key'),
        max_length=200,
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': _('Enter your ChatGLM API key')
        }),
        help_text=_('API key for ChatGLM service')
    )
    
    chatglm_model = forms.CharField(
        label=_('ChatGLM Model'),
        initial='glm-4-flash',
        max_length=100,
        widget=forms.TextInput(attrs={
            'class': 'form-control'
        }),
        help_text=_('ChatGLM model name to use')
    )
    
    chatglm_temperature = forms.FloatField(
        label=_('ChatGLM Temperature'),
        initial=0.7,
        min_value=0.0,
        max_value=2.0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.1'
        }),
        help_text=_('Controls randomness in responses (0.0 = deterministic, 2.0 = very random)')
    )
    
    # Embedding Configuration
    embedding_model = forms.CharField(
        label=_('Embedding Model'),
        initial='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        max_length=200,
        widget=forms.TextInput(attrs={
            'class': 'form-control'
        }),
        help_text=_('Hugging Face model for text embeddings')
    )
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Load current configuration values if SystemConfig is available
        try:
            config_mappings = {
                'chatglm_api_key': ('chatglm_api_key', 'a74b8073a98d4da4a066fc72095f58b0.gulObfhh7fnNcAmp'),
                'chatglm_model': ('chatglm_model', 'glm-4-flash'),
                'chatglm_temperature': ('chatglm_temperature', '0.7'),
                'embedding_model': ('embedding_model', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'),
            }
            
            for field_name, (config_key, default_value) in config_mappings.items():
                if field_name in self.fields:
                    try:
                        current_value = SystemConfig.get_config(config_key, default_value)
                        
                        # Type conversion based on field type
                        if isinstance(self.fields[field_name], forms.FloatField):
                            try:
                                current_value = float(current_value)
                            except ValueError:
                                current_value = float(default_value)
                        elif isinstance(self.fields[field_name], forms.IntegerField):
                            try:
                                current_value = int(current_value)
                            except ValueError:
                                current_value = int(default_value)
                        elif isinstance(self.fields[field_name], forms.BooleanField):
                            current_value = str(current_value).lower() == 'true'
                        
                        self.fields[field_name].initial = current_value
                    except:
                        # If there's any error accessing SystemConfig, use defaults
                        pass
        except:
            # If SystemConfig is not available, use defaults
            pass

    def save(self):
        """Save configuration to database"""
        try:
            config_mappings = {
                'chatglm_api_key': 'chatglm_api_key',
                'chatglm_model': 'chatglm_model',
                'chatglm_temperature': 'chatglm_temperature',
                'embedding_model': 'embedding_model',
            }
            
            for field_name, config_key in config_mappings.items():
                if field_name in self.cleaned_data:
                    value = self.cleaned_data[field_name]
                    SystemConfig.set_config(config_key, str(value))
        except Exception as e:
            # If saving fails, silently continue
            pass

class FilterForm(forms.Form):
    """Form for filtering QA entries and documents"""
    
    search = forms.CharField(
        label=_('Search'),
        required=False,
        max_length=200,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': _('Search in content...'),
            'data-toggle': 'live-search'
        })
    )
    
    content_type = forms.ChoiceField(
        label=_('Content Type'),
        choices=[
            ('', _('All Types')),
            ('qa_entry', _('QA Entries')),
            ('document', _('Documents')),
        ],
        required=False,
        widget=forms.Select(attrs={
            'class': 'form-control',
            'data-toggle': 'content-filter'
        })
    )
    
    category = forms.CharField(
        label=_('Category'),
        required=False,
        max_length=100,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': _('Filter by category'),
            'data-toggle': 'category-filter'
        })
    )

# Utility functions
def clean_image_url(url: str) -> str:
    """Clean and fix image URLs - ENHANCED VERSION"""
    if not url:
        return ""
    
    url = url.strip()
    
    # Fix common URL formatting issues
    
    # Case 1: Missing :// after protocol
    if 'httpsae01' in url:
        url = url.replace('httpsae01', 'https://ae01')
    elif 'httpae01' in url:
        url = url.replace('httpae01', 'http://ae01')
    elif 'httpswww' in url:
        url = url.replace('httpswww', 'https://www')
    elif 'httpwww' in url:
        url = url.replace('httpwww', 'http://www')
    
    # Case 2: Missing slash after domain - SPECIFIC FOR ALICDN
    if 'ae01.alicdn.comkf' in url:
        url = url.replace('ae01.alicdn.comkf', 'ae01.alicdn.com/kf/')
    
    # Case 3: General missing :// fix
    if url.startswith('https') and '://' not in url:
        url = url.replace('https', 'https://', 1)
    elif url.startswith('http') and '://' not in url:
        url = url.replace('http', 'http://', 1)
    
    # Case 4: URL doesn't start with protocol at all
    elif not url.startswith(('http://', 'https://')):
        if 'alicdn.com' in url or 'ae01' in url:
            url = 'https://' + url
        else:
            url = 'https://' + url
    
    return url