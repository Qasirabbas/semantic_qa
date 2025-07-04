# semantic_qa/services/document_processor.py
import logging
import os
import tempfile
from typing import Dict, List, Optional, Tuple
from io import BytesIO
import hashlib

logger = logging.getLogger('semantic_qa')

# semantic_qa/services/document_processor.py
import logging
import os
import tempfile
from typing import Dict, List, Optional, Tuple
from io import BytesIO
import hashlib
import re

logger = logging.getLogger('semantic_qa')

class DocumentProcessor:
    """Simple document processor for basic functionality"""
    
    def __init__(self):
        self.ocr_reader = None
        self._ocr_initialized = False
        logger.info("Document processor initialized (basic version)")
    
    def _initialize_ocr(self):
        """Initialize OCR if available - Fixed version"""
        if self._ocr_initialized:
            return
            
        try:
            import easyocr
            # Initialize with English and Chinese support
            self.ocr_reader = easyocr.Reader(['en', 'ch_sim'], gpu=False)
            self._ocr_initialized = True
            logger.info("✅ EasyOCR initialized successfully")
        except ImportError:
            logger.warning("⚠️ EasyOCR not available - install with: pip install easyocr")
            self.ocr_reader = None
            self._ocr_initialized = True
        except Exception as e:
            logger.error(f"❌ Failed to initialize OCR: {str(e)}")
            self.ocr_reader = None
            self._ocr_initialized = True
    
    def process_document(self, document, job=None) -> Dict:
        """
        Main document processing method - routes to appropriate processor
        FIXED: Removed recursion bug for PDF processing
        """
        try:
            if not document:
                return {'success': False, 'error': 'No document provided'}
            
            logger.info(f"Processing document: {document.title} (Type: {document.document_type})")
            
            # Mark document as processing
            document.processing_status = 'processing'
            document.save()
            
            # Route to appropriate processor - FIXED ROUTING
            if document.document_type == 'pdf':
                result = self.process_pdf(document, job)  # ✅ FIXED: Call process_pdf, not process_document
            elif document.document_type == 'image':
                result = self.process_image(document, job)
            elif document.document_type == 'link':
                result = self.process_link(document, job)
            else:
                return {'success': False, 'error': f'Unsupported document type: {document.document_type}'}
            
            if result['success']:
                # Update document fields one by one to avoid recursion
                extracted_text = result['extracted_text']
                
                document.extracted_text = extracted_text
                document.processing_status = 'completed'
                document.language_detected = self.detect_language(extracted_text)
                document.tags = self.extract_keywords(extracted_text)
                document.category = self.auto_categorize(extracted_text)
                
                # Update metadata separately
                if 'metadata' in result:
                    if not document.metadata:
                        document.metadata = {}
                    document.metadata.update(result['metadata'])
                
                # Save all changes at once
                document.save()
                
                if job:
                    job.update_progress(80, "Creating text chunks")
                
                # Create text chunks
                chunks_created = 0
                if extracted_text.strip():
                    chunks_created = self.create_text_chunks(document, extracted_text)
                
                result['chunks_created'] = chunks_created
                
                if job:
                    job.update_progress(100, "Processing completed")
                
                logger.info(f"✅ Document processing completed: {document.title}")
                return result
            else:
                # Mark document as failed
                document.processing_status = 'failed'
                document.save()
                
                logger.error(f"❌ Document processing failed: {document.title} - {result.get('error', 'Unknown error')}")
                return result
                
        except Exception as e:
            # Mark document as failed
            document.processing_status = 'failed'
            document.save()
            
            logger.error(f"❌ Document processing error: {str(e)}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            return {'success': False, 'error': str(e)}
    
    def process_pdf(self, document, job=None) -> Dict:
        
        """
        Process PDF files - Extract text from PDF documents
        RENAMED: This method was being called incorrectly as process_document
        """
        try:
            logger.info(f"📄 Starting PDF processing for document {document.id}: {document.title}")
            
            if job:
                job.update_progress(20, "Loading PDF file")
            
            if not document.original_file:
                logger.error("❌ No PDF file provided")
                return {'success': False, 'error': 'No PDF file provided'}
            
            # Get PDF file path
            pdf_path = document.original_file.path
            logger.info(f"📁 PDF file path: {pdf_path}")
            
            if not os.path.exists(pdf_path):
                logger.error(f"❌ PDF file not found: {pdf_path}")
                return {'success': False, 'error': 'PDF file not found on disk'}
            
            # Initialize PDF processing
            try:
                import fitz  # PyMuPDF
            except ImportError:
                logger.error("❌ PyMuPDF not installed")
                return {
                    'success': False,
                    'error': 'PyMuPDF not installed. Please install with: pip install PyMuPDF'
                }
            
            pdf_document = None
            try:
                # Open PDF document
                logger.info(f"📖 Opening PDF: {pdf_path}")
                pdf_document = fitz.open(pdf_path)
                page_count = len(pdf_document)
                
                if job:
                    job.update_progress(40, f"Processing {page_count} pages")
                
                logger.info(f"Processing PDF with {page_count} pages")
                
                # Extract text from all pages
                extracted_text = ""
                page_texts = []
                
                for page_num in range(page_count):
                    try:
                        page = pdf_document[page_num]
                        page_text = page.get_text()
                        
                        if page_text.strip():
                            extracted_text += page_text + "\n"
                            page_texts.append({
                                'page_number': page_num + 1,
                                'text': page_text.strip()
                            })
                        
                        # Update progress
                        if job and page_count > 1:
                            progress = 40 + int((page_num + 1) / page_count * 40)
                            job.update_progress(progress, f"Processing page {page_num + 1}/{page_count}")
                            
                    except Exception as page_error:
                        logger.warning(f"Error processing page {page_num + 1}: {str(page_error)}")
                        continue
                
                # Update document metadata
                document.page_count = page_count
                
                metadata = {
                    'pages': page_texts,
                    'page_count': page_count,
                    'text_extraction_method': 'pymupdf',
                    'has_images': False,
                    'file_size': document.file_size or 0
                }
                
                logger.info(f"✅ Successfully extracted {len(extracted_text)} characters from {page_count}-page PDF")
                
                return {
                    'success': True,
                    'extracted_text': extracted_text.strip(),
                    'metadata': metadata
                }
                
            finally:
                # Always close the document
                if pdf_document and not pdf_document.is_closed:
                    pdf_document.close()
            
        except Exception as e:
            logger.error(f"❌ PDF processing error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def process_image(self, document, job=None) -> Dict:
        """Process image files with OCR - Enhanced for Chinese text with low confidence"""
        try:
            logger.info(f"🖼️ Starting image processing for document {document.id}: {document.title}")
            
            if job:
                job.update_progress(20, "Loading image file")

            
            if not document.original_file:
                logger.error("❌ No image file provided")
                return {'success': False, 'error': 'No image file provided'}
            
            # Get image path and validate
            image_path = document.original_file.path
            logger.info(f"📁 Image file path: {image_path}")
            
            if not os.path.exists(image_path):
                logger.error(f"❌ Image file not found: {image_path}")
                return {'success': False, 'error': 'Image file not found on disk'}
            
            # Initialize OCR exactly like your working script
            if not self.ocr_reader:
                logger.info("🔧 Initializing EasyOCR reader...")
                try:
                    import easyocr
                    # Initialize exactly like your working script
                    self.ocr_reader = easyocr.Reader(['en', 'ch_sim'])
                    self._ocr_initialized = True
                    logger.info("✅ EasyOCR initialized successfully")
                except Exception as e:
                    logger.error(f"❌ Failed to initialize OCR: {str(e)}")
                    return {
                        'success': False,
                        'error': f'Failed to initialize OCR: {str(e)}'
                    }
            
            if job:
                job.update_progress(40, "Performing OCR on image")
            
            # Perform OCR exactly like your working script
            try:
                logger.info(f"🤖 Starting OCR processing on: {image_path}")
                
                # Use the same method as your working script
                results = self.ocr_reader.readtext(image_path)
                
                logger.info(f"✅ OCR completed. Found {len(results)} text regions")
                
            except Exception as ocr_error:
                logger.error(f"❌ OCR processing failed: {str(ocr_error)}")
                return {'success': False, 'error': f'OCR failed: {str(ocr_error)}'}
            
            if job:
                job.update_progress(70, "Processing OCR results")
            
            # Process results with LOWER confidence threshold for Chinese text
            extracted_text = ""
            text_blocks = []
            total_confidence = 0
            
            # Analyze all results first to determine if image has mostly Chinese text
            chinese_char_count = 0
            total_char_count = 0
            
            for bbox, text, confidence in results:
                for char in text:
                    total_char_count += 1
                    if '\u4e00' <= char <= '\u9fff':  # Chinese character range
                        chinese_char_count += 1
            
            # Determine if this is primarily Chinese text
            is_chinese_dominant = (chinese_char_count / max(total_char_count, 1)) > 0.1
            
            # Set confidence threshold based on text type
            if is_chinese_dominant:
                confidence_threshold = 0.001  # Very low threshold for Chinese
                logger.info(f"🇨🇳 Chinese text detected, using low confidence threshold: {confidence_threshold}")
            else:
                confidence_threshold = 0.3  # Normal threshold for English
                logger.info(f"🇺🇸 English text detected, using normal confidence threshold: {confidence_threshold}")
            
            logger.info(f"📊 Processing {len(results)} OCR results with threshold {confidence_threshold}...")
            
            for i, (bbox, text, confidence) in enumerate(results):
                logger.info(f"  Text {i+1}: '{text}' (confidence: {confidence:.6f})")
                
                # Use dynamic confidence threshold
                if confidence >= confidence_threshold and text.strip():
                    text_blocks.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': bbox
                    })
                    extracted_text += text + " "
                    total_confidence += confidence
                    logger.info(f"    ✅ ACCEPTED: '{text}' (confidence: {confidence:.6f})")
                else:
                    logger.info(f"    ❌ REJECTED: '{text}' (confidence: {confidence:.6f} < {confidence_threshold})")
            
            # If no text was accepted with the threshold, try accepting all non-empty text
            if len(text_blocks) == 0:
                logger.warning("⚠️ No text accepted with confidence threshold, accepting all non-empty results...")
                for bbox, text, confidence in results:
                    if text.strip():
                        text_blocks.append({
                            'text': text,
                            'confidence': confidence,
                            'bbox': bbox
                        })
                        extracted_text += text + " "
                        total_confidence += confidence
                        logger.info(f"    ✅ FALLBACK ACCEPTED: '{text}' (confidence: {confidence:.6f})")
            
            # Calculate average confidence
            avg_confidence = total_confidence / len(text_blocks) if text_blocks else 0
            
            # Get image info
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    image_size = img.size
            except Exception as img_error:
                logger.warning(f"Could not get image dimensions: {img_error}")
                image_size = 'unknown'
            
            metadata = {
                'image_size': image_size,
                'ocr_confidence': avg_confidence,
                'text_blocks': len(text_blocks),
                'ocr_method': 'easyocr_chinese_enhanced',
                'file_size': document.file_size or 0,
                'total_ocr_results': len(results),
                'confidence_threshold': confidence_threshold,
                'is_chinese_dominant': is_chinese_dominant,
                'chinese_char_ratio': chinese_char_count / max(total_char_count, 1)
            }
            
            if text_blocks:
                logger.info(f"✅ OCR completed successfully:")
                logger.info(f"   📝 {len(text_blocks)} text blocks extracted")
                logger.info(f"   📊 Average confidence: {avg_confidence:.6f}")
                logger.info(f"   📄 Total text length: {len(extracted_text)} characters")
                logger.info(f"   🇨🇳 Chinese text dominant: {is_chinese_dominant}")
                logger.info(f"   🎯 Extracted text: '{extracted_text.strip()}'")
            else:
                logger.warning("⚠️ No text detected in image")
                extracted_text = f"Image processed but no readable text detected. File: {document.title}"
            
            return {
                'success': True,
                'extracted_text': extracted_text.strip(),
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"❌ Image processing error: {str(e)}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            return {'success': False, 'error': str(e)}
    
    def process_link(self, document, job=None) -> Dict:
        """Enhanced web scraping with better error handling"""
        try:
            if job:
                job.update_progress(20, "Fetching web content")
            
            if not document.source_url:
                return {'success': False, 'error': 'No URL provided'}
            
            logger.info(f"Processing web link: {document.source_url}")
            
            try:
                import requests
                from bs4 import BeautifulSoup
                import time
            except ImportError:
                return {
                    'success': False,
                    'error': 'Required packages not installed. Install with: pip install requests beautifulsoup4'
                }
            
            # Enhanced headers to avoid blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7',
                'Accept-Encoding': 'gzip, deflate, br',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            # Add small delay to be respectful
            time.sleep(1)
            
            try:
                response = requests.get(
                    document.source_url, 
                    headers=headers, 
                    timeout=30,
                    allow_redirects=True,
                    verify=True  # SSL verification
                )
                response.raise_for_status()
            except requests.exceptions.SSLError:
                # Retry without SSL verification as fallback
                logger.warning("SSL verification failed, retrying without verification")
                response = requests.get(
                    document.source_url, 
                    headers=headers, 
                    timeout=30,
                    allow_redirects=True,
                    verify=False
                )
                response.raise_for_status()
            
            if job:
                job.update_progress(50, "Parsing HTML content")
            
            # Detect encoding
            response.encoding = response.apparent_encoding
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe', 'noscript']):
                element.decompose()
            
            # Extract title
            title = ""
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text().strip()
                if not document.title.strip():
                    document.title = title[:200]  # Limit title length
                    document.save()
            
            # Try to find main content area
            main_content = None
            
            # Try different selectors for main content
            content_selectors = [
                'main', 'article', '[role="main"]', '.main-content', 
                '.content', '.post-content', '.entry-content', '#content'
            ]
            
            for selector in content_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            # Fallback to body if no main content found
            if not main_content:
                main_content = soup.find('body') or soup
            
            # Extract text with better formatting
            extracted_text = ""
            
            # Get paragraphs and headings
            for element in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'div']):
                text = element.get_text(separator=' ', strip=True)
                if len(text) > 20:  # Only meaningful text
                    extracted_text += text + "\n\n"
            
            # Clean up text
            lines = extracted_text.split('\n')
            meaningful_lines = []
            
            for line in lines:
                line = line.strip()
                if len(line) > 10 and not line.startswith('http'):  # Filter out URLs and short lines
                    meaningful_lines.append(line)
            
            extracted_text = '\n'.join(meaningful_lines)
            
            if len(extracted_text.strip()) < 100:
                return {
                    'success': False,
                    'error': 'Insufficient content found on the web page'
                }
            
            # Extract metadata
            metadata = {
                'url': document.source_url,
                'title': title,
                'description': '',
                'content_length': len(extracted_text),
                'status_code': response.status_code
            }
            
            # Get meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
            if meta_desc:
                metadata['description'] = meta_desc.get('content', '').strip()[:500]
            
            # Get language
            html_lang = soup.find('html')
            if html_lang and html_lang.get('lang'):
                metadata['language'] = html_lang.get('lang')[:10]
            
            logger.info(f"✅ Web scraping completed: {len(extracted_text)} characters extracted")
            
            return {
                'success': True,
                'extracted_text': extracted_text,
                'metadata': metadata
            }
            
        except requests.exceptions.Timeout:
            return {'success': False, 'error': 'Request timeout - website took too long to respond'}
        except requests.exceptions.ConnectionError:
            return {'success': False, 'error': 'Connection error - unable to reach the website'}
        except requests.exceptions.HTTPError as e:
            return {'success': False, 'error': f'HTTP error {e.response.status_code}: {str(e)}'}
        except Exception as e:
            logger.error(f"❌ Web scraping error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def create_text_chunks(self, document, text: str) -> int:
        """Create text chunks for RAG retrieval - Fixed recursion issue"""
        try:
            # Import here to avoid circular import
            from semantic_qa.models import TextChunk
            
            # Simple chunking strategy
            chunk_size = 1000
            chunk_overlap = 200
            
            if len(text) <= chunk_size:
                # Text is small enough for one chunk
                chunks = [text]
            else:
                # Split into overlapping chunks
                chunks = []
                start = 0
                while start < len(text):
                    end = start + chunk_size
                    if end >= len(text):
                        chunks.append(text[start:])
                        break
                    else:
                        # Find a good breaking point (end of sentence or paragraph)
                        break_point = text.rfind('.', start, end)
                        if break_point == -1:
                            break_point = text.rfind(' ', start, end)
                        if break_point == -1:
                            break_point = end
                        
                        chunks.append(text[start:break_point])
                        start = break_point - chunk_overlap
                        if start < 0:
                            start = 0
            
            # Delete existing chunks to avoid duplicates
            try:
                TextChunk.objects.filter(document=document).delete()
            except Exception as delete_error:
                logger.warning(f"Could not delete existing chunks: {str(delete_error)}")
            
            # Create new chunks
            chunks_created = 0
            for i, chunk_text in enumerate(chunks):
                if len(chunk_text.strip()) < 20:  # Skip very short chunks
                    continue
                
                try:
                    # Extract context
                    chunk_start = text.find(chunk_text)
                    context_before = ""
                    context_after = ""
                    
                    if chunk_start > 0:
                        context_start = max(0, chunk_start - 100)
                        context_before = text[context_start:chunk_start][-100:]
                    
                    chunk_end = chunk_start + len(chunk_text)
                    if chunk_end < len(text):
                        context_end = min(len(text), chunk_end + 100)
                        context_after = text[chunk_end:context_end][:100]
                    
                    # Determine page number for PDFs (simple estimation)
                    page_number = None
                    if document.document_type == 'pdf' and hasattr(document, 'page_count') and document.page_count:
                        text_position = chunk_start / len(text)
                        page_number = int(text_position * document.page_count) + 1
                    
                    # Extract keywords for this chunk
                    chunk_keywords = self.extract_keywords(chunk_text)
                    
                    # Create text chunk - avoid any potential recursion
                    TextChunk.objects.create(
                        document_id=document.id,  # Use ID instead of object
                        text=chunk_text,
                        chunk_index=i,
                        chunk_size=len(chunk_text),
                        page_number=page_number,
                        context_before=context_before,
                        context_after=context_after,
                        keywords=chunk_keywords
                    )
                    
                    chunks_created += 1
                    
                except Exception as chunk_error:
                    logger.error(f"Error creating chunk {i}: {str(chunk_error)}")
                    continue
            
            logger.info(f"Created {chunks_created} text chunks for document {document.id}")
            return chunks_created
            
        except Exception as e:
            logger.error(f"Error creating text chunks: {str(e)}")
            return 0
    
    def detect_language(self, text: str) -> str:
        """Simple language detection"""
        if not text:
            return 'unknown'
        
        # Count Chinese characters
        chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
        total_chars = len([c for c in text if c.isalpha()])
        
        if total_chars == 0:
            return 'unknown'
        
        chinese_ratio = chinese_chars / total_chars
        
        if chinese_ratio > 0.3:
            return 'zh'
        else:
            return 'en'
    
    def extract_keywords(self, text: str) -> str:
        """Extract keywords from text - basic implementation"""
        if not text:
            return ""
        
        # Simple keyword extraction
        import re
        
        # Remove punctuation and split into words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'this', 'that', 'these', 'those'
        }
        
        # Get unique words that are not stop words and longer than 2 characters
        keywords = list(set([
            word for word in words 
            if word not in stop_words and len(word) > 2
        ]))
        
        # Return top 20 keywords
        return ' '.join(keywords[:20])
    
    def auto_categorize(self, text: str) -> str:
        """Auto-categorize document based on content"""
        if not text:
            return 'general'
        
        text_lower = text.lower()
        
        # Category keywords
        categories = {
            'installation': ['install', 'setup', 'mount', 'assembly'],
            'troubleshooting': ['problem', 'issue', 'error', 'fix', 'repair'],
            'maintenance': ['maintain', 'service', 'clean', 'replace'],
            'specifications': ['spec', 'specification', 'dimension', 'size'],
            'manual': ['manual', 'guide', 'instruction', 'handbook'],
            'technical': ['technical', 'engineering', 'circuit', 'diagram'],
            'safety': ['safety', 'warning', 'caution', 'danger'],
        }
        
        # Count matches for each category
        category_scores = {}
        for category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                category_scores[category] = score
        
        # Return category with highest score
        if category_scores:
            return max(category_scores, key=category_scores.get)
        else:
            return 'general'