from django.core.management.base import BaseCommand
from semantic_qa.models import Document
from semantic_qa.document_processor import DocumentProcessor
import logging

class Command(BaseCommand):
    help = 'Debug OCR processing for image documents'
    
    def add_arguments(self, parser):
        parser.add_argument('document_id', type=int, help='Document ID to process')
        parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    def handle(self, *args, **options):
        if options['verbose']:
            logging.basicConfig(level=logging.DEBUG)
        
        document_id = options['document_id']
        
        try:
            document = Document.objects.get(id=document_id, document_type='image')
            self.stdout.write(f"Processing document: {document.title}")
            
            # Create processor and replace with debug version
            processor = DocumentProcessor()
            processor.process_image = self.debug_process_image.__get__(processor, DocumentProcessor)
            
            # Process
            result = processor.process_image(document)
            
            if result['success']:
                self.stdout.write(
                    self.style.SUCCESS(f"Success! Extracted: '{result['extracted_text']}'")
                )
            else:
                self.stdout.write(
                    self.style.ERROR(f"Failed: {result['error']}")
                )
                
        except Document.DoesNotExist:
            self.stdout.write(
                self.style.ERROR(f"Document {document_id} not found or not an image")
            )
    # Debug version of process_image function with detailed logging
def process_image(self, document, job=None) -> dict:
    """Process image files with OCR - Debug version with extensive logging"""
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
        
        # Initialize OCR
        if not self.ocr_reader:
            logger.info("🔧 Initializing EasyOCR reader...")
            try:
                import easyocr
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
        
        # Perform OCR
        try:
            logger.info(f"🤖 Starting OCR processing on: {image_path}")
            results = self.ocr_reader.readtext(image_path)
            logger.info(f"✅ OCR completed. Found {len(results)} text regions")
            
            # DEBUG: Print all raw OCR results
            logger.info("=" * 50)
            logger.info("DEBUG: All OCR Results:")
            for i, (bbox, text, confidence) in enumerate(results):
                logger.info(f"  Result {i+1}:")
                logger.info(f"    Text: '{text}'")
                logger.info(f"    Confidence: {confidence:.6f}")
                logger.info(f"    Text type: {type(text)}")
                logger.info(f"    Text length: {len(text)}")
                logger.info(f"    Text repr: {repr(text)}")
            logger.info("=" * 50)
            
        except Exception as ocr_error:
            logger.error(f"❌ OCR processing failed: {str(ocr_error)}")
            return {'success': False, 'error': f'OCR failed: {str(ocr_error)}'}
        
        if job:
            job.update_progress(70, "Processing OCR results")
        
        # Process results with detailed debugging
        extracted_text = ""
        text_blocks = []
        total_confidence = 0
        
        # Analyze for Chinese text
        chinese_char_count = 0
        total_char_count = 0
        
        for bbox, text, confidence in results:
            for char in text:
                total_char_count += 1
                if '\u4e00' <= char <= '\u9fff':  # Chinese character range
                    chinese_char_count += 1
        
        is_chinese_dominant = (chinese_char_count / max(total_char_count, 1)) > 0.3
        confidence_threshold = 0.001 if is_chinese_dominant else 0.3
        
        logger.info(f"🔍 Text Analysis:")
        logger.info(f"   Total characters: {total_char_count}")
        logger.info(f"   Chinese characters: {chinese_char_count}")
        logger.info(f"   Chinese dominant: {is_chinese_dominant}")
        logger.info(f"   Confidence threshold: {confidence_threshold}")
        
        # Process each OCR result
        for i, (bbox, text, confidence) in enumerate(results):
            logger.info(f"Processing text block {i+1}/{len(results)}: '{text}' (confidence: {confidence:.6f})")
            
            if confidence >= confidence_threshold and text.strip():
                text_blocks.append({
                    'text': text,
                    'confidence': confidence,
                    'bbox': bbox
                })
                extracted_text += text + " "
                total_confidence += confidence
                logger.info(f"    ✅ ACCEPTED: '{text}'")
            else:
                logger.info(f"    ❌ REJECTED: '{text}' (confidence too low or empty)")
        
        # Fallback if no text accepted
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
                    logger.info(f"    ✅ FALLBACK ACCEPTED: '{text}'")
        
        # Calculate average confidence
        avg_confidence = total_confidence / len(text_blocks) if text_blocks else 0
        
        # Clean and prepare final text
        extracted_text = extracted_text.strip()
        
        # DEBUG: Print final extracted text details
        logger.info("=" * 50)
        logger.info("DEBUG: Final Extracted Text:")
        logger.info(f"  Text: '{extracted_text}'")
        logger.info(f"  Text type: {type(extracted_text)}")
        logger.info(f"  Text length: {len(extracted_text)}")
        logger.info(f"  Text repr: {repr(extracted_text)}")
        logger.info(f"  Text is empty: {not extracted_text}")
        logger.info(f"  Text is whitespace only: {extracted_text.isspace() if extracted_text else 'N/A'}")
        logger.info("=" * 50)
        
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
        
        # Final result preparation
        if text_blocks:
            logger.info(f"✅ OCR completed successfully:")
            logger.info(f"   📝 {len(text_blocks)} text blocks extracted")
            logger.info(f"   📊 Average confidence: {avg_confidence:.6f}")
            logger.info(f"   📄 Total text length: {len(extracted_text)} characters")
            logger.info(f"   🇨🇳 Chinese text dominant: {is_chinese_dominant}")
            logger.info(f"   🎯 Final extracted text: '{extracted_text}'")
            
            # Ensure we have actual content
            if not extracted_text or extracted_text.isspace():
                logger.warning("⚠️ Extracted text is empty or whitespace only")
                extracted_text = f"Image processed but no readable text detected. File: {document.title}"
        else:
            logger.warning("⚠️ No text detected in image")
            extracted_text = f"Image processed but no readable text detected. File: {document.title}"
        
        # Final debug before return
        logger.info(f"🔚 FINAL RETURN VALUES:")
        logger.info(f"   Success: True")
        logger.info(f"   Extracted text: '{extracted_text}'")
        logger.info(f"   Metadata: {metadata}")
        
        return {
            'success': True,
            'extracted_text': extracted_text,
            'metadata': metadata
        }
        
    except Exception as e:
        logger.error(f"❌ Image processing error: {str(e)}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        return {'success': False, 'error': str(e)}