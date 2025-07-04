# semantic_qa/translation_service.py
import logging
from typing import Dict, Optional
import openai
from .models import Translation

logger = logging.getLogger('semantic_qa')

class TranslationService:
    """Enhanced translation service using ChatGLM"""
    
    def __init__(self):
        self.llm = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize ChatGLM for translation"""
        try:
            self.chatglm_client = openai.OpenAI(
                api_key="a74b8073a98d4da4a066fc72095f58b0.gulObfhh7fnNcAmp",
                base_url="https://open.bigmodel.cn/api/paas/v4/"
            )
            logger.info("Successfully initialized ChatGLM for translation")
        except Exception as e:
            logger.error(f"Failed to initialize translation model: {str(e)}")
    
    def translate_text(self, text: str, target_language: str, source_language: str = 'en') -> str:
        """Translate text using ChatGLM"""
        if not self.chatglm_client or source_language == target_language:
            return text
        
        try:
            # Check cache first
            cached_translation = self._get_cached_translation(text, source_language, target_language)
            if cached_translation:
                return cached_translation
            
            # Create translation prompt
            language_names = {
                'en': 'English',
                'zh': 'Chinese',
                'es': 'Spanish',
                'fr': 'French',
                'de': 'German',
                'ja': 'Japanese'
            }
            
            source_lang_name = language_names.get(source_language, source_language)
            target_lang_name = language_names.get(target_language, target_language)
            
            prompt = f"""Translate the following {source_lang_name} text to {target_lang_name}. 
Only provide the translation, no explanations or additional text:

{text}"""
            
            response = self.chatglm_client.chat.completions.create(
                model="glm-4-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1024
            )
            
            translated_text = response.choices[0].message.content
            
            # Cache the translation
            self._cache_translation(text, translated_text, source_language, target_language)
            
            return translated_text.strip()
            
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return text  # Return original text if translation fails
    
    def translate_qa_result(self, search_result: Dict, target_language: str) -> Dict:
        """Translate search results to target language"""
        if target_language == 'en' or not self.chatglm_client:
            return search_result
        
        try:
            translated_results = []
            
            for result in search_result.get('results', []):
                if result.get('type') == 'qa_entry':
                    entry = result['entry']
                    
                    # Translate question and answer
                    translated_question = self.translate_text(entry.question, target_language)
                    translated_answer = self.translate_text(entry.answer, target_language)
                    
                    # Create a copy of the result with translated content
                    translated_result = result.copy()
                    translated_result['entry'] = type('TranslatedEntry', (), {
                        'id': entry.id,
                        'sku': entry.sku,  # Don't translate SKU
                        'question': translated_question,
                        'answer': translated_answer,
                        'image_link': entry.image_link,
                        'category': entry.category,
                        'keywords': entry.keywords
                    })()
                    
                    translated_results.append(translated_result)
                else:
                    # For document chunks, translate the text content
                    translated_result = result.copy()
                    if 'text_content' in result:
                        translated_result['text_content'] = self.translate_text(
                            result['text_content'], target_language
                        )
                    translated_results.append(translated_result)
            
            # Translate generated answer if present
            if search_result.get('generated_answer'):
                search_result['generated_answer'] = self.translate_text(
                    search_result['generated_answer'], target_language
                )
            
            # Update the search result
            search_result = search_result.copy()
            search_result['results'] = translated_results
            
            return search_result
            
        except Exception as e:
            logger.error(f"Result translation error: {str(e)}")
            return search_result
    
    def _get_cached_translation(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """Get cached translation if available"""
        try:
            import hashlib
            text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
            translation = Translation.objects.get(
                source_text_hash=text_hash,
                source_language=source_lang,
                target_language=target_lang
            )
            return translation.translated_text
        except Translation.DoesNotExist:
            return None
    
    def _cache_translation(self, text: str, translated_text: str, source_lang: str, target_lang: str):
        """Cache translation for future use"""
        try:
            import hashlib
            text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
            Translation.objects.get_or_create(
                source_text_hash=text_hash,
                source_language=source_lang,
                target_language=target_lang,
                defaults={
                    'source_text': text,
                    'translated_text': translated_text,
                    'translation_service': 'chatglm'
                }
            )
        except Exception as e:
            logger.error(f"Failed to cache translation: {str(e)}")