# semantic_qa/rag_service.py
import logging
import time
import traceback
import numpy as np
import os
import pickle
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from django.db.models import Q, Max
from django.conf import settings
from django.utils import timezone
from django.core.cache import cache

from langchain_community.vectorstores import FAISS
from langchain.schema import Document as LangchainDocument
from langchain_core.prompts import ChatPromptTemplate
import openai
import re

from .models import QAEntry, TextChunk, Document, SemanticQuery, QueryMatch, SystemConfig
from .utils import SearchQueryProcessor, get_client_ip

logger = logging.getLogger('semantic_qa')

class RAGService:
    """Optimized RAG service with caching and lazy loading"""
    
    # Class-level shared instances for performance
    _shared_vector_store = None
    _shared_embeddings = None
    _last_vector_build = None
    _last_qa_update = None
    _initialization_lock = False
    
    def __init__(self):
        """Lightweight initialization - heavy operations deferred until needed"""
        self.query_processor = SearchQueryProcessor()
        self.chatglm_client = None
        self.llm = None
        self.alpha = 0.9  # Keyword priority weight
        self.sku_alpha = 50  # SKU matching boost
        
        # Initialize basic models (lightweight)
        self._initialize_basic_models()
        
        logger.info("✅ RAG service instance created (lazy loading enabled)")
    
    def _initialize_basic_models(self):
        """Initialize only the basic models (ChatGLM client)"""
        try:
            # Initialize ChatGLM client
            chatglm_api_key = SystemConfig.get_config(
                'chatglm_api_key', 
                'a74b8073a98d4da4a066fc72095f58b0.gulObfhh7fnNcAmp'
            )
            chatglm_base_url = SystemConfig.get_config(
                'chatglm_base_url', 
                'https://open.bigmodel.cn/api/paas/v4/'
            )
            
            self.chatglm_client = openai.OpenAI(
                api_key=chatglm_api_key,
                base_url=chatglm_base_url
            )
            self.llm = self.chatglm_client
            
            logger.debug("✅ Basic models initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize basic models: {str(e)}")
    
#     def _ensure_embeddings_initialized(self):
#         """Initialize embeddings model (shared across instances)"""
#         if RAGService._shared_embeddings is None:
#             try:
#                 from langchain_community.embeddings import HuggingFaceEmbeddings
                
#                 embedding_model = SystemConfig.get_config(
#                     'embedding_model', 
#                     '/root/autodl-tmp/downloaded_models/models--sentence-transformers--all-MiniLM-L6-v2'
#                 )
                
#                 logger.info(f"🔧 Initializing embeddings model: {embedding_model}")
#                 start_time = time.time()
                
#                 RAGService._shared_embeddings = HuggingFaceEmbeddings(
#                     model_name=embedding_model,
#                     model_kwargs={'device': 'cuda'},
#                     encode_kwargs={'normalize_embeddings': True}
#                 )
                
#                 init_time = time.time() - start_time
#                 logger.info(f"✅ Embeddings initialized in {init_time:.2f}s")
                
#             except Exception as e:
#                 logger.error(f"❌ Failed to initialize embeddings: {str(e)}")
# #                 raise
    def _ensure_embeddings_initialized(self):
        """Initialize embeddings model (shared across instances)"""
        if RAGService._shared_embeddings is None:
            try:
                # Try ModelScope model first
                try:
                    from modelscope import AutoModel, AutoTokenizer
                    from langchain_core.embeddings import Embeddings
                    import torch
                    import numpy as np

                    logger.info("🔧 Attempting to load ModelScope model...")

                    class ModelScopeEmbeddings(Embeddings):
                        """Custom embeddings class for ModelScope models"""

                        def __init__(self, model_path: str, device: str = 'cuda'):
                            self.model_path = model_path
                            self.device = device
                            self.model = None
                            self.tokenizer = None
                            self._load_model()

                        def _load_model(self):
                            """Load ModelScope model and tokenizer"""
                            try:
                                logger.info(f"Loading ModelScope model from: {self.model_path}")
                                self.model = AutoModel.from_pretrained(
                                    self.model_path,
                                    trust_remote_code=True
                                ).to(self.device)
                                self.tokenizer = AutoTokenizer.from_pretrained(
                                    self.model_path,
                                    trust_remote_code=True
                                )
                                self.model.eval()
                                logger.info("✅ ModelScope model loaded successfully")
                            except Exception as e:
                                logger.error(f"❌ Failed to load ModelScope model: {e}")
                                raise

                        def embed_documents(self, texts: List[str]) -> List[List[float]]:
                            """Embed a list of documents"""
                            embeddings = []
                            for text in texts:
                                embedding = self._embed_single(text)
                                embeddings.append(embedding.tolist())
                            return embeddings

                        def embed_query(self, text: str) -> List[float]:
                            """Embed a single query"""
                            embedding = self._embed_single(text)
                            return embedding.tolist()

                        def _embed_single(self, text: str) -> np.ndarray:
                            """Embed a single text"""
                            with torch.no_grad():
                                # Tokenize
                                inputs = self.tokenizer(
                                    text,
                                    return_tensors='pt',
                                    padding=True,
                                    truncation=True,
                                    max_length=512
                                ).to(self.device)

                                # Get embeddings
                                outputs = self.model(**inputs)

                                # Use pooler output or mean pooling
                                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                                    embedding = outputs.pooler_output
                                else:
                                    # Mean pooling
                                    embedding = outputs.last_hidden_state.mean(dim=1)

                                # Normalize
                                embedding = torch.nn.functional.normalize(embedding, dim=-1)
                                return embedding.cpu().numpy().squeeze()

                    # Try to load ModelScope model
                    modelscope_path = '/root/autodl-tmp/downloaded_models/nlp_corom_sentence-embedding_english-base'

                    if os.path.exists(modelscope_path):
                        logger.info(f"🔧 Loading ModelScope model: {modelscope_path}")
                        RAGService._shared_embeddings = ModelScopeEmbeddings(
                            model_path=modelscope_path,
                            device='cuda'
                        )
                        logger.info("✅ ModelScope embeddings initialized successfully")
                        return
                    else:
                        logger.warning(f"ModelScope model not found at: {modelscope_path}")
                        raise FileNotFoundError("ModelScope model not found")

                except Exception as modelscope_error:
                    logger.warning(f"ModelScope loading failed: {modelscope_error}")

                    # Fallback to sentence-transformers if available locally
                    try:
                        from sentence_transformers import SentenceTransformer
                        from langchain_community.embeddings import HuggingFaceEmbeddings

                        logger.info("🔄 Falling back to local sentence-transformers...")

                        # Try to find sentence-transformers model in cache
                        cache_paths = [
                            '/root/autodl-tmp/downloaded_models/models--sentence-transformers--all-MiniLM-L6-v2',
                            '/root/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2'
                        ]

                        for cache_path in cache_paths:
                            if os.path.exists(cache_path):
                                # Look for snapshots
                                import glob
                                snapshots = glob.glob(os.path.join(cache_path, 'snapshots', '*'))
                                if snapshots:
                                    model_path = snapshots[0]
                                    logger.info(f"🔧 Using cached model: {model_path}")
                                    RAGService._shared_embeddings = HuggingFaceEmbeddings(
                                        model_name=model_path,
                                        model_kwargs={'device': 'cuda'},
                                        encode_kwargs={'normalize_embeddings': True}
                                    )
                                    logger.info("✅ Cached embeddings loaded successfully")
                                    return

                        # If no cached model found, create a simple fallback
                        raise Exception("No suitable local model found")

                    except Exception as fallback_error:
                        logger.error(f"❌ All embedding methods failed: {fallback_error}")

                        # Create a dummy embeddings class for testing
                        class DummyEmbeddings(Embeddings):
                            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                                return [[0.1] * 384 for _ in texts]  # 384-dim dummy vectors

                            def embed_query(self, text: str) -> List[float]:
                                return [0.1] * 384

                        logger.warning("⚠️ Using dummy embeddings - search functionality will be limited")
                        RAGService._shared_embeddings = DummyEmbeddings()

            except Exception as e:
                logger.error(f"❌ Failed to initialize embeddings: {str(e)}")
                # Don't raise to prevent complete system failure
    
    def _get_embeddings(self):
        """Get embeddings instance (lazy loading)"""
        if RAGService._shared_embeddings is None:
            self._ensure_embeddings_initialized()
        return RAGService._shared_embeddings
    
    def _needs_vector_store_rebuild(self) -> bool:
        """Check if vector store needs rebuilding"""
        try:
            # Check if QA entries have been updated
            latest_qa_update = QAEntry.objects.aggregate(
                latest=Max('updated_at')
            )['latest']
            
            if latest_qa_update is None:
                return False  # No QA entries
            
            # Compare with last build time
            if (RAGService._last_qa_update is None or 
                latest_qa_update > RAGService._last_qa_update):
                logger.info("🔄 QA entries updated, vector store rebuild needed")
                return True
            
            # Check if vector store exists
            if RAGService._shared_vector_store is None:
                logger.info("🔄 Vector store not loaded, rebuild needed")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error checking rebuild status: {str(e)}")
            return True  # Rebuild on error to be safe
    
    def _get_vector_store_path(self) -> str:
        """Get path for persistent vector store"""
        vector_dir = os.path.join(settings.BASE_DIR, 'vector_stores')
        os.makedirs(vector_dir, exist_ok=True)
        return os.path.join(vector_dir, 'qa_entries_faiss')
    
    def _load_persistent_vector_store(self) -> bool:
        """Try to load vector store from disk"""
        try:
            vector_path = self._get_vector_store_path()
            
            if not os.path.exists(vector_path):
                logger.info("📁 No persistent vector store found")
                return False
            
            logger.info("📂 Loading persistent vector store...")
            start_time = time.time()
            
            embeddings = self._get_embeddings()
            RAGService._shared_vector_store = FAISS.load_local(
                vector_path, 
                embeddings,
                allow_dangerous_deserialization=True
            )
            
            load_time = time.time() - start_time
            logger.info(f"✅ Vector store loaded in {load_time:.2f}s")
            
            # Load metadata
            metadata_path = f"{vector_path}_metadata.pkl"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                    RAGService._last_qa_update = metadata.get('last_qa_update')
                    RAGService._last_vector_build = metadata.get('build_time')
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load persistent vector store: {str(e)}")
            return False
    
    def _save_persistent_vector_store(self):
        """Save vector store to disk"""
        try:
            vector_path = self._get_vector_store_path()
            
            logger.info("💾 Saving vector store to disk...")
            start_time = time.time()
            
            RAGService._shared_vector_store.save_local(vector_path)
            
            # Save metadata
            metadata = {
                'last_qa_update': RAGService._last_qa_update,
                'build_time': RAGService._last_vector_build,
                'entry_count': QAEntry.objects.count()
            }
            
            metadata_path = f"{vector_path}_metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            save_time = time.time() - start_time
            logger.info(f"✅ Vector store saved in {save_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Failed to save vector store: {str(e)}")
    
    def _build_vector_store(self):
        """Build vector store from QA entries (optimized)"""
        try:
            logger.info("🔧 Building vector store from QA entries...")
            start_time = time.time()
            
            # Optimized query - only load needed fields
            qa_entries = QAEntry.objects.only(
                'id', 'sku', 'question', 'answer', 'category', 'keywords'
            ).order_by('id')
            
            entry_count = qa_entries.count()
            if entry_count == 0:
                logger.warning("⚠️ No QA entries found")
                return None
            
            logger.info(f"📊 Processing {entry_count} QA entries...")
            
            # Build documents in batches for memory efficiency
            documents = []
            entry_mapping = {}
            batch_size = 1000
            
            for i in range(0, entry_count, batch_size):
                batch = qa_entries[i:i + batch_size]
                batch_docs, batch_mapping = self._create_documents_batch(batch, i)
                documents.extend(batch_docs)
                entry_mapping.update(batch_mapping)
                
                if i % (batch_size * 5) == 0:  # Log every 5 batches
                    logger.info(f"   Processed {min(i + batch_size, entry_count)}/{entry_count} entries")
            
            # Create FAISS vector store
            if not documents:
                logger.warning("⚠️ No documents created")
                return None
            
            embeddings = self._get_embeddings()
            logger.info(f"🔧 Creating FAISS index with {len(documents)} documents...")
            
            vector_start = time.time()
            RAGService._shared_vector_store = FAISS.from_documents(documents, embeddings)
            vector_time = time.time() - vector_start
            
            # Update timestamps
            RAGService._last_vector_build = timezone.now()
            RAGService._last_qa_update = QAEntry.objects.aggregate(
                latest=Max('updated_at')
            )['latest']
            
            # Save to disk for future use
            self._save_persistent_vector_store()
            
            total_time = time.time() - start_time
            logger.info(f"✅ Vector store built successfully!")
            logger.info(f"   Total time: {total_time:.2f}s")
            logger.info(f"   Vector creation: {vector_time:.2f}s")
            logger.info(f"   Documents: {len(documents)}")
            
            return RAGService._shared_vector_store
            
        except Exception as e:
            logger.error(f"❌ Failed to build vector store: {str(e)}")
            traceback.print_exc()
            return None
    
    def _create_documents_batch(self, entries_batch, start_index: int) -> Tuple[List[LangchainDocument], Dict[int, QAEntry]]:
        """Create documents for a batch of QA entries"""
        documents = []
        mapping = {}
        
        for idx, entry in enumerate(entries_batch):
            doc_parts = []
            
            # Enhanced document content for better matching
            doc_parts.append(f"Product: {entry.sku}")
            doc_parts.append(f"SKU: {entry.sku}")
            doc_parts.append(f"Question: {entry.question}")
            doc_parts.append(f"Answer: {entry.answer}")
            
            if entry.category:
                doc_parts.append(f"Category: {entry.category}")
            
            if entry.keywords:
                doc_parts.append(f"Keywords: {entry.keywords}")
            
            doc_text = " | ".join(doc_parts)
            
            document = LangchainDocument(
                page_content=doc_text,
                metadata={
                    'id': entry.id,
                    'sku': entry.sku,
                    'category': entry.category or '',
                    'type': 'qa_entry',
                    'doc_index': start_index + idx
                }
            )
            
            documents.append(document)
            mapping[start_index + idx] = entry
        
        return documents, mapping
    def _get_vector_store(self):
        """Get vector store with lazy loading and caching"""
        # Prevent concurrent initialization
        if RAGService._initialization_lock:
            # Wait for initialization to complete
            max_wait = 30  # seconds
            wait_time = 0
            while RAGService._initialization_lock and wait_time < max_wait:
                time.sleep(0.1)
                wait_time += 0.1
            
            if RAGService._shared_vector_store is not None:
                return RAGService._shared_vector_store
        
        # Check if rebuild is needed
        if self._needs_vector_store_rebuild():
            RAGService._initialization_lock = True
            try:
                # Try to load from disk first
                if not self._load_persistent_vector_store():
                    # Build new vector store if loading failed
                    self._build_vector_store()
            finally:
                RAGService._initialization_lock = False
        
        return RAGService._shared_vector_store
    
    def enhanced_search(self, query: str, language: str = 'en', 
                       search_qa_entries: bool = True, search_documents: bool = True,
                       document_types: List[str] = None, user_ip: str = None, 
                       user_agent: str = None, use_rag: bool = False, 
                       max_results: int = 20) -> Dict:
        """
        Enhanced search with optimized performance
        """
        start_time = time.time()
        
        try:
            logger.info(f"🔍 Enhanced search for: '{query}'")
            
            # Input validation
            if not query or len(query.strip()) < 1:
                return self._empty_result(query, start_time, 'Empty query')
            
            # Clean and process query
            cleaned_query = self.query_processor.clean_query(query)
            semantic_terms = self.query_processor.extract_semantic_terms(cleaned_query)
            
            # Initialize search results
            search_results = []
            query_type = 'semantic'
            
            # 1. QA Entry Search (optimized)
            if search_qa_entries:
                qa_results = self._enhanced_qa_search(cleaned_query, semantic_terms)
                search_results.extend(qa_results)
            
            # 2. Document Search (if requested)
            if document_types and search_documents:
                doc_results = self._search_documents(cleaned_query, semantic_terms, document_types)
                search_results.extend(doc_results)
            
            # 3. RAG Generation (if enabled)
            generated_answer = ""
            rag_context = ""
            
            if use_rag and query_type == 'rag' and (search_results or self._has_relevant_chunks(cleaned_query)):
                rag_result = self._generate_rag_answer(cleaned_query, search_results)
                generated_answer = rag_result['answer']
                rag_context = rag_result['context']
                query_type = 'rag'
            
            # 4. Enhanced ranking and limiting
            search_results = self._enhanced_rank_and_limit_results(
                search_results, cleaned_query, max_results
            )
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Log the query (optimized)
            query_log = self._log_enhanced_query(
                query, cleaned_query, query_type, document_types, use_rag,
                len(search_results), rag_context, generated_answer,
                user_ip, user_agent, response_time
            )
            
            # Log matches (async if possible)
            self._log_query_matches(query_log, search_results)
            response_time = time.time() - start_time
            return {
                'query': query,
                'processed_query': cleaned_query,
                'query_type': query_type,
                'results': search_results,
                'total_results': len(search_results),
                'response_time': response_time,
                'generated_answer': generated_answer,
                'rag_context_used': bool(rag_context),
                'document_types_searched': document_types or [],
                'success': True
            }
            
        except Exception as e:
            logger.error(f"❌ Enhanced search error: {str(e)}")
            traceback.print_exc()
            return self._error_result(query, start_time, str(e)) 
        
    def _enhanced_qa_search(self, query: str, terms: List[str]) -> List[Dict]:

        
        def _is_continuous_substring(keyword_letters, text_letters):
            """检查关键词字母是否是文本字母的连续子串"""
            if not keyword_letters or not text_letters: return False
            return keyword_letters in text_letters

        def _calculate_letter_similarity(keyword_letters, text_letters):
            """计算字母相似度"""
            if not keyword_letters or not text_letters: return 0.0
            keyword_upper = keyword_letters.upper()
            text_upper = text_letters.upper()
            
            if keyword_upper in text_upper:
                return 1.0
            
            keyword_chars = set(keyword_upper)
            text_chars = set(text_upper)
            return len(keyword_chars & text_chars) / len(keyword_chars)

        def ngram_overlap(keyword, text, n=3):
            """
            An advanced, hierarchical matching function for SKUs and other text.
            It processes matches in distinct, prioritized levels.
            """
            keyword = keyword.lower()
            text = text.lower()

            def get_parts(s):
                return ''.join(re.findall(r'[a-zA-Z]', s)), ''.join(re.findall(r'\d', s))

            # --- Level 1: Exact and High-Confidence Prefix Match ---
            if keyword == text:
                return 1.0
            # Only apply prefix match for longer keywords to avoid 's' matching 'sharan'
            if len(keyword) > 2 and text.startswith(keyword):
                return 0.99

            kw_letters, kw_numbers = get_parts(keyword)
            is_sku_pattern = '-' in text and '*' in text

            # --- Level 1.5: Single-Letter SKU Prefix Match (TARGETED FIX) ---
            # Specifically for queries like "s VARA421" to match "*MS*".
            if len(keyword) == 1 and is_sku_pattern:
                try:
                    text_prefix_part = text.split('-', 1)[0]
                    text_prefix_letters = get_parts(text_prefix_part.replace('*',''))[0]
                    if keyword in text_prefix_letters:
                        # Give a very high score for matching a single letter in the SKU prefix.
                        # This ensures this match outranks vector similarity differences.
                        return 0.97
                except (ValueError, IndexError):
                    pass # Ignore malformed SKU strings

            # --- Level 2: Primary SKU Match (*XT*-JUKE104 Pattern) ---
            # This is the most critical logic for your main issue.
            if is_sku_pattern:
                try:
                    text_prefix_part, text_code_part = text.split('-', 1)
                    text_prefix_letters = get_parts(text_prefix_part.replace('*',''))[0]
                    text_code_letters, text_code_numbers = get_parts(text_code_part)

                    # Check for number prefix match (e.g., '10' in '104')
                    if kw_numbers and text_code_numbers.startswith(kw_numbers):
                        # Combine all letters for a comprehensive check
                        # For query 'xjuke10', kw_letters is 'xjuke'
                        # For sku '*XT*-JUKE104', combined_sku_letters is 'xtjuke'
                        combined_sku_letters = text_prefix_letters + text_code_letters

                        # Check if all letters from the query are present in the combined SKU letters
                        if all(char in combined_sku_letters for char in kw_letters):
                            # This is a very strong match. Now, add a bonus for prefix similarity.
                            # This bonus will differentiate *XT* from *DW*.
                            prefix_bonus = 0
                            if text_prefix_letters:
                                # How many of the keyword's letters are in the SKU's prefix?
                                matching_prefix_chars = sum(1 for char in kw_letters if char in text_prefix_letters)
                                prefix_bonus = matching_prefix_chars * 0.02

                            # Base score of 0.96, plus a bonus for matching the prefix letters
                            return min(0.98, 0.96 + prefix_bonus) # Cap score at 0.98
                except (ValueError, IndexError):
                    pass # Ignore malformed SKU strings

            # --- Level 3: Variant SKU Match (hs3 vs HS13 Pattern) ---
            # This preserves the functionality you were satisfied with.
            if kw_letters and kw_numbers:
                text_letters, text_numbers = get_parts(text)
                if text_letters and text_numbers:
                    # e.g., kw='hs3', text='hs13'
                    if text_letters.startswith(kw_letters) and kw_numbers in text_numbers:
                        return 0.95 # High score for this trusted pattern

            # --- Level 4: Fallback for General Containment ---
            # This now has a very low score to prevent it from outranking SKU matches.
            if len(keyword) >= 1 and keyword.lower() in text.lower():
                return 0.2

            return 0.0
        results = []
        starttime1 = time.time()
        try:
            # Get vector store (with caching)
            vector_store = self._get_vector_store()
            
            if vector_store is None:
                logger.warning("⚠️ Vector store not available")
                return results
            
            logger.info(f"🧠 Starting optimized QA search for: '{query}'")
            
            # Get all QA entries for enhanced matching
            qa_entries = QAEntry.objects.all()
            
            if not qa_entries.exists():
                return results
            
            logger.info(f"📊 Processing {len(qa_entries)} entries for enhanced search")
            
            # Optimized vector search
            search_start = time.time()
            search_results = vector_store.similarity_search_with_score(
                query, 
                k=min(100, vector_store.index.ntotal),  # Limit k to available docs
                fetch_k=min(200, vector_store.index.ntotal * 2)  # Fetch more for better results
            )
            search_time = time.time() - search_start
            
            logger.info(f"🔍 Vector search completed in {search_time:.3f}s, found {len(search_results)} results")
            
            query_lower = query.lower().strip()
            
            # 1. 关键词分词
            keywords = re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', query_lower)
            logger.info(f"🔍 Extracted keywords: {keywords}")
            
            # 2. 归一化向量分数
            if search_results:
                vec_scores = [score for _, score in search_results]
                max_vec = max(vec_scores)
                min_vec = min(vec_scores)
                logger.info(f"🔍 Vector score range: {min_vec:.3f} - {max_vec:.3f}")
                
                def norm_vec_score(score):
                    return 1 - (score - min_vec) / (max_vec - min_vec + 1e-8)
            else:
                def norm_vec_score(score):
                    return max(0.0, 1.0 - (score / 2.0))
            
            # 3. 计算综合分数并排序
            scored_entries = []
            debug_count = 0
            max_debug_entries = 10  # Limit debug output
            
            process_start = time.time()
            
            # Create mapping from search results for faster lookup
            search_result_map = {}
            for doc, distance in search_results:
                entry_id = doc.metadata.get('id')
                if entry_id:
                    search_result_map[entry_id] = norm_vec_score(distance)
            
            # Debug: Show some vector scores
            logger.info(f"🔍 Sample vector scores: {list(search_result_map.values())[:5]}")
            
            for entry in qa_entries:
                try:
                    # Get vector similarity score (0 if not in search results)
                    vec_score = search_result_map.get(entry.id, 0.0)
                    
                    # 计算关键词匹配分数
                    keyword_score = 0.0
                    total_weight = 0.0
                    
                    # Debug info for first few entries
                    debug_this_entry = debug_count < max_debug_entries
                    
                    if debug_this_entry:
                        logger.info(f"🔍 Debug entry {entry.sku}: vec_score={vec_score:.3f}")
                    
                    for keyword in keywords:
                        if not keyword:
                            continue
                            
                        # SKU匹配 (权重最高)
                        sku_match = ngram_overlap(keyword.lower(), entry.sku.lower())
                        
                        # 新增：字符级匹配（当关键词长度大于1时）
                        if len(keyword) > 1:
                            char_match = self._enhanced_char_match(keyword, entry.sku)
                            # 取两种匹配方式的最大值
                            sku_match = max(sku_match, char_match)
                        
                        keyword_score += sku_match * self.sku_alpha
                        total_weight += self.sku_alpha
                        
                        # 问题匹配
                        question_match = ngram_overlap(keyword, entry.question)
                        keyword_score += question_match * 10
                        total_weight += 10
                        
                        # 答案匹配
                        answer_match = ngram_overlap(keyword, entry.answer)
                        keyword_score += answer_match * 5
                        total_weight += 5
                        
                        # 类别匹配
                        if entry.category:
                            category_match = ngram_overlap(keyword, entry.category)
                            keyword_score += category_match * 3
                            total_weight += 3
                        
                        if debug_this_entry:
                            logger.info(f"   Keyword '{keyword}': sku={sku_match:.3f}, q={question_match:.3f}, a={answer_match:.3f}")
                    
                    # 归一化关键词分数
                    norm_keyword_score = keyword_score / total_weight if total_weight > 0 else 0.0
                    
                    # 综合分数 = α * 关键词分数 + (1-α) * 向量分数
                    final_score = self.alpha * norm_keyword_score + (1 - self.alpha) * vec_score
                    
                    if debug_this_entry:
                        logger.info(f"   Final: kw_raw={keyword_score:.1f}, kw_norm={norm_keyword_score:.3f}, final={final_score:.3f}")
                        debug_count += 1
                    
                    # Lower threshold for debugging - let's see what we get
                    min_threshold = 0.01  # Much lower threshold
                    if final_score >= min_threshold:
                        scored_entries.append((entry, final_score))
                        
                        if final_score > 0.1:  # Only log higher scoring entries
                            logger.debug(f"   ✅ Entry {entry.sku}: vec={vec_score:.3f}, kw={norm_keyword_score:.3f}, final={final_score:.3f}")
                    
                except Exception as e:
                    logger.error(f"Error processing entry {entry.id}: {str(e)}")
                    continue
            
            process_time = time.time() - process_start
            
            # Debug: Show threshold analysis
            if scored_entries:
                scores = [score for _, score in scored_entries]
                logger.info(f"📊 Score analysis: min={min(scores):.3f}, max={max(scores):.3f}, count={len(scores)}")
                
                # Count entries above different thresholds
                thresholds = [0.01, 0.05, 0.1, 0.2, 0.5]
                for thresh in thresholds:
                    count = sum(1 for score in scores if score >= thresh)
                    logger.info(f"   Entries >= {thresh}: {count}")
            else:
                logger.warning("⚠️ No entries scored above minimum threshold!")
            
            logger.info(f"📊 Enhanced scoring completed in {process_time:.3f}s")
            
            # 排序并取前k个结果
            scored_entries.sort(key=lambda x: x[1], reverse=True)
            top_results = scored_entries[:20]
            
            # Format results
            for entry, score in top_results:
                results.append({
                    'type': 'qa_entry',
                    'source': 'qa_database',
                    'entry': entry,
                    'score': score,
                    'base_similarity': score,
                    'match_type': self._determine_match_type(entry, query_lower),
                    'match_reason': f'Enhanced similarity: {score:.3f}',
                    'text_content': f"Q: {entry.question}\nA: {entry.answer}",
                    'source_info': {
                        'sku': entry.sku,
                        'category': entry.category or ''
                    }
                })
                logger.debug(f"   ✅ Added result: {entry.sku} - Final: {score:.3f}")
            
            total_time = time.time() - starttime1
            logger.info(f"🎉 Enhanced QA search completed: {len(results)} results in {total_time:.3f}s")
            
        except Exception as e:
            logger.error(f"❌ Enhanced QA search error: {str(e)}")
            traceback.print_exc()
        
        return results
    
    def _calculate_boosted_score(self, base_similarity: float, entry: QAEntry, query_lower: str) -> float:
        """Calculate boosted score with exact/partial match bonuses"""
        final_score = base_similarity
        sku_lower = entry.sku.lower()
        question_lower = entry.question.lower()
        
        # Exact match bonuses
        if query_lower == sku_lower:
            final_score += 0.3
        elif query_lower in sku_lower:
            final_score += 0.2
        elif sku_lower in query_lower:
            final_score += 0.15
        
        # Question match bonuses
        if query_lower in question_lower:
            final_score += 0.1
        
        # Prefix match bonus
        if sku_lower.startswith(query_lower):
            final_score += 0.25
        
        # Cap the score
        return min(final_score, 1.5)
    
    def _determine_match_type(self, entry: QAEntry, query_lower: str) -> str:
        """Determine match type for categorization"""
        sku_lower = entry.sku.lower()
        question_lower = entry.question.lower()
        
        if query_lower == sku_lower:
            return 'exact_sku'
        elif query_lower in sku_lower or sku_lower in query_lower:
            return 'partial_sku'
        elif sku_lower.startswith(query_lower):
            return 'sku_prefix'
        elif query_lower in question_lower:
            return 'question_match'
        else:
            return 'semantic'
    
    def _enhanced_rank_and_limit_results(self, results: List[Dict], query_lower: str, max_results: int) -> List[Dict]:
        """Enhanced ranking with multi-criteria sorting"""
        if not results:
            return results
        
        # Sort by multiple criteria
        def sort_key(result):
            score = result.get('score', 0)
            match_type = result.get('match_type', 'semantic')
            
            # Priority weights
            type_weights = {
                'exact_sku': 1000,
                'sku_prefix': 900,
                'partial_sku': 800,
                'question_match': 700,
                'semantic': 600
            }
            
            type_weight = type_weights.get(match_type, 500)
            return (type_weight + score * 100)
        
        sorted_results = sorted(results, key=sort_key, reverse=True)
        return sorted_results[:max_results]
    
    def _search_documents(self, query: str, terms: List[str], document_types: List[str]) -> List[Dict]:
        """Search in document chunks (placeholder - implement as needed)"""
        # Implementation depends on your document search requirements
        return []
    
    def _has_relevant_chunks(self, query: str) -> bool:
        """Check if there are relevant document chunks (placeholder)"""
        return False
    
    def _generate_rag_answer(self, query: str, context_results: List[Dict]) -> Dict:
        """Generate RAG answer using ChatGLM (placeholder - implement as needed)"""
        return {'answer': '', 'context': ''}
    
    def _log_enhanced_query(self, *args, **kwargs):
        """Log query (implement as needed)"""
        pass
    
    def _log_query_matches(self, query_log, results):
        """Log query matches (implement as needed)"""
        pass
    
    def _empty_result(self, query: str, start_time: float, error: str) -> Dict:
        """Return empty result structure"""
        return {
            'query': query,
            'processed_query': query,
            'query_type': 'empty',
            'results': [],
            'total_results': 0,
            'response_time': time.time() - start_time,
            'generated_answer': '',
            'rag_context_used': False,
            'document_types_searched': [],
            'success': False,
            'error': error
        }
    
    def _error_result(self, query: str, start_time: float, error: str) -> Dict:
        """Return error result structure"""
        return {
            'query': query,
            'processed_query': query,
            'query_type': 'error',
            'results': [],
            'total_results': 0,
            'response_time': time.time() - start_time,
            'generated_answer': '',
            'rag_context_used': False,
            'document_types_searched': [],
            'success': False,
            'error': error
        }
    
    def force_rebuild_vector_store(self):
        """Force rebuild of vector store (for admin use)"""
        logger.info("🔄 Force rebuilding vector store...")
        RAGService._shared_vector_store = None
        RAGService._last_qa_update = None
        self._build_vector_store()
        logger.info("✅ Vector store force rebuild completed")
    
    def get_vector_store_info(self) -> Dict:
        """Get information about current vector store"""
        info = {
            'exists': RAGService._shared_vector_store is not None,
            'last_build': RAGService._last_vector_build,
            'last_qa_update': RAGService._last_qa_update,
            'qa_count': QAEntry.objects.count()
        }
        
        if RAGService._shared_vector_store:
            info['vector_count'] = RAGService._shared_vector_store.index.ntotal
        
        return info

    def _enhanced_char_match(self, keyword: str, sku: str) -> float:
        """
        增强的字符匹配算法
        考虑字符顺序、连续性和位置信息
        """
        keyword = keyword.lower()
        sku = sku.lower()
        
        if not keyword or not sku:
            return 0.0
        
        # 1. 基础字符匹配
        keyword_chars = set(keyword)
        sku_chars = set(sku)
        char_overlap = len(keyword_chars & sku_chars) / len(keyword_chars)
        
        # 2. 连续字符匹配
        max_consecutive = 0
        for i in range(len(keyword)):
            for j in range(i + 1, len(keyword) + 1):
                substring = keyword[i:j]
                if substring in sku:
                    max_consecutive = max(max_consecutive, len(substring))
        consecutive_score = max_consecutive / len(keyword)
        
        # 3. 字符顺序匹配
        order_score = 0.0
        if len(keyword) > 1:
            # 检查关键词中的字符在SKU中是否保持相对顺序
            sku_char_positions = {}
            for i, char in enumerate(sku):
                if char in keyword:
                    if char not in sku_char_positions:
                        sku_char_positions[char] = []
                    sku_char_positions[char].append(i)
            
            # 计算顺序匹配分数
            order_matches = 0
            for i in range(len(keyword) - 1):
                char1, char2 = keyword[i], keyword[i + 1]
                if char1 in sku_char_positions and char2 in sku_char_positions:
                    # 检查char2是否在char1之后出现
                    for pos1 in sku_char_positions[char1]:
                        for pos2 in sku_char_positions[char2]:
                            if pos2 > pos1:
                                order_matches += 1
                                break
                        if order_matches > 0:
                            break
            
            order_score = order_matches / (len(keyword) - 1) if len(keyword) > 1 else 0.0
        
        # 4. 加权组合
        final_score = (
            char_overlap * 0.3 +      # 基础字符匹配
            consecutive_score * 0.4 + # 连续字符匹配
            order_score * 0.3         # 字符顺序匹配
        )
        
        return final_score