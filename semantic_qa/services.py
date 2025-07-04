# semantic_qa/services.py
import logging
import time
import numpy as np
import re
import os
from typing import List, Dict, Optional, Tuple
from django.db.models import Q
from django.conf import settings
import requests
from urllib.parse import urlparse
import hashlib

# Fix HuggingFace tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_community.vectorstores import FAISS
from langchain.schema import Document as LangchainDocument
from langchain_core.prompts import ChatPromptTemplate
import openai

from .models import QAEntry, TextChunk, Document, SemanticQuery, QueryMatch, SystemConfig
from .utils import SearchQueryProcessor, get_client_ip

logger = logging.getLogger('semantic_qa')

class RAGService:
    """Enhanced RAG service with document search and generation - SEMANTIC SEARCH ONLY"""
    
    def __init__(self):
        self.query_processor = SearchQueryProcessor()
        self.chatglm_client = None
        self.embeddings = None
        self.vector_store_qa = None
        self.vector_store_docs = None
        self.llm = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ChatGLM and embeddings"""
        try:
            # Initialize ChatGLM client
            chatglm_api_key = SystemConfig.get_config('chatglm_api_key', 'a74b8073a98d4da4a066fc72095f58b0.gulObfhh7fnNcAmp')
            chatglm_base_url = SystemConfig.get_config('chatglm_base_url', 'https://open.bigmodel.cn/api/paas/v4/')
            
            self.chatglm_client = openai.OpenAI(
                api_key=chatglm_api_key,
                base_url=chatglm_base_url
            )
            
            # Initialize embeddings
            from langchain_community.embeddings import HuggingFaceEmbeddings
            embedding_model = SystemConfig.get_config('embedding_model', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            self.llm = self.chatglm_client  # Use for text generation
            
            # Initialize vector stores
            self._initialize_vector_stores()
            
            logger.info("✅ RAG service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize RAG service: {str(e)}")
            self.embeddings = None
            self.llm = None
    
    def _initialize_vector_stores(self):
        """Initialize FAISS vector stores for QA entries and documents"""
        try:
            # Initialize QA vector store
            qa_entries = QAEntry.objects.all()
            if qa_entries.exists():
                qa_documents = []
                for entry in qa_entries:
                    doc_text = f"Product: {entry.sku} | Question: {entry.question} | Answer: {entry.answer}"
                    if entry.category:
                        doc_text += f" | Category: {entry.category}"
                    
                    qa_documents.append(LangchainDocument(
                        page_content=doc_text,
                        metadata={
                            'id': entry.id,
                            'sku': entry.sku,
                            'category': entry.category or '',
                            'type': 'qa_entry'
                        }
                    ))
                
                if qa_documents:
                    # self.vector_store_qa = FAISS.from_documents(qa_documents, self.embeddings)
                    vector_store_path = os.path.join(settings.BASE_DIR, 'vector_stores')
                    if os.path.exists(vector_store_path):
                        self.vector_store = FAISS.load_local(vector_store_path, self.embeddings)
                    logger.info(f"✅ QA vector store initialized with {len(qa_documents)} documents")
            
            # Initialize document vector store
            text_chunks = TextChunk.objects.filter(document__processing_status='completed')
            if text_chunks.exists():
                doc_documents = []
                for chunk in text_chunks:
                    doc_documents.append(LangchainDocument(
                        page_content=chunk.content,
                        metadata={
                            'id': chunk.id,
                            'document_id': chunk.document.id,
                            'document_title': chunk.document.title,
                            'document_type': chunk.document.document_type,
                            'page_number': chunk.page_number,
                            'type': 'document_chunk'
                        }
                    ))
                
                if doc_documents:
                    # self.vector_store_docs = FAISS.from_documents(doc_documents, self.embeddings)
                    vector_store_path = os.path.join(settings.BASE_DIR, 'vector_stores')
                    if os.path.exists(vector_store_path):
                        self.vector_store = FAISS.load_local(vector_store_path, self.embeddings)
                    logger.info(f"✅ Document vector store initialized with {len(doc_documents)} chunks")
                    
        except Exception as e:
            logger.error(f"❌ Failed to initialize vector stores: {str(e)}")
    def _rebuild_vector_store(self):
        """Rebuild and save vector store"""
        # Build vector store logic here
        vector_store_path = os.path.join(settings.BASE_DIR, 'vector_stores')
        self.vector_store.save_local(vector_store_path)
    def enhanced_search(self, query: str, language: str = 'en', 
                       search_qa_entries: bool = True, search_documents: bool = True,
                       document_types: List[str] = None, user_ip: str = None, 
                       user_agent: str = None, use_rag: bool = False) -> Dict:
        """
        Enhanced search using ONLY semantic search - NO exact SKU matching
        """
        start_time = time.time()
        
        try:
            logger.info(f"🔍 Enhanced search for: '{query}'")
            
            # Basic validation
            if not query or len(query.strip()) < 1:
                return {
                    'query': query,
                    'processed_query': query,
                    'query_type': 'empty',
                    'results': [],
                    'total_results': 0,
                    'response_time': time.time() - start_time,
                    'error': 'Empty query'
                }
            
            # Check if we have any data to search
            if search_qa_entries and not QAEntry.objects.exists():
                if search_documents and not Document.objects.filter(processing_status='completed').exists():
                    return {
                        'query': query,
                        'processed_query': query,
                        'query_type': 'no_data',
                        'results': [],
                        'total_results': 0,
                        'response_time': time.time() - start_time,
                        'error': 'No QA entries or documents in database'
                    }
            
            # Clean and process query - NO SKU EXTRACTION
            cleaned_query = self.query_processor.clean_query(query)
            semantic_terms = self.query_processor.extract_semantic_terms(cleaned_query)
            
            logger.info(f"🧹 Cleaned query: '{cleaned_query}'")
            logger.info(f"🔤 Semantic terms: {semantic_terms}")
            logger.info(f"🚫 SKU extraction DISABLED - treating all queries as semantic")
            
            results = []
            query_type = 'semantic'
            
            # ONLY SEMANTIC SEARCH - NO SKU PATTERN MATCHING AT ALL
            if self.embeddings and self.llm:
                logger.info("🧠 Performing semantic search with ChatGLM and Hugging Face embeddings...")
                try:
                    semantic_matches = self._enhanced_semantic_search(cleaned_query)
                    if semantic_matches:
                        logger.info(f"✅ Found {len(semantic_matches)} semantic matches")
                        results.extend(semantic_matches)
                        query_type = 'semantic'
                    else:
                        logger.info("❌ No semantic matches found")
                except Exception as e:
                    logger.error(f"❌ Semantic search failed: {str(e)}")
                    # Fall through to keyword search
            else:
                logger.warning("⚠️ Semantic search not available (models not initialized)")
            
            # Strategy 2: Keyword-based search (always include as fallback)
            logger.info(f"🔍 Adding keyword search results with terms: {semantic_terms}")
            keyword_matches = self._keyword_search(semantic_terms)
            if keyword_matches:
                logger.info(f"✅ Found {len(keyword_matches)} keyword matches")
                if not results:
                    results.extend(keyword_matches)
                    query_type = 'keyword'
                else:
                    # Merge keyword results
                    results = self._merge_results(results, keyword_matches)
            
            # Remove duplicates and sort by relevance
            results = self._deduplicate_results(results)
            results = sorted(results, key=lambda x: x['score'], reverse=True)
            
            logger.info(f"🧹 After deduplication and sorting: {len(results)} results")
            
            # Set a generous limit but keep all high-quality results
            max_results = 50  # Increased limit to show more results
            if len(results) > max_results:
                results = results[:max_results]
                logger.info(f"📊 Limited to top {max_results} results")
            
            # Generate RAG answer if requested
            generated_answer = ''
            rag_context = ''
            if use_rag and results:
                logger.info("🤖 Generating RAG answer...")
                rag_result = self._generate_rag_answer(query, results)
                generated_answer = rag_result.get('answer', '')
                rag_context = rag_result.get('context', '')
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Log the query
            self._log_query(query, cleaned_query, query_type, user_ip, user_agent, response_time, results)
            
            # Log final summary
            if results:
                logger.info(f"🎉 Search completed successfully: {len(results)} results in {response_time:.3f}s")
                for i, result in enumerate(results[:5]):  # Log first 5 results
                    entry_sku = getattr(result.get('entry'), 'sku', 'N/A')
                    logger.info(f"   Result {i+1}: {entry_sku} - {result['match_type']} - Score: {result['score']:.3f}")
            else:
                logger.warning(f"😔 Search returned no results for query: '{query}'")
                # Still try to provide some results with very low threshold
                if self.embeddings:
                    fallback_results = self._fallback_search(cleaned_query)
                    if fallback_results:
                        results = fallback_results
                        logger.info(f"💡 Fallback search found {len(results)} results")
            
            return {
                'success': True,
                'query': query,
                'processed_query': cleaned_query,
                'query_type': query_type,
                'results': results,
                'total_results': len(results),
                'response_time': response_time,
                'generated_answer': generated_answer,
                'rag_context': rag_context
            }
            
        except Exception as e:
            logger.error(f"❌ Search error: {str(e)}")
            return {
                'success': False,
                'query': query,
                'processed_query': query,
                'query_type': 'error',
                'results': [],
                'total_results': 0,
                'response_time': time.time() - start_time,
                'error': str(e),
                'generated_answer': '',
                'rag_context': ''
            }

    def _enhanced_semantic_search(self, query: str, min_threshold: float = 0.05) -> List[Dict]:
        """Enhanced semantic search with exact match boosting and proper ranking"""
        try:
            logger.info(f"🧠 Starting enhanced semantic search for: '{query}'")
            
            # Get all QA entries
            entries = QAEntry.objects.all()
            
            if not entries:
                logger.info("❌ No QA entries found for semantic search")
                return []
            
            logger.info(f"📊 Processing {len(entries)} entries for semantic search")
            
            # Create documents for vector search
            documents = []
            entry_mapping = {}
            
            for entry in entries:
                # Create comprehensive document text for better matching
                doc_parts = []
                
                # Add SKU with special formatting for exact matches
                doc_parts.append(f"Product: {entry.sku}")
                doc_parts.append(f"SKU: {entry.sku}")  # Add explicit SKU field
                
                # Add question
                doc_parts.append(f"Question: {entry.question}")
                
                # Add answer
                doc_parts.append(f"Answer: {entry.answer}")
                
                # Add category if available
                if entry.category:
                    doc_parts.append(f"Category: {entry.category}")
                
                # Add keywords if available
                if entry.keywords:
                    doc_parts.append(f"Keywords: {entry.keywords}")
                
                doc_text = " | ".join(doc_parts)
                documents.append(doc_text)
                entry_mapping[len(documents) - 1] = entry
            
            # Create vector store using FAISS
            if len(documents) > 0:
                try:
                    from langchain.schema import Document
                    doc_objects = [Document(page_content=doc) for doc in documents]
                    
                    logger.info(f"🔧 Creating FAISS vector store with {len(doc_objects)} documents")
                    vector_store = FAISS.from_documents(doc_objects, self.embeddings)
                    
                    # Perform similarity search with larger k to get more results
                    logger.info(f"🔍 Performing similarity search for: '{query}'")
                    search_results = vector_store.similarity_search_with_score(query, k=len(documents))
                    
                    logger.info(f"📊 Raw search results: {len(search_results)} matches")
                    
                    results = []
                    query_lower = query.lower().strip()
                    
                    for i, (doc, score) in enumerate(search_results):
                        logger.debug(f"   Result {i+1}: Score={score:.4f}, Doc='{doc.page_content[:100]}...'")
                        
                        # Find corresponding entry
                        try:
                            doc_index = documents.index(doc.page_content)
                            entry = entry_mapping[doc_index]
                            
                            # FIXED: Proper FAISS cosine distance to similarity conversion
                            # FAISS cosine similarity returns distance (0 = identical, 2 = opposite)
                            # Convert to similarity: similarity = 1 - (distance / 2)
                            # But clamp to ensure 0 <= similarity <= 1
                            base_similarity = max(0.0, min(1.0, 1.0 - (score / 2.0)))
                            
                            # CRITICAL: Add exact/partial match boosting
                            final_score = self._calculate_boosted_score(
                                base_similarity, entry, query_lower
                            )
                            
                            # Apply threshold
                            if final_score >= min_threshold:
                                results.append({
                                    'entry': entry,
                                    'score': final_score,
                                    'base_similarity': base_similarity,  # For debugging
                                    'match_type': self._determine_match_type(entry, query_lower),
                                    'match_reason': f'Enhanced similarity: {final_score:.3f}',
                                    'type': 'qa_entry',
                                    'text_content': f"Q: {entry.question}\nA: {entry.answer}",
                                    'source_info': {
                                        'sku': entry.sku,
                                        'category': entry.category or ''
                                    }
                                })
                                logger.debug(f"   ✅ Added result: {entry.sku} - Base: {base_similarity:.3f}, Final: {final_score:.3f}")
                            else:
                                logger.debug(f"   ❌ Below threshold: {final_score:.3f} < {min_threshold}")
                        
                        except ValueError as e:
                            logger.error(f"   ❌ Error mapping document: {str(e)}")
                            continue
                    
                    # ENHANCED SORTING: Multi-criteria ranking
                    results = self._enhanced_sort_results(results, query_lower)
                    
                    logger.info(f"🎉 Semantic search completed: {len(results)} results above threshold")
                    return results
                    
                except Exception as e:
                    logger.error(f"❌ FAISS vector store error: {str(e)}")
                    return []
            
            else:
                logger.warning("❌ No documents to search")
                return []
                
        except Exception as e:
            logger.error(f"❌ Enhanced semantic search error: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return []

    def _calculate_boosted_score(self, base_similarity: float, entry: QAEntry, query_lower: str) -> float:
        """Calculate boosted score with exact/partial match bonuses"""
        
        final_score = base_similarity
        sku_lower = entry.sku.lower()
        question_lower = entry.question.lower()
        
        # EXACT MATCH BONUSES (these should rank highest)
        if query_lower == sku_lower:
            final_score += 0.3  # Major boost for exact SKU match
            logger.debug(f"   🎯 Exact SKU match bonus: {entry.sku}")
        elif query_lower in sku_lower:
            final_score += 0.2  # Good boost for SKU contains query
            logger.debug(f"   🎯 SKU contains match bonus: {entry.sku}")
        elif sku_lower in query_lower:
            final_score += 0.15  # Medium boost for query contains SKU
            logger.debug(f"   🎯 Query contains SKU bonus: {entry.sku}")
        
        # QUESTION/CONTENT EXACT MATCHES
        if query_lower in question_lower:
            final_score += 0.1  # Boost for question containing query
            logger.debug(f"   📝 Question contains query bonus: {entry.sku}")
        
        # STARTING CHARACTER BONUS (f10 vs f30 - f10 should rank higher when searching "f")
        if sku_lower.startswith(query_lower):
            final_score += 0.25  # Strong boost for SKU starting with query
            logger.debug(f"   🚀 SKU starts with query bonus: {entry.sku}")
        
        # CAP THE FINAL SCORE TO REASONABLE RANGE
        final_score = min(final_score, 1.5)  # Allow boosted scores above 1.0
        
        return final_score

    def _determine_match_type(self, entry: QAEntry, query_lower: str) -> str:
        """Determine the type of match for better categorization"""
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

    def _enhanced_sort_results(self, results: List[Dict], query_lower: str) -> List[Dict]:
        """Enhanced multi-criteria sorting for better ranking"""
        
        def sort_key(result):
            entry = result['entry']
            score = result['score']
            sku_lower = entry.sku.lower()
            
            # PRIMARY: Exact SKU matches first
            exact_sku_match = (query_lower == sku_lower)
            
            # SECONDARY: SKU prefix matches (f10 when searching "f")
            sku_prefix_match = sku_lower.startswith(query_lower)
            
            # TERTIARY: SKU contains query
            sku_contains_query = (query_lower in sku_lower)
            
            # QUATERNARY: Similarity score
            similarity_score = score
            
            # QUINTENARY: SKU length (shorter SKUs preferred for similar matches)
            sku_length_penalty = len(entry.sku)
            
            # Return tuple for sorting (higher values sort first)
            # Note: We negate sku_length_penalty so shorter SKUs rank higher
            return (
                exact_sku_match,      # True > False
                sku_prefix_match,     # True > False  
                sku_contains_query,   # True > False
                similarity_score,     # Higher scores first
                -sku_length_penalty   # Shorter SKUs first
            )
        
        # Sort with multiple criteria
        sorted_results = sorted(results, key=sort_key, reverse=True)
        
        # Log the sorting for debugging
        logger.info("🔄 Results after enhanced sorting:")
        for i, result in enumerate(sorted_results[:10]):  # Log top 10
            entry = result['entry']
            logger.info(f"   {i+1}. {entry.sku} - Score: {result['score']:.3f} - Type: {result['match_type']}")
        
        return sorted_results
    
    def _search_vector_store(self, query: str, vector_store, result_type: str, min_threshold: float) -> List[Dict]:
        """Search a specific vector store and return formatted results"""
        try:
            # Query the vector store
            docs_and_scores = vector_store.similarity_search_with_score(
                query, 
                k=50  # Get up to 50 results
            )
            
            results = []
            logger.info(f"🔍 Vector search returned {len(docs_and_scores)} {result_type} results")
            
            for doc, score in docs_and_scores:
                try:
                    # Convert distance to similarity score (FAISS returns distance)
                    similarity_score = max(0.0, 1.0 - (score / 2.0))
                    
                    # Apply threshold
                    if similarity_score >= min_threshold:
                        if result_type == 'qa_entry':
                            # Find the corresponding QA entry
                            try:
                                entry = QAEntry.objects.get(id=doc.metadata['id'])
                                results.append({
                                    'entry': entry,
                                    'score': similarity_score,
                                    'match_type': 'semantic',
                                    'match_reason': f'Semantic similarity: {similarity_score:.3f}',
                                    'type': 'qa_entry',
                                    'text_content': f"Q: {entry.question}\nA: {entry.answer}",
                                    'source_info': {
                                        'sku': entry.sku,
                                        'category': entry.category or ''
                                    }
                                })
                                logger.debug(f"   ✅ Added QA result: {entry.sku} - Score: {similarity_score:.3f}")
                            except QAEntry.DoesNotExist:
                                logger.warning(f"   ⚠️ QA entry not found: {doc.metadata['id']}")
                                continue
                                
                        elif result_type == 'document_chunk':
                            # Find the corresponding text chunk
                            try:
                                chunk = TextChunk.objects.select_related('document').get(id=doc.metadata['id'])
                                results.append({
                                    'entry': chunk,  # For compatibility
                                    'score': similarity_score,
                                    'match_type': 'semantic',
                                    'match_reason': f'Document similarity: {similarity_score:.3f}',
                                    'type': 'document_chunk',
                                    'text_content': chunk.content,
                                    'source_info': {
                                        'document_id': chunk.document.id,
                                        'document_title': chunk.document.title,
                                        'document_type': chunk.document.document_type,
                                        'page_number': chunk.page_number
                                    }
                                })
                                logger.debug(f"   ✅ Added doc result: {chunk.document.title} - Score: {similarity_score:.3f}")
                            except TextChunk.DoesNotExist:
                                logger.warning(f"   ⚠️ Text chunk not found: {doc.metadata['id']}")
                                continue
                    else:
                        logger.debug(f"   ❌ Below threshold: {similarity_score:.3f} < {min_threshold}")
                
                except Exception as e:
                    logger.error(f"   ❌ Error processing result: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Vector store search error: {str(e)}")
            return []

    def _fallback_search(self, query: str) -> List[Dict]:
        """Fallback search with extremely low threshold to always return some results"""
        try:
            logger.info(f"💡 Performing fallback search with minimal threshold")
            return self._enhanced_semantic_search(query, min_threshold=0.01)  # Very low threshold
        except Exception as e:
            logger.error(f"❌ Fallback search failed: {str(e)}")
            return []

    def _keyword_search(self, terms: List[str]) -> List[Dict]:
        """Keyword-based search as fallback - searches all fields including SKU as regular terms"""
        if not terms:
            return []
        
        try:
            logger.info(f"🔍 Performing keyword search with terms: {terms}")
            
            results = []
            
            # Search QA entries - treat ALL terms as equal (no special SKU handling)
            qa_query = Q()
            for term in terms:
                qa_query |= (
                    Q(sku__icontains=term) |
                    Q(question__icontains=term) |
                    Q(answer__icontains=term) |
                    Q(category__icontains=term)
                )
            
            qa_entries = QAEntry.objects.filter(qa_query).distinct()
            
            for entry in qa_entries:
                score = self._calculate_keyword_score(entry, terms)
                if score > 0:
                    results.append({
                        'entry': entry,
                        'score': score,
                        'match_type': 'keyword',
                        'match_reason': f'Keyword match: {score:.3f}',
                        'type': 'qa_entry',
                        'text_content': f"Q: {entry.question}\nA: {entry.answer}",
                        'source_info': {
                            'sku': entry.sku,
                            'category': entry.category or ''
                        }
                    })
            
            logger.info(f"✅ Keyword search found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"❌ Keyword search error: {str(e)}")
            return []

    def _calculate_keyword_score(self, entry, terms: List[str]) -> float:
        """Calculate keyword matching score for an entry - all terms treated equally"""
        if not terms:
            return 0.0
        
        total_score = 0.0
        max_possible_score = 0.0
        
        # Define field weights - no special SKU preference since we treat everything semantically
        text_fields = {
            'sku': entry.sku or '',
            'question': entry.question or '',
            'answer': entry.answer or '',
            'category': entry.category or ''
        }
        
        field_weights = {
            'sku': 1.2,      # Slightly higher for exact matches, but not dominant
            'question': 1.5,  # Question matches are most important
            'answer': 1.0,    # Answer matches are important
            'category': 0.8   # Category matches are less important
        }
        
        for term in terms:
            term_lower = term.lower()
            term_score = 0.0
            
            for field_name, field_value in text_fields.items():
                if field_value:
                    field_lower = field_value.lower()
                    count = field_lower.count(term_lower)
                    if count > 0:
                        term_score += count * field_weights[field_name]
            
            total_score += term_score
            max_possible_score += max(field_weights.values())
        
        # Normalize score
        if max_possible_score > 0:
            normalized_score = min(total_score / max_possible_score, 1.0)
        else:
            normalized_score = 0.0
        
        return normalized_score

    def _merge_results(self, primary_results: List[Dict], secondary_results: List[Dict]) -> List[Dict]:
        """Merge two result sets, avoiding duplicates and maintaining scores"""
        # Create a map of existing entries
        existing_entries = {}
        for result in primary_results:
            if result['type'] == 'qa_entry':
                existing_entries[f"qa_{result['entry'].id}"] = result
            elif result['type'] == 'document_chunk':
                existing_entries[f"doc_{result['entry'].id}"] = result
        
        # Add secondary results that don't already exist, or update scores if better
        for secondary_result in secondary_results:
            if secondary_result['type'] == 'qa_entry':
                key = f"qa_{secondary_result['entry'].id}"
            elif secondary_result['type'] == 'document_chunk':
                key = f"doc_{secondary_result['entry'].id}"
            else:
                key = f"other_{secondary_result.get('entry', {}).get('id', 'unknown')}"
            
            if key not in existing_entries:
                # Add new result
                primary_results.append(secondary_result)
            else:
                # Update existing result if secondary has better score
                existing_result = existing_entries[key]
                if secondary_result['score'] > existing_result['score']:
                    existing_result['score'] = secondary_result['score']
                    existing_result['match_type'] = f"{existing_result['match_type']}+{secondary_result['match_type']}"
                    existing_result['match_reason'] = f"{existing_result['match_reason']} | {secondary_result['match_reason']}"
        
        return primary_results

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate entries from results"""
        seen_entries = set()
        deduplicated = []
        
        for result in results:
            if result['type'] == 'qa_entry':
                key = f"qa_{result['entry'].id}"
            elif result['type'] == 'document_chunk':
                key = f"doc_{result['entry'].id}"
            else:
                key = f"other_{result.get('entry', {}).get('id', 'unknown')}"
            
            if key not in seen_entries:
                seen_entries.add(key)
                deduplicated.append(result)
        
        return deduplicated

    def _log_query(self, original_query: str, processed_query: str, query_type: str, 
                   user_ip: str, user_agent: str, response_time: float, results: List[Dict]):
        """Log search query for analytics - with enhanced error handling"""
        try:
            # Create semantic query record
            semantic_query = SemanticQuery.objects.create(
                query_text=original_query,
                processed_query=processed_query,
                query_type=query_type,
                total_results=len(results),
                user_ip=user_ip,
                user_agent=user_agent or '',
                response_time=response_time
            )
            
            # Log query matches
            for i, result in enumerate(results[:20]):  # Log top 20 matches
                try:
                    if result['type'] == 'qa_entry':
                        # Handle both regular QAEntry objects and TranslatedEntry objects
                        entry = result['entry']
                        
                        # Get the original entry ID safely
                        if hasattr(entry, 'id'):
                            entry_id = entry.id
                        elif hasattr(entry, 'pk'):
                            entry_id = entry.pk
                        else:
                            # Skip this result if we can't get an ID
                            logger.warning(f"⚠️ Skipping result without ID: {type(entry)}")
                            continue
                        
                        # Get the original QAEntry object for logging
                        try:
                            original_entry = QAEntry.objects.get(id=entry_id)
                            QueryMatch.objects.create(
                                query=semantic_query,
                                qa_entry=original_entry,
                                relevance_score=result['score'],
                                match_reason=result['match_reason'],
                                rank_position=i + 1
                            )
                        except QAEntry.DoesNotExist:
                            logger.warning(f"⚠️ QAEntry {entry_id} not found for logging")
                            continue
                            
                    elif result['type'] == 'document_chunk':
                        # Handle document chunks
                        chunk = result['entry']  # This is actually a TextChunk
                        
                        # Get chunk ID safely
                        if hasattr(chunk, 'id'):
                            chunk_id = chunk.id
                        elif hasattr(chunk, 'pk'):
                            chunk_id = chunk.pk
                        else:
                            logger.warning(f"⚠️ Skipping chunk result without ID: {type(chunk)}")
                            continue
                        
                        try:
                            original_chunk = TextChunk.objects.get(id=chunk_id)
                            QueryMatch.objects.create(
                                query=semantic_query,
                                text_chunk=original_chunk,
                                relevance_score=result['score'],
                                match_reason=result['match_reason'],
                                rank_position=i + 1
                            )
                        except TextChunk.DoesNotExist:
                            logger.warning(f"⚠️ TextChunk {chunk_id} not found for logging")
                            continue
                            
                except Exception as match_error:
                    logger.error(f"❌ Error logging match {i}: {str(match_error)}")
                    continue
            
            logger.info(f"✅ Query logged successfully: {original_query[:50]}")
            
        except Exception as e:
            logger.error(f"❌ Error logging query: {str(e)}")
            # Don't re-raise the error, just log it so search continues to work

    def _generate_rag_answer(self, query: str, search_results: List[Dict]) -> Dict:
        """Generate RAG answer using ChatGLM"""
        try:
            logger.info(f"🤖 Generating RAG answer for: {query[:50]}...")
            
            if not self.chatglm_client or not search_results:
                return {'answer': '', 'context': ''}
            
            # Select best results for context
            context_items = []
            max_context_items = 8
            
            # Prioritize high-scoring results
            sorted_results = sorted(search_results, key=lambda x: x['score'], reverse=True)
            
            for result in sorted_results[:max_context_items]:
                if result['score'] >= 0.3:  # Only use high-quality results
                    source_info = result['source_info']
                    
                    if result['type'] == 'qa_entry':
                        context = f"[QA Entry - SKU: {source_info['sku']}]\n{result['text_content']}"
                    else:
                        context = f"[Document: {source_info['document_title']} - {source_info['document_type'].upper()}"
                        if source_info.get('page_number'):
                            context += f", Page {source_info['page_number']}"
                        context += f"]\n{result['text_content']}"
                    
                    context_items.append(context)
            
            if not context_items:
                return {'answer': '', 'context': ''}
            
            # Prepare context
            context_text = "\n\n---\n\n".join(context_items)
            
            # Create prompt
            prompt = self._create_rag_prompt(query, context_text)
            
            # Generate answer
            response = self.chatglm_client.chat.completions.create(
                model="glm-4-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1024
            )
            
            generated_answer = response.choices[0].message.content
            
            logger.info(f"✅ RAG answer generated for query: {query[:50]}...")
            
            return {
                'answer': generated_answer,
                'context': context_text
            }
            
        except Exception as e:
            logger.error(f"❌ RAG generation error: {str(e)}")
            return {'answer': '', 'context': ''}

    def _create_rag_prompt(self, query: str, context: str) -> str:
        """Create optimized prompt for RAG generation"""
        
        system_msg = """You are an intelligent assistant that helps users find information about products, troubleshooting, and technical questions. You have access to both structured Q&A data and document content."""
        
        prompt = f"""{system_msg}

Based on the following context information, please provide a comprehensive and accurate answer to the user's question.

Context Information:
{context}

User Question: {query}

Instructions:
1. Provide a direct and helpful answer based on the context provided
2. If the context contains multiple relevant pieces of information, synthesize them into a coherent response
3. If the context doesn't fully answer the question, clearly state what information is available and what might be missing
4. Use clear, concise language appropriate for the user's question
5. Include specific details like SKUs, model numbers, or technical specifications when relevant
6. If providing troubleshooting steps, organize them in a logical order

Answer:"""
        
        return prompt


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


class ImageProxyService:
    """Service for handling image proxy requests with enhanced error handling"""
    
    @staticmethod
    def fetch_image(image_url: str) -> Tuple[bytes, str, int]:
        """
        Fetch image with multiple retry strategies and return (content, content_type, status_code)
        """
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
            raise ValueError("Invalid URL format")
        
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
                    logger.info(f"✅ Successfully fetched image on attempt {attempt + 1}: {image_url}")
                    return response.content, content_type, 200
                else:
                    logger.warning(f"⚠️ HTTP {response.status_code} on attempt {attempt + 1} for: {image_url}")
                    if attempt == max_retries - 1:
                        return b'', 'text/plain', response.status_code
                    
            except requests.exceptions.SSLError as e:
                logger.warning(f"🔒 SSL error on attempt {attempt + 1} for {image_url}: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                continue
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"🌐 Request error on attempt {attempt + 1} for {image_url}: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                continue
        
        raise requests.exceptions.RequestException("All retry attempts failed")


class SafeTranslatedEntry:
    """Safe wrapper for translated entries that preserves all original attributes"""
    
    def __init__(self, original_entry, translated_question, translated_answer):
        # Copy all attributes from original entry
        for attr in dir(original_entry):
            if not attr.startswith('_') and not callable(getattr(original_entry, attr)):
                setattr(self, attr, getattr(original_entry, attr))
        
        # Override with translations
        self.question = translated_question
        self.answer = translated_answer
        
        # Ensure we have all required attributes for logging
        if not hasattr(self, 'id'):
            self.id = original_entry.id if hasattr(original_entry, 'id') else original_entry.pk
        if not hasattr(self, 'created_at'):
            self.created_at = getattr(original_entry, 'created_at', None)


class TranslationService:
    """Service for handling text translations"""
    
    def __init__(self):
        self.chatglm_client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChatGLM client for translation"""
        try:
            chatglm_api_key = SystemConfig.get_config('chatglm_api_key', 'a74b8073a98d4da4a066fc72095f58b0.gulObfhh7fnNcAmp')
            chatglm_base_url = SystemConfig.get_config('chatglm_base_url', 'https://open.bigmodel.cn/api/paas/v4/')
            
            self.chatglm_client = openai.OpenAI(
                api_key=chatglm_api_key,
                base_url=chatglm_base_url
            )
            
            logger.info("✅ Translation service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize translation service: {str(e)}")
    
    def translate_text(self, text: str, target_language: str, source_language: str = 'auto') -> str:
        """Translate text using ChatGLM"""
        try:
            if not self.chatglm_client or not text.strip():
                return text
            
            # Check if translation is needed
            if source_language == target_language:
                return text
            
            # Create translation prompt
            language_names = {
                'en': 'English',
                'zh': 'Chinese',
                'es': 'Spanish',
                'fr': 'French',
                'de': 'German',
                'ja': 'Japanese'
            }
            
            target_lang_name = language_names.get(target_language, target_language)
            
            if source_language == 'auto':
                prompt = f"Please translate the following text to {target_lang_name}. Only return the translation, nothing else:\n\n{text}"
            else:
                source_lang_name = language_names.get(source_language, source_language)
                prompt = f"Please translate the following text from {source_lang_name} to {target_lang_name}. Only return the translation, nothing else:\n\n{text}"
            
            # Generate translation
            response = self.chatglm_client.chat.completions.create(
                model="glm-4-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            
            translated_text = response.choices[0].message.content.strip()
            
            logger.info(f"✅ Translation completed: {source_language} -> {target_language}")
            return translated_text
            
        except Exception as e:
            logger.error(f"❌ Translation error: {str(e)}")
            return text  # Return original text on error
    
    def translate_search_results(self, search_results: List[Dict], target_language: str) -> List[Dict]:
        """Translate search results to target language using safe wrapper"""
        if target_language == 'en' or not self.chatglm_client:
            return search_results
        
        try:
            translated_results = []
            
            for result in search_results:
                if result['type'] == 'qa_entry':
                    entry = result['entry']
                    
                    # Translate question and answer
                    translated_question = self.translate_text(entry.question, target_language)
                    translated_answer = self.translate_text(entry.answer, target_language)
                    
                    # Create safe translated entry
                    translated_entry = SafeTranslatedEntry(
                        original_entry=entry,
                        translated_question=translated_question,
                        translated_answer=translated_answer
                    )
                    
                    # Update result with translated entry
                    translated_result = result.copy()
                    translated_result['entry'] = translated_entry
                    translated_result['text_content'] = f"Q: {translated_question}\nA: {translated_answer}"
                    
                    translated_results.append(translated_result)
                else:
                    # For non-QA entries, just add as-is
                    translated_results.append(result)
            
            return translated_results
            
        except Exception as e:
            logger.error(f"❌ Result translation error: {str(e)}")
            return search_results  # Return original on error
    
    def translate_qa_result(self, search_result: Dict, target_language: str) -> Dict:
        """Translate search results to target language - SAFE VERSION with debugging"""
        logger.info(f"🌐 Translation requested: {target_language}")
        logger.info(f"📊 Original results count: {len(search_result.get('results', []))}")
        
        if target_language == 'en' or not self.chatglm_client:
            logger.info("🚫 No translation needed - returning original results")
            return search_result
        
        try:
            if not search_result.get('success', False):
                logger.warning("⚠️ Search was not successful, skipping translation")
                return search_result
            
            original_results = search_result.get('results', [])
            if not original_results:
                logger.warning("⚠️ No results to translate")
                return search_result
            
            translated_results = []
            translation_errors = 0
            
            for i, result in enumerate(original_results):
                try:
                    if result.get('type') == 'qa_entry':
                        entry = result['entry']
                        logger.info(f"🔄 Translating QA entry {i+1}: {entry.sku}")
                        
                        # Translate question and answer
                        original_question = entry.question
                        original_answer = entry.answer
                        
                        translated_question = self.translate_text(original_question, target_language)
                        translated_answer = self.translate_text(original_answer, target_language)
                        
                        logger.info(f"✅ Translation completed for {entry.sku}")
                        logger.debug(f"   Q: {original_question[:50]}... -> {translated_question[:50]}...")
                        
                        # Create safe translated entry - FIXED VERSION
                        translated_entry = SafeTranslatedEntry(
                            original_entry=entry,
                            translated_question=translated_question,
                            translated_answer=translated_answer
                        )
                        
                        # Update result with translated entry
                        translated_result = result.copy()
                        translated_result['entry'] = translated_entry
                        translated_result['text_content'] = f"Q: {translated_question}\nA: {translated_answer}"
                        
                        translated_results.append(translated_result)
                        
                    else:
                        # For document chunks, translate the text content
                        logger.info(f"🔄 Translating document chunk {i+1}")
                        translated_result = result.copy()
                        if 'text_content' in result:
                            original_content = result['text_content']
                            translated_content = self.translate_text(original_content, target_language)
                            translated_result['text_content'] = translated_content
                            logger.info(f"✅ Document chunk translation completed")
                        translated_results.append(translated_result)
                        
                except Exception as result_error:
                    logger.error(f"❌ Error translating result {i+1}: {str(result_error)}")
                    translation_errors += 1
                    # Add original result if translation fails
                    translated_results.append(result)
                    continue
            
            # Translate generated answer if present
            translated_answer = search_result.get('generated_answer', '')
            if translated_answer:
                logger.info("🔄 Translating RAG generated answer")
                try:
                    translated_answer = self.translate_text(translated_answer, target_language)
                    logger.info("✅ RAG answer translation completed")
                except Exception as e:
                    logger.error(f"❌ Error translating RAG answer: {str(e)}")
                    # Keep original answer if translation fails
            
            # Update the search result
            updated_result = search_result.copy()
            updated_result['results'] = translated_results
            updated_result['generated_answer'] = translated_answer
            
            logger.info(f"🎉 Translation completed: {len(translated_results)} results, {translation_errors} errors")
            
            return updated_result
            
        except Exception as e:
            logger.error(f"❌ Critical translation error: {str(e)}")
            logger.error(f"❌ Returning original results to prevent data loss")
            return search_result  # Return original on critical error


class DocumentProcessingService:
    """Service for handling document processing and chunking"""
    
    def __init__(self):
        self.ocr_reader = None
        self._initialize_ocr()
    
    def _initialize_ocr(self):
        """Initialize OCR reader"""
        try:
            import easyocr
            self.ocr_reader = easyocr.Reader(['en', 'zh'])
            logger.info("✅ OCR service initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OCR service: {str(e)}")
    
    def process_document(self, document: Document) -> bool:
        """Process a document and create text chunks"""
        try:
            logger.info(f"📄 Processing document: {document.title}")
            
            if document.document_type == 'pdf':
                return self._process_pdf(document)
            elif document.document_type == 'image':
                return self._process_image(document)
            elif document.document_type == 'link':
                return self._process_link(document)
            else:
                logger.error(f"❌ Unsupported document type: {document.document_type}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Document processing error: {str(e)}")
            document.processing_status = 'failed'
            document.processing_log = str(e)
            document.save()
            return False
    
    def _process_pdf(self, document: Document) -> bool:
        """Process PDF document"""
        try:
            import PyPDF2
            from io import BytesIO
            
            # Read PDF file
            with document.original_file.open('rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                document.page_count = len(pdf_reader.pages)
                document.save()
                
                # Extract text from each page
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        text = page.extract_text()
                        if text.strip():
                            # Create chunks from page text
                            chunks = self._create_text_chunks(text, max_chunk_size=1000)
                            
                            for chunk_num, chunk_text in enumerate(chunks):
                                TextChunk.objects.create(
                                    document=document,
                                    content=chunk_text,
                                    page_number=page_num,
                                    chunk_index=chunk_num,
                                    chunk_type='text'
                                )
                    except Exception as e:
                        logger.warning(f"⚠️ Error processing page {page_num}: {str(e)}")
                        continue
            
            document.processing_status = 'completed'
            document.save()
            
            logger.info(f"✅ PDF processing completed: {document.title}")
            return True
            
        except Exception as e:
            logger.error(f"❌ PDF processing error: {str(e)}")
            return False
    
    def _process_image(self, document: Document) -> bool:
        """Process image document using OCR"""
        try:
            if not self.ocr_reader:
                logger.error("❌ OCR reader not available")
                return False
            
            # Read image file
            with document.original_file.open('rb') as file:
                image_data = file.read()
            
            # Perform OCR
            results = self.ocr_reader.readtext(image_data)
            
            # Extract text
            extracted_texts = []
            for (bbox, text, conf) in results:
                if conf > 0.5:  # Only include high-confidence results
                    extracted_texts.append(text)
            
            if extracted_texts:
                full_text = '\n'.join(extracted_texts)
                
                # Create text chunks
                chunks = self._create_text_chunks(full_text, max_chunk_size=800)
                
                for chunk_num, chunk_text in enumerate(chunks):
                    TextChunk.objects.create(
                        document=document,
                        content=chunk_text,
                        page_number=1,
                        chunk_index=chunk_num,
                        chunk_type='ocr_text'
                    )
                
                # Calculate average confidence
                avg_confidence = sum(conf for _, _, conf in results) / len(results) if results else 0
                document.ocr_confidence = avg_confidence
            
            document.processing_status = 'completed'
            document.save()
            
            logger.info(f"✅ Image processing completed: {document.title}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Image processing error: {str(e)}")
            return False
    
    def _process_link(self, document: Document) -> bool:
        """Process web link document"""
        try:
            import requests
            from bs4 import BeautifulSoup
            
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
            
            # Fetch web page
            response = requests.get(
                document.source_url, 
                headers=headers, 
                timeout=30,
                allow_redirects=True,
                verify=True
            )
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Extract text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks_raw = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks_raw if chunk)
            
            if text.strip():
                # Create text chunks
                chunks = self._create_text_chunks(text, max_chunk_size=1200)
                
                for chunk_num, chunk_text in enumerate(chunks):
                    TextChunk.objects.create(
                        document=document,
                        content=chunk_text,
                        page_number=1,
                        chunk_index=chunk_num,
                        chunk_type='web_text'
                    )
            
            document.processing_status = 'completed'
            document.save()
            
            logger.info(f"✅ Link processing completed: {document.title}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Link processing error: {str(e)}")
            return False
    
    def _create_text_chunks(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        """Create text chunks with overlap for better context"""
        if not text.strip():
            return []
        
        chunks = []
        overlap_size = max_chunk_size // 4  # 25% overlap
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) > max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    # Start new chunk with overlap
                    if len(current_chunk) > overlap_size:
                        current_chunk = current_chunk[-overlap_size:] + "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
                else:
                    # Single paragraph is too long, split it
                    if len(paragraph) > max_chunk_size:
                        words = paragraph.split()
                        temp_chunk = ""
                        
                        for word in words:
                            if len(temp_chunk) + len(word) + 1 > max_chunk_size:
                                if temp_chunk:
                                    chunks.append(temp_chunk.strip())
                                    # Add overlap
                                    temp_words = temp_chunk.split()
                                    if len(temp_words) > 10:
                                        temp_chunk = " ".join(temp_words[-10:]) + " " + word
                                    else:
                                        temp_chunk = word
                                else:
                                    temp_chunk = word
                            else:
                                temp_chunk += " " + word if temp_chunk else word
                        
                        if temp_chunk:
                            current_chunk = temp_chunk
                    else:
                        current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks


class AnalyticsService:
    """Service for handling analytics and reporting"""
    
    @staticmethod
    def get_search_analytics(days: int = 30) -> Dict:
        """Get search analytics for the specified number of days"""
        try:
            from datetime import datetime, timedelta
            from django.utils import timezone
            
            end_date = timezone.now()
            start_date = end_date - timedelta(days=days)
            
            queries = SemanticQuery.objects.filter(
                created_at__range=[start_date, end_date]
            )
            
            # Basic stats
            total_queries = queries.count()
            avg_response_time = queries.aggregate(
                avg_time=models.Avg('response_time')
            )['avg_time'] or 0
            
            # Query type distribution
            query_types = queries.values('query_type').annotate(
                count=models.Count('id'),
                avg_results=models.Avg('total_results')
            ).order_by('-count')
            
            # Daily query counts
            daily_queries = []
            for i in range(days):
                day = start_date + timedelta(days=i)
                day_queries = queries.filter(
                    created_at__date=day.date()
                ).count()
                daily_queries.append({
                    'date': day.strftime('%Y-%m-%d'),
                    'count': day_queries
                })
            
            # Top queries
            top_queries = queries.values('query_text').annotate(
                count=models.Count('id'),
                avg_results=models.Avg('total_results')
            ).order_by('-count')[:20]
            
            # Response time distribution
            response_time_ranges = [
                ('< 1s', queries.filter(response_time__lt=1.0).count()),
                ('1-2s', queries.filter(response_time__gte=1.0, response_time__lt=2.0).count()),
                ('2-5s', queries.filter(response_time__gte=2.0, response_time__lt=5.0).count()),
                ('> 5s', queries.filter(response_time__gte=5.0).count()),
            ]
            
            return {
                'total_queries': total_queries,
                'avg_response_time': round(avg_response_time, 3),
                'query_types': list(query_types),
                'daily_queries': daily_queries,
                'top_queries': list(top_queries),
                'response_time_distribution': response_time_ranges
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting search analytics: {str(e)}")
            return {}
    
    @staticmethod
    def get_content_analytics() -> Dict:
        """Get content analytics"""
        try:
            # QA entries stats
            total_qa_entries = QAEntry.objects.count()
            qa_by_category = QAEntry.objects.values('category').annotate(
                count=models.Count('id')
            ).order_by('-count')[:10]
            
            # Document stats
            total_documents = Document.objects.count()
            completed_documents = Document.objects.filter(processing_status='completed').count()
            failed_documents = Document.objects.filter(processing_status='failed').count()
            
            doc_by_type = Document.objects.values('document_type').annotate(
                count=models.Count('id')
            ).order_by('-count')
            
            # Text chunks stats
            total_chunks = TextChunk.objects.count()
            avg_chunk_length = TextChunk.objects.aggregate(
                avg_length=models.Avg(models.Length('content'))
            )['avg_length'] or 0
            
            return {
                'qa_entries': {
                    'total': total_qa_entries,
                    'by_category': list(qa_by_category)
                },
                'documents': {
                    'total': total_documents,
                    'completed': completed_documents,
                    'failed': failed_documents,
                    'by_type': list(doc_by_type)
                },
                'text_chunks': {
                    'total': total_chunks,
                    'avg_length': round(avg_chunk_length, 0)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting content analytics: {str(e)}")
            return {}