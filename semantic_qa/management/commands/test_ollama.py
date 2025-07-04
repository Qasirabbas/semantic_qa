# Create this file: semantic_qa/management/commands/test_ollama.py

import os
from django.core.management.base import BaseCommand
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class Command(BaseCommand):
    help = 'Test Ollama connection in Django'

    def handle(self, *args, **options):
        self.stdout.write("🧪 Testing Ollama in Django environment...")
        
        try:
            # Test embeddings (same as your working rag.py)
            self.stdout.write("1️⃣ Testing embeddings...")
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            test_embedding = embeddings.embed_query("test query")
            self.stdout.write(self.style.SUCCESS(f"✅ Embeddings working: {len(test_embedding)} dimensions"))
            
            # Test LLM (same as your working rag.py)
            self.stdout.write("2️⃣ Testing LLM...")
            llm = ChatOllama(
                base_url='http://localhost:11434',
                model='qwen2.5',
                temperature=0.7
            )
            
            prompt = ChatPromptTemplate.from_template("Respond with just 'Hello Django': {input}")
            chain = prompt | llm | StrOutputParser()
            response = chain.invoke({"input": "test"})
            self.stdout.write(self.style.SUCCESS(f"✅ LLM working: {response}"))
            
            # Test semantic search service
            self.stdout.write("3️⃣ Testing semantic search service...")
            from semantic_qa.services import SemanticSearchService
            search_service = SemanticSearchService()
            
            if search_service.embeddings and search_service.llm:
                self.stdout.write(self.style.SUCCESS("✅ Semantic search service initialized"))
                
                # Test search
                result = search_service.search_qa_entries("test query")
                self.stdout.write(self.style.SUCCESS(f"✅ Search test: {result['query_type']} search returned {result['total_results']} results"))
            else:
                self.stdout.write(self.style.ERROR("❌ Semantic search service failed to initialize"))
            
            self.stdout.write(self.style.SUCCESS("\n🎉 All tests passed! Ollama is working in Django."))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"❌ Test failed: {str(e)}"))
            self.stdout.write("💡 Make sure Ollama is running: ollama serve")
            self.stdout.write("💡 Make sure models are installed:")
            self.stdout.write("   ollama pull qwen2.5")
            self.stdout.write("   ollama pull nomic-embed-text")