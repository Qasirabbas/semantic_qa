# Create: semantic_qa/management/commands/debug_search.py

from django.core.management.base import BaseCommand
from semantic_qa.models import QAEntry, SemanticQuery
from semantic_qa.services import SemanticSearchService

class Command(BaseCommand):
    help = 'Debug search data and functionality'

    def handle(self, *args, **options):
        self.stdout.write("🔍 Debugging Search Data and Functionality...")
        
        # Check 1: Database entries
        self.stdout.write("\n1️⃣ Checking database entries...")
        total_entries = QAEntry.objects.count()
        self.stdout.write(f"Total QA entries in database: {total_entries}")
        
        if total_entries > 0:
            # Show sample entries
            self.stdout.write("\n📋 Sample entries:")
            for i, entry in enumerate(QAEntry.objects.all()[:5]):
                self.stdout.write(f"  {i+1}. SKU: {entry.sku}")
                self.stdout.write(f"     Question: {entry.question[:100]}...")
                self.stdout.write(f"     Answer: {entry.answer[:100]}...")
                self.stdout.write(f"     Keywords: {entry.keywords[:50]}...")
                self.stdout.write("")
        else:
            self.stdout.write(self.style.WARNING("❌ No QA entries found in database!"))
            self.stdout.write("💡 This is likely why search returns no results.")
            self.stdout.write("   Upload some Excel data first!")
            return
        
        # Check 2: Test different search strategies
        self.stdout.write("2️⃣ Testing search strategies...")
        
        # Get a sample SKU and question for testing
        sample_entry = QAEntry.objects.first()
        test_queries = [
            sample_entry.sku,  # Test SKU search
            sample_entry.question.split()[0],  # Test first word of question
            "wiring",  # Test your actual search
            sample_entry.question[:20],  # Test partial question
        ]
        
        search_service = SemanticSearchService()
        
        for query in test_queries:
            self.stdout.write(f"\n🔍 Testing query: '{query}'")
            try:
                result = search_service.search_qa_entries(query, "127.0.0.1", "debug")
                self.stdout.write(f"   Query type: {result['query_type']}")
                self.stdout.write(f"   Results found: {result['total_results']}")
                self.stdout.write(f"   Response time: {result['response_time']:.3f}s")
                
                if result['results']:
                    for i, res in enumerate(result['results'][:2]):
                        self.stdout.write(f"   Result {i+1}: {res['entry'].sku} - Score: {res['score']:.3f}")
                else:
                    self.stdout.write("   ❌ No results returned")
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"   ❌ Search failed: {str(e)}"))
        
        # Check 3: Test individual search methods
        self.stdout.write("\n3️⃣ Testing individual search methods...")
        
        # Test SKU search
        sample_sku = sample_entry.sku
        sku_results = search_service._search_by_sku([sample_sku], exact=True)
        self.stdout.write(f"Direct SKU search for '{sample_sku}': {len(sku_results)} results")
        
        # Test keyword search
        keywords = sample_entry.question.split()[:3]
        keyword_results = search_service._keyword_search(keywords)
        self.stdout.write(f"Keyword search for {keywords}: {len(keyword_results)} results")
        
        # Test semantic search (if available)
        if search_service.embeddings and search_service.llm:
            try:
                semantic_results = search_service._semantic_search(sample_entry.question[:50])
                self.stdout.write(f"Semantic search: {len(semantic_results)} results")
            except Exception as e:
                self.stdout.write(f"Semantic search failed: {str(e)}")
        else:
            self.stdout.write("Semantic search not available")
        
        # Check 4: Recent search logs
        self.stdout.write("\n4️⃣ Checking recent search logs...")
        recent_queries = SemanticQuery.objects.order_by('-created_at')[:5]
        if recent_queries:
            for query in recent_queries:
                self.stdout.write(f"   Query: '{query.query_text}' -> {query.query_type} -> {query.querymatch_set.count()} matches")
        else:
            self.stdout.write("   No search logs found")
        
        # Summary and recommendations
        self.stdout.write("\n📝 SUMMARY:")
        self.stdout.write("="*50)
        
        if total_entries == 0:
            self.stdout.write("🔴 PROBLEM: No QA entries in database")
            self.stdout.write("   SOLUTION: Upload Excel file with QA data")
        elif total_entries == 1:
            self.stdout.write("🟡 LIMITED: Only 1 QA entry in database")
            self.stdout.write("   SUGGESTION: Upload more data for better testing")
        else:
            self.stdout.write(f"🟢 DATABASE: {total_entries} QA entries available")
            
        self.stdout.write("\n💡 NEXT STEPS:")
        if total_entries == 0:
            self.stdout.write("   1. Go to /upload-excel/ and upload your FAQ测试.xlsx file")
            self.stdout.write("   2. Make sure to fix the 'import re' error in utils.py first")
            self.stdout.write("   3. Then test search again")
        else:
            self.stdout.write("   1. Try searching for specific SKUs from the sample entries above")
            self.stdout.write("   2. Try searching for keywords from the questions")
            self.stdout.write("   3. Check if semantic search is working with Ollama")