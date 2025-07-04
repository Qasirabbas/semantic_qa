# METHOD 1: Django Management Command (Recommended)
# Create: semantic_qa/management/commands/clear_entries.py

from django.core.management.base import BaseCommand
from semantic_qa.models import QAEntry, SemanticQuery, QueryMatch, Translation

class Command(BaseCommand):
    help = 'Clear all QA entries and related data'

    def add_arguments(self, parser):
        parser.add_argument(
            '--confirm',
            action='store_true',
            help='Confirm deletion without interactive prompt',
        )
        parser.add_argument(
            '--keep-queries',
            action='store_true',
            help='Keep search query logs',
        )

    def handle(self, *args, **options):
        confirm = options['confirm']
        keep_queries = options['keep_queries']
        
        # Count current entries
        qa_count = QAEntry.objects.count()
        query_count = SemanticQuery.objects.count()
        match_count = QueryMatch.objects.count()
        translation_count = Translation.objects.count()
        
        self.stdout.write(f"📊 Current data:")
        self.stdout.write(f"   QA Entries: {qa_count}")
        self.stdout.write(f"   Search Queries: {query_count}")
        self.stdout.write(f"   Query Matches: {match_count}")
        self.stdout.write(f"   Translations: {translation_count}")
        
        if qa_count == 0:
            self.stdout.write(self.style.SUCCESS("✅ No QA entries to delete"))
            return
        
        # Confirm deletion
        if not confirm:
            response = input(f"\n⚠️  This will DELETE ALL {qa_count} QA entries. Are you sure? (yes/no): ")
            if response.lower() != 'yes':
                self.stdout.write("❌ Operation cancelled")
                return
        
        try:
            # Delete in correct order due to foreign key constraints
            deleted_counts = {}
            
            # 1. Delete query matches first
            if not keep_queries:
                deleted_counts['query_matches'] = QueryMatch.objects.count()
                QueryMatch.objects.all().delete()
                self.stdout.write("🗑️  Deleted all query matches")
            
            # 2. Delete QA entries (this will cascade to remaining matches)
            deleted_counts['qa_entries'] = QAEntry.objects.count()
            QAEntry.objects.all().delete()
            self.stdout.write("🗑️  Deleted all QA entries")
            
            # 3. Optionally delete search queries
            if not keep_queries:
                deleted_counts['queries'] = SemanticQuery.objects.count()
                SemanticQuery.objects.all().delete()
                self.stdout.write("🗑️  Deleted all search queries")
            
            # 4. Clear translations (optional)
            deleted_counts['translations'] = Translation.objects.count()
            Translation.objects.all().delete()
            self.stdout.write("🗑️  Deleted all translations")
            
            # Summary
            self.stdout.write(self.style.SUCCESS("\n🎉 Successfully cleared database!"))
            self.stdout.write(f"   QA Entries deleted: {deleted_counts['qa_entries']}")
            if not keep_queries:
                self.stdout.write(f"   Search Queries deleted: {deleted_counts.get('queries', 0)}")
                self.stdout.write(f"   Query Matches deleted: {deleted_counts.get('query_matches', 0)}")
            self.stdout.write(f"   Translations deleted: {deleted_counts['translations']}")
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"❌ Error during deletion: {str(e)}"))