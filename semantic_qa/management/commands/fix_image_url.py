from django.core.management.base import BaseCommand
from semantic_qa.models import QAEntry
from semantic_qa.utils import clean_image_url

class Command(BaseCommand):
    help = 'Fix malformed image URLs in existing QA entries'

    def handle(self, *args, **options):
        self.stdout.write("🔧 Fixing image URLs in existing QA entries...")
        
        # Get entries with image links
        entries_with_images = QAEntry.objects.exclude(image_link='').exclude(image_link__isnull=True)
        total_entries = entries_with_images.count()
        
        if total_entries == 0:
            self.stdout.write(self.style.SUCCESS("✅ No entries with image links found"))
            return
        
        self.stdout.write(f"📊 Found {total_entries} entries with image links")
        
        fixed_count = 0
        error_count = 0
        
        for entry in entries_with_images:
            try:
                original_url = entry.image_link
                cleaned_url = clean_image_url(original_url)
                
                if original_url != cleaned_url:
                    self.stdout.write(f"🔧 Fixing URL for {entry.sku}:")
                    self.stdout.write(f"   Before: {original_url}")
                    self.stdout.write(f"   After:  {cleaned_url}")
                    
                    entry.image_link = cleaned_url
                    entry.save()
                    fixed_count += 1
                
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f"❌ Error fixing URL for entry {entry.sku}: {str(e)}")
                )
                error_count += 1
        
        self.stdout.write(
            self.style.SUCCESS(
                f"\n🎉 URL fixing completed!\n"
                f"   Total entries checked: {total_entries}\n"
                f"   URLs fixed: {fixed_count}\n"
                f"   Errors: {error_count}"
            )
        )