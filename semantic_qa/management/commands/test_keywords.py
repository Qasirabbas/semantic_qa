# Create: semantic_qa/management/commands/test_keywords.py

from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = 'Test keyword extraction function'

    def handle(self, *args, **options):
        self.stdout.write("🧪 Testing keyword extraction function...")
        
        try:
            # Test the import and function that was causing errors
            from semantic_qa.utils import extract_keywords_from_text, determine_category
            
            # Test with Chinese text (like your Excel content)
            test_texts = [
                "奔驰E级4.0的安装视频",
                "方向盘控制学习",
                "How to install steering wheel",
                "混合中英文 mixed content test"
            ]
            
            for text in test_texts:
                self.stdout.write(f"\n📝 Testing: '{text}'")
                try:
                    keywords = extract_keywords_from_text(text)
                    category = determine_category(text, text)
                    self.stdout.write(f"   ✅ Keywords: {keywords}")
                    self.stdout.write(f"   ✅ Category: {category}")
                except Exception as e:
                    self.stdout.write(self.style.ERROR(f"   ❌ Error: {str(e)}"))
                    return
            
            self.stdout.write(self.style.SUCCESS("\n🎉 Keyword extraction is working!"))
            self.stdout.write("Now you can upload your Excel file without errors.")
            
        except ImportError as e:
            self.stdout.write(self.style.ERROR(f"❌ Import error: {str(e)}"))
            self.stdout.write("Make sure you added 'import re' to the top of utils.py")
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"❌ Function error: {str(e)}"))
            self.stdout.write("Check the keyword extraction functions in utils.py")