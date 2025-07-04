from django.apps import AppConfig


class SemanticQaConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'semantic_qa'


    def ready(self):
            import semantic_qa.signals  #