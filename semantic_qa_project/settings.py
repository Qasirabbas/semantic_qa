import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import pymysql
pymysql.install_as_MySQLdb()


# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.getenv('SECRET_KEY', 'django-insecure-change-this-in-production')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'

ALLOWED_HOSTS = os.getenv('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'semantic_qa',
    'corsheaders',  # For API access
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.locale.LocaleMiddleware',  # For internationalization
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'semantic_qa_project.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'django.template.context_processors.i18n',  # For translations
            ],
        },
    },
]

WSGI_APPLICATION = 'semantic_qa_project.wsgi.application'
OCR_CONFIG = {
    'LANGUAGES': ['en', 'ch_sim'],  # English and Simplified Chinese
    'GPU': False,  # Set to True if you have GPU
    'CONFIDENCE_THRESHOLD': 0.3,
}

# Document Processing Configuration
DOCUMENT_PROCESSING = {
    'MAX_FILE_SIZE_MB': 50,
    'SUPPORTED_IMAGE_FORMATS': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'],
    'SUPPORTED_DOCUMENT_FORMATS': ['.pdf'],
    'OCR_TIMEOUT': 300,  # 5 minutes
    'WEB_SCRAPING_TIMEOUT': 30,  # 30 seconds
}
# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'walmart',
        'USER': 'root',
        'PASSWORD': 'Zb_200407',
        'HOST': 'gz-cdb-d4j7h16x.sql.tencentcdb.com',
        'PORT': '22333',
        'OPTIONS': {
            'init_command': "SET sql_mode='STRICT_TRANS_TABLES'",
            'charset': 'utf8mb4',
            'connect_timeout': 10,
        },
        'CONN_MAX_AGE': 60,
    }
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Supported languages
LANGUAGES = [
    ('en', 'English'),
    ('zh', 'Chinese'),
    ('es', 'Spanish'),
    ('fr', 'French'),
    ('de', 'German'),
    ('ja', 'Japanese'),
]

LOCALE_PATHS = [
    BASE_DIR / 'locale',
]

# Static files (CSS, JavaScript, Images)
STATIC_URL = '/static/'
STATICFILES_DIRS = [
    BASE_DIR / 'static',
]
STATIC_ROOT = BASE_DIR / 'staticfiles'

# Media files
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# CORS settings (for API access)
CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]

CORS_ALLOW_CREDENTIALS = True

# ChatGLM Configuration
CHATGLM_CONFIG = {
    'API_KEY': os.getenv('CHATGLM_API_KEY', 'a74b8073a98d4da4a066fc72095f58b0.gulObfhh7fnNcAmp'),
    'BASE_URL': os.getenv('CHATGLM_BASE_URL', 'https://open.bigmodel.cn/api/paas/v4/'),
    'MODEL_NAME': os.getenv('CHATGLM_MODEL', 'glm-4-flash'),
    'TEMPERATURE': float(os.getenv('CHATGLM_TEMPERATURE', '0.7')),
    'MAX_TOKENS': int(os.getenv('CHATGLM_MAX_TOKENS', '2048')),
}

# Hugging Face Embeddings Configuration
EMBEDDINGS_CONFIG = {
    'MODEL_NAME': os.getenv('EMBEDDINGS_MODEL', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'),
    'DEVICE': os.getenv('EMBEDDINGS_DEVICE', 'cpu'),  # Use 'cuda' if GPU available
    'CACHE_DIR': os.getenv('EMBEDDINGS_CACHE_DIR', str(BASE_DIR / 'models_cache')),
    'NORMALIZE_EMBEDDINGS': True,
}

# Alternative embedding models (you can switch by changing EMBEDDINGS_MODEL):
# - 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'  # Good for multilingual
# - 'sentence-transformers/all-MiniLM-L6-v2'  # Fast and good for English
# - 'sentence-transformers/distiluse-base-multilingual-cased'  # Balanced multilingual
# - 'BAAI/bge-m3'  # Good for Chinese and multilingual (requires more memory)

# File Upload Settings
FILE_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024  # 10MB
DATA_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024  # 10MB

# Logging Configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': BASE_DIR / 'semantic_qa.log',
            'formatter': 'verbose',
        },
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
        },
    },
    'loggers': {
        'semantic_qa': {
            'handlers': ['file', 'console'],
            'level': 'INFO',
            'propagate': True,
        },
    },
}

# Cache Configuration (optional, for better performance)
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'semantic-qa-cache',
        'TIMEOUT': 300,  # 5 minutes
        'OPTIONS': {
            'MAX_ENTRIES': 1000,
        }
    }
}

# Session Configuration
SESSION_ENGINE = 'django.contrib.sessions.backends.db'
SESSION_COOKIE_AGE = 86400  # 24 hours

# Security Settings
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = 'DENY'

# Custom Settings for Semantic Search
SEMANTIC_SEARCH_SETTINGS = {
    'SIMILARITY_THRESHOLD': float(os.getenv('SIMILARITY_THRESHOLD', '0.1')),  # Very low threshold
    'MIN_SIMILARITY_THRESHOLD': float(os.getenv('MIN_SIMILARITY_THRESHOLD', '0.01')),  # Fallback threshold
    'MAX_RESULTS': int(os.getenv('MAX_RESULTS', '50')),  # Increased limit
    'ENABLE_TRANSLATION': os.getenv('ENABLE_TRANSLATION', 'True').lower() == 'true',
    'DEFAULT_LANGUAGE': os.getenv('DEFAULT_LANGUAGE', 'en'),
    'ENABLE_KEYWORD_FALLBACK': True,  # Always include keyword search
    'KEYWORD_BOOST_FACTOR': 0.8,  # Boost factor for keyword matches
}

# Email Configuration (for notifications)
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = os.getenv('EMAIL_HOST', 'smtp.gmail.com')
EMAIL_PORT = int(os.getenv('EMAIL_PORT', '587'))
EMAIL_USE_TLS = os.getenv('EMAIL_USE_TLS', 'True').lower() == 'true'
EMAIL_HOST_USER = os.getenv('EMAIL_HOST_USER', '')
EMAIL_HOST_PASSWORD = os.getenv('EMAIL_HOST_PASSWORD', '')
DEFAULT_FROM_EMAIL = os.getenv('DEFAULT_FROM_EMAIL', 'noreply@semanticqa.com')

# Model Download and Cache Settings
MODELS_CACHE_DIR = BASE_DIR / 'models_cache'
MODELS_CACHE_DIR.mkdir(exist_ok=True)

# Ensure the models cache directory exists
os.makedirs(MODELS_CACHE_DIR, exist_ok=True)