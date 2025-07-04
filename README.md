# Semantic QA System

A comprehensive Django-based Semantic Question-Answering System with RAG (Retrieval-Augmented Generation) capabilities, document processing, and multilingual support.

## 🚀 Features

### Core Functionality
- **Semantic Search**: Advanced semantic search using sentence transformers
- **RAG Integration**: Retrieval-Augmented Generation using ChatGLM API
- **Document Processing**: Support for PDF, DOCX, TXT, and web scraping
- **Multilingual Support**: Translation services and international language support
- **Hybrid Search**: Combined exact, semantic, and keyword matching

### Document Management
- **Multi-format Support**: PDF, DOCX, TXT, and web URLs
- **OCR Processing**: Optical Character Recognition for image-based documents
- **Batch Upload**: Process multiple documents simultaneously
- **Text Chunking**: Smart text segmentation for better retrieval

### Analytics & Monitoring
- **Real-time Analytics**: Query performance and usage statistics
- **Processing Jobs**: Background task monitoring
- **User Activity Tracking**: IP-based activity logging
- **System Health**: Service status monitoring

### Administration
- **Django Admin**: Full administrative interface
- **System Configuration**: Dynamic configuration management
- **Data Export**: Excel template generation and bulk operations
- **User Management**: IP-based access tracking

## 🛠️ Technology Stack

- **Backend**: Django 4.x, Python 3.8+
- **Database**: SQLite (default), PostgreSQL/MySQL supported
- **AI/ML**: 
  - ChatGLM API for text generation
  - Sentence Transformers for embeddings
  - Hugging Face Transformers
- **Document Processing**: 
  - PyPDF2, python-docx
  - PaddleOCR for OCR
  - BeautifulSoup for web scraping
- **Frontend**: Bootstrap 5, jQuery
- **Task Queue**: Background job processing
- **Storage**: Django file storage system

## 📦 Installation

### Prerequisites
```bash
# Python 3.8 or higher
python --version

# pip package manager
pip --version
```

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/semantic-qa-system.git
cd semantic-qa-system
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Environment Configuration**
Create a `.env` file in the project root:
```env
DEBUG=True
SECRET_KEY=your-secret-key-here
CHATGLM_API_KEY=your-chatglm-api-key
DATABASE_URL=sqlite:///db.sqlite3
ALLOWED_HOSTS=localhost,127.0.0.1
```

5. **Database Setup**
```bash
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser
```

6. **Static Files**
```bash
python manage.py collectstatic
```

7. **Run the Development Server**
```bash
python manage.py runserver
```

Visit `http://localhost:8000` to access the application.

## 📋 Requirements

Create a `requirements.txt` file with the following dependencies:

```txt
Django>=4.2,<5.0
djangorestframework>=3.14.0
django-cors-headers>=4.0.0
openai>=1.0.0
langchain>=0.1.0
langchain-community>=0.0.20
sentence-transformers>=2.2.0
transformers>=4.30.0
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
openpyxl>=3.1.0
PyPDF2>=3.0.0
python-docx>=0.8.11
beautifulsoup4>=4.12.0
requests>=2.31.0
pillow>=10.0.0
paddlepaddle>=2.5.0
paddleocr>=2.7.0
python-dotenv>=1.0.0
celery>=5.3.0
redis>=4.5.0
```

## 🔧 Configuration

### ChatGLM API Setup
1. Register at [ChatGLM Platform](https://open.bigmodel.cn/)
2. Get your API key
3. Add to system configuration or environment variables

### Embedding Models
The system uses Hugging Face sentence transformers:
- Default: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Configurable via admin interface

### Document Processing
- OCR: PaddleOCR (supports multiple languages)
- PDF: PyPDF2 for text extraction
- Web: BeautifulSoup for content scraping

## 🚀 Usage

### Basic Search
1. Navigate to the home page
2. Enter your question in the search box
3. Choose search options (QA entries, documents, or both)
4. Enable RAG for AI-generated responses

### Document Upload
1. Go to "Manage Documents"
2. Upload PDF, DOCX, or TXT files
3. Or add web URLs for scraping
4. Monitor processing status

### QA Management
1. Access "Manage QA Entries"
2. Add/edit questions and answers
3. Categorize and add keywords
4. Include relevant images

### Analytics
1. Visit "Analytics Dashboard"
2. View query statistics
3. Monitor system performance
4. Track user activity

## 📊 API Endpoints

### Search API
```http
POST /api/search/
Content-Type: application/json

{
  "query": "your question here",
  "use_rag": true,
  "document_types": ["pdf", "docx"],
  "max_results": 10
}
```

### Document Upload API
```http
POST /api/documents/upload/
Content-Type: multipart/form-data

{
  "file": "document.pdf",
  "title": "Document Title",
  "category": "technical"
}
```

## 🔍 Advanced Features

### RAG (Retrieval-Augmented Generation)
- Combines document retrieval with AI generation
- Uses ChatGLM for natural language responses
- Context-aware answer generation

### Semantic Search
- Vector embeddings for similarity matching
- Multi-language support
- Hybrid scoring with exact and semantic matches

### Document Processing Pipeline
1. **Upload**: File validation and storage
2. **Extract**: Text extraction with OCR if needed
3. **Chunk**: Smart text segmentation
4. **Embed**: Vector embedding generation
5. **Index**: Database storage for retrieval

## 🌐 Multilingual Support

### Supported Languages
- English (en)
- Chinese (zh)
- Spanish (es)
- French (fr)
- German (de)
- Japanese (ja)

### Translation Features
- Automatic query translation
- Response translation
- Cached translation storage

## 📈 Performance Optimization

### Caching Strategy
- Query result caching
- Embedding caching
- Translation caching

### Database Optimization
- Indexed fields for fast searches
- Efficient query structure
- Pagination for large datasets

## 🔒 Security

### Features
- IP-based activity tracking
- Input validation and sanitization
- CSRF protection
- XSS prevention

### Best Practices
- Regular security updates
- API key protection
- File upload validation
- Database security

## 🧪 Testing

Run tests with:
```bash
python manage.py test
```

### Test Coverage
- Unit tests for models
- Integration tests for views
- API endpoint testing
- Document processing tests

## 📁 Project Structure

```
semantic_qa_project/
├── semantic_qa/               # Main application
│   ├── models.py             # Database models
│   ├── views.py              # View functions
│   ├── forms.py              # Django forms
│   ├── admin.py              # Admin configuration
│   ├── rag_service.py        # RAG implementation
│   ├── document_processor.py # Document processing
│   ├── translations_service.py # Translation service
│   └── utils.py              # Utility functions
├── templates/                # HTML templates
├── static/                   # CSS, JS, images
├── media/                    # User uploaded files
├── manage.py                 # Django management
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Common Issues
- **OCR not working**: Ensure PaddleOCR is properly installed
- **ChatGLM API errors**: Check API key and network connectivity
- **Slow search**: Consider database optimization and caching

### Getting Help
- Create an issue on GitHub
- Check the documentation
- Review the logs for error details

## 🏗️ Roadmap

### Upcoming Features
- [ ] Advanced analytics dashboard
- [ ] Real-time collaboration
- [ ] Mobile app integration
- [ ] Cloud deployment templates
- [ ] Enhanced multilingual support
- [ ] Performance optimizations

## 👥 Authors

- **Your Name** - Initial work - [YourGitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- ChatGLM for AI capabilities
- Hugging Face for transformer models
- Django community for the framework
- PaddleOCR for OCR functionality
- All contributors and testers

---

**Note**: This is a development version. For production deployment, ensure proper security configurations, environment variables, and database setup.
