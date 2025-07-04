// semantic_qa/static/semantic_qa/js/main.js

// Global utilities and common functionality
class SemanticQA {
    constructor() {
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupImageHandling();
        this.setupTooltips();
        this.setupLanguageSelector();
        this.setupCopyFunctionality();
    }

    setupEventListeners() {
        // Handle form submissions
        document.addEventListener('submit', this.handleFormSubmit.bind(this));
        
        // Handle click events
        document.addEventListener('click', this.handleClicks.bind(this));
        
        // Handle search input
        const searchInputs = document.querySelectorAll('input[type="search"], input[name="q"]');
        searchInputs.forEach(input => {
            input.addEventListener('input', this.debounce(this.handleSearchInput.bind(this), 300));
        });
    }

    setupCopyFunctionality() {
        // Enhanced event delegation for copy buttons
        document.addEventListener('click', (e) => {
            const target = e.target.closest('[class*="copy"], [data-action="copy"]');
            
            if (!target) return;

            e.preventDefault();

            // Copy result buttons
            if (target.classList.contains('copy-result')) {
                const resultId = target.getAttribute('data-result-id') || target.dataset.resultId;
                if (resultId) {
                    this.copyResult(resultId);
                    return;
                }
            }

            // Copy entry buttons
            if (target.classList.contains('copy-entry')) {
                const entryId = target.getAttribute('data-entry-id') || target.dataset.entryId;
                if (entryId) {
                    this.copyEntry(entryId);
                    return;
                }
            }

            // Copy button with data-text or data-content
            if (target.classList.contains('copy-btn') || target.hasAttribute('data-text') || target.hasAttribute('data-content')) {
                this.copyContent(target);
                return;
            }

            // Generic copy buttons
            if (target.textContent.includes('复制') || target.textContent.includes('Copy')) {
                this.copyContent(target);
            }
        });

        console.log('✅ Enhanced copy functionality initialized');
    }

    handleFormSubmit(e) {
        // Add loading state to submit buttons
        const submitBtn = e.target.querySelector('button[type="submit"]');
        if (submitBtn && !submitBtn.disabled) {
            const originalText = submitBtn.innerHTML;
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
            
            // Reset after 10 seconds as fallback
            setTimeout(() => {
                submitBtn.disabled = false;
                submitBtn.innerHTML = originalText;
            }, 10000);
        }
    }

    handleClicks(e) {
        // Handle various click events
        if (e.target.matches('[data-action="share"]')) {
            this.shareContent(e.target.dataset.content);
        }
    }

    handleSearchInput(e) {
        const query = e.target.value.trim();
        if (query.length > 2) {
            // Could implement search suggestions here
            console.log('Search input:', query);
        }
    }

    setupImageHandling() {
        // Setup image lazy loading
        if ('IntersectionObserver' in window) {
            const imageObserver = new IntersectionObserver((entries, observer) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        const img = entry.target;
                        if (img.dataset.src) {
                            img.src = img.dataset.src;
                            img.classList.remove('lazy');
                            observer.unobserve(img);
                        }
                    }
                });
            });

            document.querySelectorAll('img[data-src]').forEach(img => {
                imageObserver.observe(img);
            });
        }

        // Setup image modal
        this.setupImageModal();
    }

    setupImageModal() {
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('result-image')) {
                this.openImageModal(e.target.src, e.target.alt);
            }
        });
    }

    openImageModal(src, alt) {
        const modal = document.createElement('div');
        modal.className = 'image-modal fade-in';
        modal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.8);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 10000;
            cursor: pointer;
        `;
        
        modal.innerHTML = `
            <img src="${src}" alt="${alt}" style="max-width: 90%; max-height: 90%; object-fit: contain;">
            <div style="position: absolute; top: 20px; right: 20px;">
                <button class="btn btn-light btn-sm" onclick="this.closest('.image-modal').remove()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
        
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.remove();
            }
        });

        document.body.appendChild(modal);
        
        // Close on escape
        const handleEscape = (e) => {
            if (e.key === 'Escape') {
                modal.remove();
                document.removeEventListener('keydown', handleEscape);
            }
        };
        document.addEventListener('keydown', handleEscape);
    }

    setupTooltips() {
        // Initialize Bootstrap tooltips
        if (typeof bootstrap !== 'undefined') {
            const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
            tooltipTriggerList.map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl));
        }
    }

    setupLanguageSelector() {
        const languageLinks = document.querySelectorAll('.language-selector a');
        languageLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const lang = link.dataset.lang;
                if (lang) {
                    this.changeLanguage(lang);
                }
            });
        });
    }

    changeLanguage(lang) {
        const url = new URL(window.location);
        url.searchParams.set('lang', lang);
        window.location.href = url.toString();
    }

    /**
     * Enhanced copy to clipboard function with comprehensive content extraction
     * @param {string} text - Text to copy to clipboard
     * @param {string} successMessage - Custom success message (optional)
     * @param {string} errorMessage - Custom error message (optional)
     * @returns {Promise<boolean>} - Returns true if successful, false otherwise
     */
    copyToClipboard(text, successMessage = '复制成功！', errorMessage = '复制失败') {
        return new Promise((resolve) => {
            // Ensure text is a string
            if (typeof text !== 'string') {
                text = String(text);
            }

            // Clean up the text (remove extra whitespace, normalize line breaks)
            text = text.trim().replace(/\s+/g, ' ').replace(/\n\s+/g, '\n');

            // Method 1: Modern Clipboard API (preferred)
            if (navigator.clipboard && window.isSecureContext) {
                navigator.clipboard.writeText(text)
                    .then(() => {
                        this.showToast(successMessage, 'success');
                        console.log('✅ Copy successful (Clipboard API):', text.substring(0, 50) + '...');
                        resolve(true);
                    })
                    .catch((err) => {
                        console.warn('❌ Clipboard API failed:', err);
                        // Fallback to legacy method
                        this.legacyCopyToClipboard(text, successMessage, errorMessage).then(resolve);
                    });
            } 
            // Method 2: Legacy fallback for older browsers or non-HTTPS
            else {
                this.legacyCopyToClipboard(text, successMessage, errorMessage).then(resolve);
            }
        });
    }

    /**
     * Legacy copy method using document.execCommand
     * @param {string} text 
     * @param {string} successMessage 
     * @param {string} errorMessage 
     * @returns {Promise<boolean>}
     */
    legacyCopyToClipboard(text, successMessage, errorMessage) {
        return new Promise((resolve) => {
            try {
                // Create temporary textarea element
                const textArea = document.createElement('textarea');
                textArea.value = text;
                
                // Style to make it invisible but functional
                textArea.style.cssText = `
                    position: fixed;
                    top: -9999px;
                    left: -9999px;
                    width: 1px;
                    height: 1px;
                    padding: 0;
                    margin: 0;
                    border: none;
                    outline: none;
                    box-shadow: none;
                    background: transparent;
                    opacity: 0;
                    pointer-events: none;
                `;
                
                // Add to DOM
                document.body.appendChild(textArea);
                
                // Focus and select
                textArea.focus();
                textArea.select();
                textArea.setSelectionRange(0, textArea.value.length);

                // Try to copy
                const successful = document.execCommand('copy');
                
                // Remove from DOM
                document.body.removeChild(textArea);
                
                if (successful) {
                    this.showToast(successMessage, 'success');
                    console.log('✅ Copy successful (Legacy method):', text.substring(0, 50) + '...');
                    resolve(true);
                } else {
                    throw new Error('execCommand returned false');
                }
                
            } catch (err) {
                console.error('❌ Legacy copy failed:', err);
                this.showToast(errorMessage, 'error');
                resolve(false);
            }
        });
    }

    /**
     * Enhanced function to extract and format content from search results
     * @param {Element} resultElement - The result element to extract content from
     * @returns {Object} - Formatted content object
     */
    extractResultContent(resultElement) {
        const content = resultElement.textContent || resultElement.innerText || '';
        const extracted = {
            question: '',
            answer: '',
            sku: '',
            title: '',
            content: '',
            category: '',
            relevance: '',
            imageUrl: '',
            links: []
        };

        try {
            // Extract structured content using various patterns
            const patterns = {
                question: /(?:问题|Question)[：:]\s*(.*?)(?=\n|答案|Answer|$)/s,
                answer: /(?:答案|Answer)[：:]\s*(.*?)(?=\n(?:SKU|问题|相关度|分类|$))/s,
                sku: /SKU[：:]\s*(.*?)(?=\n|$)/,
                title: /(?:标题|Title)[：:]\s*(.*?)(?=\n|$)/,
                category: /(?:分类|Category)[：:]\s*(.*?)(?=\n|$)/,
                relevance: /(?:相关度|Relevance)[：:]\s*(.*?)(?=\n|$)/,
                content: /(?:内容|Content)[：:]\s*(.*?)(?=\n(?:文档|相关度|$))/s,
                document: /(?:文档|Document)[：:]\s*(.*?)(?=\n|$)/
            };

            // Extract using patterns
            Object.keys(patterns).forEach(key => {
                const match = content.match(patterns[key]);
                if (match && match[1]) {
                    extracted[key] = match[1].trim();
                }
            });

            // Extract image URL
            const imgElement = resultElement.querySelector('img') || 
                            resultElement.closest('.result-card').querySelector('img');

            if (imgElement) {
                let imgSrc = imgElement.src || imgElement.dataset.src || imgElement.getAttribute('data-src');
                
                // Remove the proxy URL and get the original image URL
                if (imgSrc && imgSrc.includes('/image-proxy/')) {
                    // Extract the original URL from the proxy URL
                    const proxyPath = '/image-proxy/';
                    const proxyIndex = imgSrc.indexOf(proxyPath);
                    if (proxyIndex !== -1) {
                        imgSrc = imgSrc.substring(proxyIndex + proxyPath.length);
                        imgSrc = decodeURIComponent(imgSrc);
                    }
                }
                
                if (imgSrc && !imgSrc.includes('data:') && !imgSrc.includes('placeholder')) {
                    extracted.imageUrl = imgSrc;
                }
            }

            // Extract all links
            const linkElements = resultElement.querySelectorAll('a[href]');
            linkElements.forEach(link => {
                if (link.href && !link.href.startsWith('javascript:') && !link.href.includes('#')) {
                    extracted.links.push({
                        text: link.textContent.trim(),
                        url: link.href
                    });
                }
            });

            // If no structured content found, use fallback
            if (!extracted.question && !extracted.answer && !extracted.content) {
                extracted.content = content.trim();
            }

        } catch (error) {
            console.warn('Content extraction failed, using raw text:', error);
            extracted.content = content.trim();
        }

        return extracted;
    }

    /**
     * Format extracted content into copyable text
     * @param {Object} extracted - Extracted content object
     * @returns {string} - Formatted text ready for copying
     */
    formatContentForCopy(extracted) {
        const lines = [];

        // Add answer or content
        if (extracted.answer) {
            lines.push(extracted.answer);
        } else if (extracted.content) {
            lines.push(extracted.content);
        }

        // Add image URL if available - REMOVE TRAILING SLASH
        if (extracted.imageUrl) {
            let cleanImageUrl = extracted.imageUrl;
            
            // Remove trailing slash
            if (cleanImageUrl.endsWith('/')) {
                cleanImageUrl = cleanImageUrl.slice(0, -1);
            }
            
            lines.push(`图片链接: ${cleanImageUrl}`);
        }

        // Add other links if available
        if (extracted.links.length > 0) {
            lines.push('\n相关链接:');
            extracted.links.forEach(link => {
                let cleanLinkUrl = link.url;
                
                // Remove trailing slash from other links too
                if (cleanLinkUrl.endsWith('/')) {
                    cleanLinkUrl = cleanLinkUrl.slice(0, -1);
                }
                
                lines.push(`- ${link.text}: ${cleanLinkUrl}`);
            });
        }

        return lines.join('\n');
    }

    /**
     * Copy search result content with enhanced formatting
     * @param {string|number} resultId - Result ID or index
     */
    copyResult(resultId) {
        const resultElement = document.getElementById('result-' + resultId);
        
        if (!resultElement) {
            console.error('Result element not found:', 'result-' + resultId);
            this.showToast('未找到要复制的内容', 'error');
            return;
        }

        // Extract and format content
        const extracted = this.extractResultContent(resultElement);
        const formattedContent = this.formatContentForCopy(extracted);

        // Copy to clipboard
        this.copyToClipboard(formattedContent, '搜索结果已复制到剪贴板！');
    }

    /**
     * Copy entry content
     * @param {string|number} entryId - Entry ID
     */
    copyEntry(entryId) {
        const entryElement = document.getElementById(`entry-content-${entryId}`);
        
        if (!entryElement) {
            console.error('Entry element not found:', `entry-content-${entryId}`);
            this.showToast('未找到条目内容', 'error');
            return;
        }

        const extracted = this.extractResultContent(entryElement);
        const formattedContent = this.formatContentForCopy(extracted);

        this.copyToClipboard(formattedContent, '条目已复制到剪贴板！');
    }

    /**
     * Copy content from element with data attributes
     * @param {Element} element - Element with data-text or data-content
     */
    copyContent(element) {
        let textToCopy = '';
        
        if (typeof element === 'string') {
            textToCopy = element;
        } else if (element.dataset && element.dataset.text) {
            textToCopy = element.dataset.text;
        } else if (element.dataset && element.dataset.content) {
            textToCopy = element.dataset.content;
        } else {
            // Try to find content in nearby elements
            const parent = element.closest('.card, .result-card, .entry-card, .chunk-card');
            if (parent) {
                const extracted = this.extractResultContent(parent);
                textToCopy = this.formatContentForCopy(extracted);
            } else {
                textToCopy = element.textContent.trim();
            }
        }

        this.copyToClipboard(textToCopy);
    }

    /**
     * Copy all visible search results
     */
    copyAllResults() {
        const visibleResults = document.querySelectorAll('.result-card:not([style*="display: none"]) [id^="result-"]');
        
        if (visibleResults.length === 0) {
            this.showToast('没有可复制的结果', 'warning');
            return;
        }

        const searchQuery = document.querySelector('input[name="q"]')?.value || '未知查询';
        const timestamp = new Date().toLocaleString();
        
        let allContent = `搜索查询: ${searchQuery}\n`;
        allContent += `搜索时间: ${timestamp}\n`;
        allContent += `结果数量: ${visibleResults.length}\n`;
        allContent += `来源页面: ${window.location.href}\n`;
        allContent += `${'='.repeat(60)}\n\n`;

        visibleResults.forEach((result, index) => {
            const extracted = this.extractResultContent(result);
            const formatted = this.formatContentForCopy(extracted);
            allContent += `=== 结果 ${index + 1} ===\n`;
            allContent += formatted + '\n\n';
        });

        this.copyToClipboard(allContent, `已复制 ${visibleResults.length} 个搜索结果！`);
    }

    shareContent(content) {
        if (navigator.share) {
            navigator.share({
                title: 'Search Result',
                text: content,
                url: window.location.href
            }).then(() => {
                this.showToast('Shared successfully!', 'success');
            }).catch(() => {
                this.copyToClipboard(content);
                this.showToast('Copied to clipboard for sharing', 'info');
            });
        } else {
            this.copyToClipboard(content);
            this.showToast('Copied to clipboard for sharing', 'info');
        }
    }

    showToast(message, type = 'info') {
        // Remove existing toasts to avoid clutter
        document.querySelectorAll('.custom-toast').forEach(toast => toast.remove());

        const toast = document.createElement('div');
        toast.className = `alert alert-${type === 'error' ? 'danger' : type} custom-toast position-fixed fade show`;
        toast.style.cssText = `
            top: 20px;
            right: 20px;
            z-index: 10000;
            min-width: 300px;
            max-width: 500px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            border-radius: 8px;
            animation: slideInRight 0.3s ease;
        `;
        
        const iconMap = {
            success: 'fas fa-check-circle',
            error: 'fas fa-times-circle',
            warning: 'fas fa-exclamation-triangle',
            info: 'fas fa-info-circle'
        };

        toast.innerHTML = `
            <div class="d-flex align-items-center">
                <i class="${iconMap[type] || iconMap.info} me-2"></i>
                <span class="flex-grow-1">${message}</span>
                <button type="button" class="btn-close ms-2" onclick="this.closest('.alert').remove()"></button>
            </div>
        `;
        
        document.body.appendChild(toast);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (toast.parentNode) {
                toast.style.animation = 'slideOutRight 0.3s ease';
                setTimeout(() => toast.remove(), 300);
            }
        }, 5000);
    }

    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    // Utility methods
    static formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    static formatDate(date) {
        return new Intl.DateTimeFormat('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        }).format(new Date(date));
    }
}

// Search specific functionality
class SearchHandler {
    constructor() {
        this.apiUrl = '/semantic_qa/api/search/';
        this.init();
    }

    init() {
        this.setupSearchForm();
        this.setupFilters();
        console.log('SearchHandler initialized');
    }

    setupSearchForm() {
        // Handle regular search form
        const searchForm = document.getElementById('searchForm');
        if (searchForm) {
            searchForm.addEventListener('submit', this.handleSearch.bind(this));
        }

        // Handle AJAX search form
        const ajaxSearchForm = document.getElementById('ajaxSearchForm');
        if (ajaxSearchForm) {
            ajaxSearchForm.addEventListener('submit', this.handleAjaxSearch.bind(this));
            console.log('AJAX search form found and attached');
        } else {
            console.warn('AJAX search form not found - check form ID');
        }

        // Check for results container
        const resultsContainer = document.getElementById('searchResults');
        if (!resultsContainer) {
            console.warn('Search results container not found - check container ID');
        }
    }

    setupFilters() {
        const filterInputs = document.querySelectorAll('.search-filter');
        filterInputs.forEach(input => {
            input.addEventListener('change', this.applyFilters.bind(this));
        });
    }

    handleSearch(e) {
        const form = e.target;
        const query = form.querySelector('input[name="q"]')?.value?.trim();
        
        if (!query) {
            e.preventDefault();
            window.app.showToast('Please enter a search query', 'warning');
            return;
        }
        
        console.log('Regular search submitted:', query);
        // Let the form submit normally for regular search
    }

    async handleAjaxSearch(e) {
        e.preventDefault();
        const form = e.target;
        const formData = new FormData(form);
        const query = formData.get('q')?.trim();
        const language = formData.get('lang') || 'en';

        if (!query) {
            window.app.showToast('Please enter a search query', 'warning');
            return;
        }

        console.log('Searching for:', query);
        this.showSearchLoading(true);

        try {
            const response = await fetch(this.apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': this.getCSRFToken()
                },
                body: JSON.stringify({
                    query: query,
                    language: language
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log('Search response:', data);

            if (data.success) {
                this.displaySearchResults(data);
                window.app.showToast(`Found ${data.results?.length || 0} results`, 'success');
            } else {
                console.error('Search failed:', data.error);
                window.app.showToast(data.error || 'Search failed', 'error');
            }
        } catch (error) {
            console.error('Search error:', error);
            window.app.showToast('Search failed. Please try again.', 'error');
        } finally {
            this.showSearchLoading(false);
        }
    }

    displaySearchResults(data) {
        const resultsContainer = document.getElementById('searchResults');
        if (!resultsContainer) {
            console.error('Search results container not found');
            return;
        }

        // Clear previous results
        resultsContainer.innerHTML = '';

        if (!data.results || data.results.length === 0) {
            resultsContainer.innerHTML = `
                <div class="text-center py-5">
                    <i class="fas fa-search fa-3x text-muted mb-3"></i>
                    <h4>No results found</h4>
                    <p class="text-muted">Try different keywords or check for typos.</p>
                </div>
            `;
            return;
        }

        let html = `
            <div class="search-meta mb-4">
                <div class="row align-items-center">
                    <div class="col-md-6">
                        <p class="text-muted mb-0">
                            Found ${data.total_results || data.results.length} results for "<strong>${data.query}</strong>"
                            ${data.response_time ? `(${data.response_time.toFixed(3)}s)` : ''}
                        </p>
                    </div>
                    <div class="col-md-6 text-end">
                        <button class="btn btn-outline-primary btn-sm" onclick="window.app.copyAllResults()">
                            <i class="fas fa-copy me-1"></i>Copy All Results
                        </button>
                        <button class="btn btn-outline-secondary btn-sm" onclick="window.searchHandler.exportResults()">
                            <i class="fas fa-download me-1"></i>Export Results
                        </button>
                    </div>
                </div>
            </div>
        `;

        // Add filter chips if there are multiple result types
        const resultTypes = [...new Set(data.results.map(r => r.category || r.type))];
        if (resultTypes.length > 1) {
            html += this.createFilterChips(data.results);
        }

        // Add results
        data.results.forEach((result, index) => {
            html += this.createResultCard(result, index);
        });

        resultsContainer.innerHTML = html;
        resultsContainer.scrollIntoView({ behavior: 'smooth' });

        // Initialize filter functionality
        this.initializeFilters();
    }

    createFilterChips(results) {
        const types = results.reduce((acc, result) => {
            const type = result.category || result.type || 'other';
            acc[type] = (acc[type] || 0) + 1;
            return acc;
        }, {});

        let html = `
            <div class="filter-chips mb-4">
                <div class="d-flex flex-wrap gap-2">
                    <button class="btn btn-outline-primary btn-sm filter-chip active" data-filter="all">
                        All (${results.length})
                    </button>
        `;

        Object.entries(types).forEach(([type, count]) => {
            const displayName = type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
            html += `
                <button class="btn btn-outline-secondary btn-sm filter-chip" data-filter="${type}">
                    ${displayName} (${count})
                </button>
            `;
        });

        html += `
                </div>
            </div>
        `;

        return html;
    }

    createResultCard(result, index) {
        const relevanceClass = this.getRelevanceClass(result.relevance_score || result.score);
        const categoryDisplay = (result.category || result.type || 'general').replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
        
        return `
            <div class="card result-card mb-4 fade-in" data-type="${result.category || result.type}" style="animation-delay: ${index * 0.1}s">
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-${result.image_link ? '8' : '12'}">
                            <div class="d-flex justify-content-between align-items-start mb-2">
                                <div class="d-flex gap-2">
                                    <span class="badge bg-primary">${categoryDisplay}</span>
                                    <span class="badge ${relevanceClass}">
                                        ${Math.round((result.relevance_score || result.score || 0) * 100)}% match
                                    </span>
                                </div>
                                <div class="dropdown">
                                    <button class="btn btn-sm btn-outline-secondary dropdown-toggle" data-bs-toggle="dropdown">
                                        <i class="fas fa-ellipsis-v"></i>
                                    </button>
                                    <ul class="dropdown-menu">
                                        <li><a class="dropdown-item copy-result" href="#" data-result-id="${index}">
                                            <i class="fas fa-copy me-2"></i>Copy Result
                                        </a></li>
                                        <li><a class="dropdown-item" href="#" onclick="window.app.shareContent(document.getElementById('result-${index}').textContent)">
                                            <i class="fas fa-share me-2"></i>Share
                                        </a></li>
                                    </ul>
                                </div>
                            </div>
                            
                            ${result.sku ? `<h6 class="text-primary mb-2">SKU: ${result.sku}</h6>` : ''}
                            
                            <div class="result-content" id="result-${index}">
                                ${this.formatResultContent(result)}
                            </div>
                            
                            ${result.metadata ? this.createMetadataSection(result.metadata) : ''}
                        </div>
                        
                        ${result.image_link ? `
                            <div class="col-md-4">
                                <div class="image-container" id="image-container-${index}">
                                    <img src="${result.image_link}" 
                                         alt="Image for ${result.sku || 'result'}"
                                         class="img-fluid result-image rounded"
                                         loading="lazy"
                                         style="cursor: pointer; max-height: 200px; object-fit: cover;"
                                         onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                                    <div class="text-center p-3 bg-light rounded" style="display: none;">
                                        <i class="fas fa-image text-muted"></i>
                                        <p class="text-muted small mb-0">Image unavailable</p>
                                    </div>
                                </div>
                            </div>
                        ` : ''}
                    </div>
                </div>
            </div>
        `;
    }

    formatResultContent(result) {
        let content = '';
        
        if (result.question && result.answer) {
            content = `
                <div class="qa-content">
                    <div class="question mb-2">
                        <strong>问题: </strong>${result.question}
                    </div>
                    <div class="answer">
                        <strong>答案: </strong>${result.answer}
                    </div>
                </div>
            `;
        } else if (result.content) {
            content = `<div class="content"><strong>内容: </strong>${result.content}</div>`;
        } else if (result.text) {
            content = `<div class="content"><strong>内容: </strong>${result.text}</div>`;
        }
        
        return content;
    }

    createMetadataSection(metadata) {
        if (!metadata || typeof metadata !== 'object') return '';
        
        const items = Object.entries(metadata)
            .filter(([key, value]) => value !== null && value !== undefined && value !== '')
            .map(([key, value]) => `<small class="text-muted">${key}: ${value}</small>`)
            .join(' | ');
            
        return items ? `<div class="metadata mt-2">${items}</div>` : '';
    }

    getRelevanceClass(score) {
        if (score >= 0.8) return 'bg-success';
        if (score >= 0.6) return 'bg-warning';
        return 'bg-secondary';
    }

    initializeFilters() {
        const filterChips = document.querySelectorAll('.filter-chip');
        filterChips.forEach(chip => {
            chip.addEventListener('click', () => {
                // Remove active class from all chips
                filterChips.forEach(c => c.classList.remove('active'));
                chip.classList.add('active');
                
                // Apply filter
                const filter = chip.dataset.filter;
                this.filterResults(filter);
            });
        });
    }

    filterResults(filter) {
        const resultCards = document.querySelectorAll('.result-card');
        
        resultCards.forEach(card => {
            if (filter === 'all' || card.dataset.type === filter) {
                card.style.display = 'block';
                card.style.animation = 'fadeIn 0.3s ease-in';
            } else {
                card.style.display = 'none';
            }
        });
    }

    showSearchLoading(show) {
        const searchBtn = document.querySelector('#ajaxSearchForm button[type="submit"]');
        const resultsContainer = document.getElementById('searchResults');

        if (show) {
            if (searchBtn) {
                searchBtn.disabled = true;
                searchBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Searching...';
            }
            if (resultsContainer) {
                resultsContainer.innerHTML = `
                    <div class="text-center py-5">
                        <div class="spinner-border text-primary" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                        <p class="mt-3 text-muted">Searching...</p>
                    </div>
                `;
            }
        } else {
            if (searchBtn) {
                searchBtn.disabled = false;
                searchBtn.innerHTML = '<i class="fas fa-search me-2"></i>Search';
            }
        }
    }

    getCSRFToken() {
        const token = document.querySelector('[name=csrfmiddlewaretoken]');
        return token ? token.value : '';
    }

    applyFilters() {
        console.log('Applying filters...');
        // Implementation for applying search filters
    }

    exportResults() {
        const results = document.querySelectorAll('.result-card:not([style*="display: none"])');
        let content = `Search Results Export\n`;
        content += `Generated: ${new Date().toLocaleString()}\n`;
        content += `Total Results: ${results.length}\n`;
        content += `Source URL: ${window.location.href}\n\n`;
        
        results.forEach((result, index) => {
            const resultContent = result.querySelector('.result-content');
            if (resultContent) {
                const extracted = window.app.extractResultContent(resultContent);
                const formatted = window.app.formatContentForCopy(extracted);
                content += `=== Result ${index + 1} ===\n`;
                content += formatted + '\n\n';
            }
        });
        
        const blob = new Blob([content], { type: 'text/plain; charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `search-results-${new Date().getTime()}.txt`;
        a.click();
        URL.revokeObjectURL(url);
        
        window.app.showToast('Results exported successfully!', 'success');
    }
}

// File upload handler
class FileUploadHandler {
    constructor() {
        this.init();
    }

    init() {
        this.setupFileInput();
        this.setupDragAndDrop();
    }

    setupFileInput() {
        const fileInputs = document.querySelectorAll('input[type="file"]');
        fileInputs.forEach(input => {
            input.addEventListener('change', this.handleFileSelect.bind(this));
        });
    }

    setupDragAndDrop() {
        const dropZones = document.querySelectorAll('.file-drop-zone');
        dropZones.forEach(zone => {
            zone.addEventListener('dragover', this.handleDragOver.bind(this));
            zone.addEventListener('dragenter', this.handleDragEnter.bind(this));
            zone.addEventListener('dragleave', this.handleDragLeave.bind(this));
            zone.addEventListener('drop', this.handleDrop.bind(this));
        });
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.validateAndPreviewFile(file, e.target);
        }
    }

    handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    handleDragEnter(e) {
        e.preventDefault();
        e.stopPropagation();
        e.target.classList.add('drag-over');
    }

    handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        e.target.classList.remove('drag-over');
    }

    handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        e.target.classList.remove('drag-over');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            const fileInput = e.target.querySelector('input[type="file"]');
            if (fileInput) {
                fileInput.files = files;
                this.validateAndPreviewFile(files[0], fileInput);
            }
        }
    }

    validateAndPreviewFile(file, input) {
        // Validate file type
        const allowedTypes = input.accept ? 
            input.accept.split(',').map(t => t.trim()) : 
            ['.xlsx', '.xls', '.pdf', '.txt', '.doc', '.docx'];
        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
        
        if (!allowedTypes.includes(fileExtension)) {
            window.app.showToast(`Invalid file type. Allowed: ${allowedTypes.join(', ')}`, 'error');
            input.value = '';
            return;
        }

        // Validate file size (10MB limit)
        if (file.size > 10 * 1024 * 1024) {
            window.app.showToast('File size must be less than 10MB', 'error');
            input.value = '';
            return;
        }

        // Show file preview
        this.showFilePreview(file, input);
    }

    showFilePreview(file, input) {
        const previewContainer = input.closest('.form-group')?.querySelector('.file-preview') ||
                                input.closest('.upload-zone')?.querySelector('.file-preview');
        
        if (previewContainer) {
            previewContainer.innerHTML = `
                <div class="file-info p-3 border rounded bg-light">
                    <div class="d-flex align-items-center">
                        <i class="fas fa-file-${this.getFileIcon(file.name)} text-primary me-3 fa-2x"></i>
                        <div class="flex-grow-1">
                            <div class="fw-bold">${file.name}</div>
                            <small class="text-muted">${SemanticQA.formatFileSize(file.size)}</small>
                        </div>
                        <button type="button" class="btn btn-sm btn-outline-danger ms-auto" 
                                onclick="this.closest('.file-info').remove(); document.querySelector('input[type=file]').value='';">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                </div>
            `;
        }
    }

    getFileIcon(filename) {
        const ext = filename.split('.').pop().toLowerCase();
        const iconMap = {
            'pdf': 'pdf',
            'xlsx': 'excel',
            'xls': 'excel',
            'doc': 'word',
            'docx': 'word',
            'txt': 'alt',
            'csv': 'csv'
        };
        return iconMap[ext] || 'file';
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, initializing application...');
    
    // Initialize main application
    window.app = new SemanticQA();
    window.searchHandler = new SearchHandler();
    window.fileUploadHandler = new FileUploadHandler();
    
    // Add smooth scrolling to anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            const target = document.getElementById(targetId);
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
    
    // Auto-hide alerts after 5 seconds
    setTimeout(() => {
        document.querySelectorAll('.alert:not(.alert-permanent)').forEach(alert => {
            const closeBtn = alert.querySelector('.btn-close');
            if (closeBtn) {
                closeBtn.click();
            }
        });
    }, 5000);

    // Handle legacy search form
    const searchForm = document.getElementById('searchForm');
    const searchQuery = document.getElementById('searchQuery');
    if (searchForm && searchQuery) {
        searchForm.addEventListener('submit', function(e) {
            if (!searchQuery.value.trim()) {
                e.preventDefault();
                window.app.showToast('Please enter search content', 'warning');
            }
        });
    }

    // Add keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + K: Focus search input
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            const searchInput = document.querySelector('input[name="q"]') || document.getElementById('searchQuery');
            if (searchInput) {
                searchInput.focus();
            }
        }
        
        // Escape: Clear search or close modals
        if (e.key === 'Escape') {
            const modals = document.querySelectorAll('.modal.show, .image-modal');
            modals.forEach(modal => {
                if (modal.classList.contains('image-modal')) {
                    modal.remove();
                } else {
                    const bsModal = bootstrap.Modal.getInstance(modal);
                    if (bsModal) bsModal.hide();
                }
            });
            
            // Clear search if no modals
            if (modals.length === 0) {
                const searchInput = document.querySelector('input[name="q"]');
                if (searchInput && searchInput.value) {
                    searchInput.value = '';
                    searchInput.focus();
                }
            }
        }
        
        // Ctrl/Cmd + Enter: Submit search form
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            const searchForm = document.getElementById('ajaxSearchForm') || document.getElementById('searchForm');
            if (searchForm) {
                searchForm.dispatchEvent(new Event('submit'));
            }
        }

        // Ctrl/Cmd + C: Copy current result (if focused)
        if ((e.ctrlKey || e.metaKey) && e.key === 'c' && e.target.closest('.result-card')) {
            const resultCard = e.target.closest('.result-card');
            const resultContent = resultCard.querySelector('[id^="result-"]');
            if (resultContent) {
                const resultId = resultContent.id.replace('result-', '');
                window.app.copyResult(resultId);
            }
        }
    });

    console.log('Application initialized successfully');
});

// Scroll to top functionality
window.addEventListener('scroll', function() {
    let scrollButton = document.getElementById('scrollTopBtn');
    
    if (!scrollButton) {
        scrollButton = document.createElement('button');
        scrollButton.id = 'scrollTopBtn';
        scrollButton.className = 'btn btn-primary position-fixed';
        scrollButton.style.cssText = `
            bottom: 20px;
            right: 20px;
            z-index: 1000;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            display: none;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        `;
        scrollButton.innerHTML = '<i class="fas fa-arrow-up"></i>';
        scrollButton.onclick = function() {
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        };
        scrollButton.title = 'Back to top';
        document.body.appendChild(scrollButton);
    }
    
    if (window.pageYOffset > 300) {
        scrollButton.style.display = 'block';
    } else {
        scrollButton.style.display = 'none';
    }
});

// Legacy support functions for existing HTML
function copyToClipboard(text, successMessage, errorMessage) {
    if (window.app) {
        return window.app.copyToClipboard(text, successMessage, errorMessage);
    }
}

function showToast(message, type) {
    if (window.app) {
        window.app.showToast(message, type);
    }
}

function copyResult(resultId) {
    if (window.app) {
        window.app.copyResult(resultId);
    }
}

function copyAllResults() {
    if (window.app) {
        window.app.copyAllResults();
    }
}

function formatFileSize(bytes) {
    return SemanticQA.formatFileSize(bytes);
}

function formatDate(date) {
    return SemanticQA.formatDate(date);
}

function searchSimilar(sku) {
    const searchInput = document.querySelector('input[name="q"]');
    if (searchInput) {
        searchInput.value = sku;
        const form = searchInput.closest('form');
        if (form) {
            form.dispatchEvent(new Event('submit'));
        }
    }
}

function shareResult(resultId) {
    const resultElement = document.getElementById('result-' + resultId);
    if (resultElement && window.app) {
        const extracted = window.app.extractResultContent(resultElement);
        const formatted = window.app.formatContentForCopy(extracted);
        window.app.shareContent(formatted);
    }
}

function openImageModal(imageSrc, title) {
    if (window.app) {
        window.app.openImageModal(imageSrc, title);
    }
}

function toggleImage(resultId) {
    const container = document.getElementById('image-container-' + resultId);
    if (container) {
        if (container.style.display === 'none') {
            container.style.display = 'block';
            container.classList.add('fade-in');
        } else {
            container.style.display = 'none';
        }
    }
}

function expandAllImages() {
    const containers = document.querySelectorAll('[id^="image-container-"]');
    let count = 0;
    containers.forEach(container => {
        if (container.style.display !== 'block') {
            container.style.display = 'block';
            container.classList.add('fade-in');
            count++;
        }
    });
    showToast(`Displayed ${count} images`, 'success');
}

function copyModalContent() {
    const modalContent = document.querySelector('.modal.show .modal-body');
    if (modalContent && window.app) {
        const content = modalContent.textContent || modalContent.innerText;
        window.app.copyToClipboard(content.trim(), '模态框内容已复制！');
    }
}

function copyChunk(chunkId) {
    const chunkCard = document.querySelector(`button[onclick*="copyChunk(${chunkId})"]`)?.closest('.chunk-card');
    
    if (!chunkCard) {
        const chunkElement = document.getElementById(`chunk-${chunkId}`);
        if (chunkElement && window.app) {
            const extracted = window.app.extractResultContent(chunkElement);
            const formatted = window.app.formatContentForCopy(extracted);
            window.app.copyToClipboard(formatted, '文本块已复制！');
            return;
        }
        
        showToast('未找到文本块', 'error');
        return;
    }

    if (window.app) {
        const extracted = window.app.extractResultContent(chunkCard);
        const formatted = window.app.formatContentForCopy(extracted);
        window.app.copyToClipboard(formatted, '文本块已复制到剪贴板！');
    }
}

// Error handling for unhandled promise rejections
window.addEventListener('unhandledrejection', function(event) {
    console.error('Unhandled promise rejection:', event.reason);
    if (window.app) {
        window.app.showToast('An unexpected error occurred', 'error');
    }
    event.preventDefault();
});

// Performance monitoring
if ('performance' in window) {
    window.addEventListener('load', function() {
        setTimeout(function() {
            const perfData = performance.getEntriesByType('navigation')[0];
            console.log('Page load time:', perfData.loadEventEnd - perfData.loadEventStart, 'ms');
        }, 0);
    });
}

// Add CSS animations and styles
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideInRight {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideOutRight {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    
    .custom-toast {
        animation: slideInRight 0.3s ease;
    }
    
    .result-card {
        transition: all 0.3s ease;
        border-left: 4px solid transparent;
    }
    
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left-color: var(--bs-primary);
    }
    
    .filter-chip {
        transition: all 0.2s ease;
    }
    
    .filter-chip:hover {
        transform: translateY(-1px);
    }
    
    .filter-chip.active {
        background-color: var(--bs-primary) !important;
        border-color: var(--bs-primary) !important;
        color: white !important;
    }
    
    .image-container img {
        transition: all 0.3s ease;
    }
    
    .image-container img:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .drag-over {
        border: 2px dashed var(--bs-primary) !important;
        background-color: rgba(var(--bs-primary-rgb), 0.1) !important;
    }
    
    .spinner-border {
        animation: spinner-border 0.75s linear infinite;
    }
    
    @keyframes spinner-border {
        to { transform: rotate(360deg); }
    }
    
    .search-meta {
        border-bottom: 1px solid #dee2e6;
        padding-bottom: 1rem;
    }
    
    .qa-content .question {
        color: var(--bs-primary);
        font-weight: 500;
    }
    
    .qa-content .answer {
        color: var(--bs-dark);
        line-height: 1.6;
    }
    
    .metadata {
        border-top: 1px solid #dee2e6;
        padding-top: 0.5rem;
        font-size: 0.875rem;
    }
    
    .file-info {
        transition: all 0.3s ease;
    }
    
    .file-info:hover {
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Copy button enhancements */
    [class*="copy"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    
    [class*="copy"].copying {
        opacity: 0.7;
        pointer-events: none;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .filter-chips .d-flex {
            flex-direction: column;
        }
        
        .filter-chip {
            margin-bottom: 0.5rem;
        }
        
        .result-card .row {
            flex-direction: column;
        }
        
        .image-container {
            margin-top: 1rem;
        }
        
        #scrollTopBtn {
            bottom: 15px !important;
            right: 15px !important;
            width: 45px !important;
            height: 45px !important;
        }
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .result-card:hover {
            box-shadow: 0 4px 15px rgba(255,255,255,0.1);
        }
        
        .image-container img:hover {
            box-shadow: 0 4px 15px rgba(255,255,255,0.2);
        }
    }
`;
document.head.appendChild(style);