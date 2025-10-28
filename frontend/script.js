// AlMenn Frontend JavaScript
class AlMennApp {
    constructor() {
        this.apiBase = 'http://localhost:8001';
        this.wsBase = 'ws://localhost:8001';
        this.token = localStorage.getItem('token');
        this.currentUser = null;
        this.currentSession = null;
        this.webSockets = {};
        this.mathJaxInitialized = false;
        this.mermaidInitialized = false;
        this.darkMode = localStorage.getItem('darkMode') === 'true';

        this.init();
    }

    init() {
        this.setupEventListeners();
        this.initializeLibraries();
        this.applyTheme();
        this.checkAuthentication();
        this.showPage('home');
    }

    setupEventListeners() {
        // Navigation
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const page = e.target.dataset.page;
                this.showPage(page);
            });
        });

        // Mobile menu
        document.getElementById('mobile-menu-button').addEventListener('click', () => {
            document.getElementById('mobile-menu').classList.toggle('hidden');
        });

        // Theme toggle
        document.getElementById('theme-toggle').addEventListener('click', () => this.toggleTheme());

        // Login form
        document.getElementById('login-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.sendMagicLink();
        });

        // AI Chat
        document.getElementById('start-session').addEventListener('click', () => this.startSession());
        document.getElementById('extend-session').addEventListener('click', () => this.extendSession());
        document.getElementById('send-message').addEventListener('click', () => this.sendMessage());
        document.getElementById('chat-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendMessage();
        });

        // File upload
        document.getElementById('file-upload').addEventListener('change', (e) => this.uploadFiles(e.target.files));

        // Global chat
        document.getElementById('send-global').addEventListener('click', () => this.sendGlobalMessage());
        document.getElementById('global-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendGlobalMessage();
        });

        // Admin chat
        document.getElementById('send-admin').addEventListener('click', () => this.sendAdminMessage());
        document.getElementById('admin-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendAdminMessage();
        });

        // Buy coins
        document.querySelectorAll('.buy-coins').forEach(button => {
            button.addEventListener('click', (e) => this.buyCoins(e.target));
        });

        // Check for magic link in URL
        this.checkMagicLink();
    }

    initializeLibraries() {
        // Initialize Mermaid
        if (typeof mermaid !== 'undefined') {
            mermaid.initialize({ startOnLoad: false });
            this.mermaidInitialized = true;
        }

        // MathJax is loaded via CDN
        if (typeof MathJax !== 'undefined') {
            this.mathJaxInitialized = true;
        }
    }

    toggleTheme() {
        this.darkMode = !this.darkMode;
        localStorage.setItem('darkMode', this.darkMode);
        this.applyTheme();
    }

    applyTheme() {
        const body = document.body;
        const html = document.documentElement;
        const themeIconSun = document.getElementById('theme-icon-sun');
        const themeIconMoon = document.getElementById('theme-icon-moon');

        if (this.darkMode) {
            html.classList.add('dark');
            body.classList.add('dark');
            themeIconSun.classList.add('hidden');
            themeIconMoon.classList.remove('hidden');
        } else {
            html.classList.remove('dark');
            body.classList.remove('dark');
            themeIconSun.classList.remove('hidden');
            themeIconMoon.classList.add('hidden');
        }
    }

    checkAuthentication() {
        if (this.token) {
            this.validateToken();
        } else {
            this.forceLogin();
        }
    }

    forceLogin() {
        // Force login on every page load
        this.showPage('login');
    }

    async validateToken() {
        try {
            const response = await this.apiRequest('/me');
            if (response.ok) {
                this.currentUser = await response.json();
                this.updateUI();
            } else {
                this.logout();
            }
        } catch (error) {
            console.error('Token validation failed:', error);
            this.logout();
        }
    }

    logout() {
        localStorage.removeItem('token');
        this.token = null;
        this.currentUser = null;
        this.disconnectWebSockets();
        this.showPage('login');
    }

    updateUI() {
        if (this.currentUser) {
            // Show authenticated navigation
            document.getElementById('login-link').classList.add('hidden');
            document.getElementById('mobile-login-link').classList.add('hidden');

            // Show admin link if user is admin
            if (this.currentUser.is_admin) {
                document.getElementById('admin-link').classList.remove('hidden');
                document.getElementById('mobile-admin-link').classList.remove('hidden');
            }

            // Update coin balance
            this.updateCoinBalance();
        }
    }

    showPage(pageId) {
        // Hide all pages
        document.querySelectorAll('.page').forEach(page => {
            page.classList.add('hidden');
        });

        // Show selected page
        const page = document.getElementById(pageId);
        if (page) {
            page.classList.remove('hidden');
        }

        // Update URL hash
        window.location.hash = pageId;

        // Page-specific initialization
        switch (pageId) {
            case 'ai-chat':
                this.initializeAIChat();
                break;
            case 'global-chat':
                this.initializeGlobalChat();
                break;
            case 'admin-chat':
                this.initializeAdminChat();
                break;
            case 'wallet':
                this.loadWalletData();
                break;
        }
    }

    async sendMagicLink() {
        const email = document.getElementById('email').value;
        const messageDiv = document.getElementById('login-message');

        try {
            this.showLoading();
            const response = await this.apiRequest('/auth/send_magic_link', {
                method: 'POST',
                body: JSON.stringify({ email })
            });

            if (response.ok) {
                messageDiv.textContent = 'Magic link sent! Check your email.';
                messageDiv.className = 'mt-4 text-center text-sm text-green-600';
            } else {
                throw new Error('Failed to send magic link');
            }
        } catch (error) {
            messageDiv.textContent = 'Error sending magic link. Please try again.';
            messageDiv.className = 'mt-4 text-center text-sm text-red-600';
        } finally {
            this.hideLoading();
            messageDiv.classList.remove('hidden');
        }
    }

    checkMagicLink() {
        const urlParams = new URLSearchParams(window.location.search);
        const token = urlParams.get('token');

        if (token) {
            this.token = token;
            localStorage.setItem('token', token);
            // Clean URL
            window.history.replaceState({}, document.title, window.location.pathname);
            this.validateToken();
        }
    }

    async initializeAIChat() {
        if (!this.currentUser) return;

        // Load user files
        await this.loadUserFiles();

        // Check session status
        await this.checkSessionStatus();

        // Connect to AI WebSocket if needed
        // Note: AI queries are typically HTTP, WebSocket might be for real-time updates
    }

    async loadUserFiles() {
        try {
            // This would typically come from the backend
            // For now, show placeholder
            const fileList = document.getElementById('file-list');
            fileList.innerHTML = '<p class="text-gray-500 text-sm">No files uploaded yet</p>';
        } catch (error) {
            console.error('Failed to load user files:', error);
        }
    }

    async checkSessionStatus() {
        try {
            const response = await this.apiRequest('/ai/session/status');
            if (response.ok) {
                const status = await response.json();
                this.updateSessionUI(status);
            }
        } catch (error) {
            console.error('Failed to check session status:', error);
        }
    }

    updateSessionUI(status) {
        const sessionTime = document.getElementById('session-time');
        const startBtn = document.getElementById('start-session');
        const extendBtn = document.getElementById('extend-session');

        if (status.active) {
            sessionTime.textContent = `${status.minutes_left} min`;
            startBtn.classList.add('hidden');
            extendBtn.classList.remove('hidden');
        } else {
            sessionTime.textContent = '0 min';
            startBtn.classList.remove('hidden');
            extendBtn.classList.add('hidden');
        }
    }

    async startSession() {
        try {
            this.showLoading();
            const response = await this.apiRequest('/ai/session/start', {
                method: 'POST'
            });

            if (response.ok) {
                const result = await response.json();
                this.currentSession = result.session_id;
                await this.checkSessionStatus();
                this.updateCoinBalance();
            } else {
                throw new Error('Failed to start session');
            }
        } catch (error) {
            this.showError('Failed to start session. Please check your coin balance.');
        } finally {
            this.hideLoading();
        }
    }

    async extendSession() {
        try {
            this.showLoading();
            const response = await this.apiRequest('/ai/session/extend', {
                method: 'POST'
            });

            if (response.ok) {
                await this.checkSessionStatus();
                this.updateCoinBalance();
            } else {
                throw new Error('Failed to extend session');
            }
        } catch (error) {
            this.showError('Failed to extend session. Please check your coin balance.');
        } finally {
            this.hideLoading();
        }
    }

    async sendMessage() {
        const input = document.getElementById('chat-input');
        const message = input.value.trim();

        if (!message) return;

        // Add user message to chat
        this.addMessage('user', message);
        input.value = '';

        try {
            const response = await this.apiRequest('/ai/query', {
                method: 'POST',
                body: JSON.stringify({
                    query: message,
                    session_id: this.currentSession
                })
            });

            if (response.ok) {
                const result = await response.json();
                this.addMessage('ai', result.response);

                // Render any visual content
                this.renderVisualContent(result.response);
            } else {
                throw new Error('Failed to get AI response');
            }
        } catch (error) {
            this.addMessage('ai', 'Sorry, I encountered an error. Please try again.');
        }
    }

    addMessage(sender, content) {
        const messagesDiv = document.getElementById('chat-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `flex ${sender === 'user' ? 'justify-end' : 'justify-start'}`;

        const messageContent = document.createElement('div');
        messageContent.className = `max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
            sender === 'user'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-200 text-gray-800'
        }`;
        messageContent.innerHTML = this.formatMessage(content);

        messageDiv.appendChild(messageContent);
        messagesDiv.appendChild(messageDiv);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }

    formatMessage(content) {
        // Basic formatting - in a real app, this would be more sophisticated
        return content.replace(/\n/g, '<br>');
    }

    renderVisualContent(content) {
        // Render MathJax
        if (this.mathJaxInitialized && content.includes('$')) {
            MathJax.typeset();
        }

        // Render Mermaid diagrams
        if (this.mermaidInitialized && content.includes('```mermaid')) {
            const mermaidBlocks = content.match(/```mermaid\n([\s\S]*?)\n```/g);
            if (mermaidBlocks) {
                mermaidBlocks.forEach((block, index) => {
                    const diagramId = `mermaid-diagram-${Date.now()}-${index}`;
                    const diagramContent = block.replace(/```mermaid\n|\n```/g, '');
                    content = content.replace(block, `<div id="${diagramId}" class="mermaid">${diagramContent}</div>`);
                });

                // Re-render after DOM update
                setTimeout(() => {
                    mermaid.init();
                }, 100);
            }
        }
    }

    async uploadFiles(files) {
        for (const file of files) {
            try {
                const formData = new FormData();
                formData.append('file', file);

                const response = await fetch(`${this.apiBase}/ai/upload_file`, {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${this.token}`
                    },
                    body: formData
                });

                if (response.ok) {
                    const result = await response.json();
                    this.addFileToList(result);
                } else {
                    throw new Error('Upload failed');
                }
            } catch (error) {
                this.showError(`Failed to upload ${file.name}`);
            }
        }
    }

    addFileToList(fileData) {
        const fileList = document.getElementById('file-list');
        const fileDiv = document.createElement('div');
        fileDiv.className = 'flex items-center justify-between p-2 bg-white rounded border';
        fileDiv.innerHTML = `
            <span class="text-sm">${fileData.filename}</span>
            <button class="text-red-500 hover:text-red-700 text-sm" onclick="removeFile('${fileData.id}')">Remove</button>
        `;
        fileList.appendChild(fileDiv);
    }

    async initializeGlobalChat() {
        this.connectWebSocket('global', '/ws/global');
    }

    async initializeAdminChat() {
        if (!this.currentUser?.is_admin) {
            this.showPage('home');
            return;
        }
        // Admin chat would connect to specific user WebSocket
        // For now, placeholder
    }

    connectWebSocket(name, endpoint) {
        if (this.webSockets[name]) {
            this.webSockets[name].close();
        }

        const ws = new WebSocket(`${this.wsBase}${endpoint}?token=${this.token}`);
        this.webSockets[name] = ws;

        ws.onopen = () => {
            console.log(`${name} WebSocket connected`);
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(name, data);
        };

        ws.onclose = () => {
            console.log(`${name} WebSocket disconnected`);
            delete this.webSockets[name];
        };

        ws.onerror = (error) => {
            console.error(`${name} WebSocket error:`, error);
        };
    }

    handleWebSocketMessage(type, data) {
        switch (type) {
            case 'global':
                this.addGlobalMessage(data);
                break;
            case 'admin':
                this.addAdminMessage(data);
                break;
        }
    }

    addGlobalMessage(data) {
        const messagesDiv = document.getElementById('global-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = 'flex justify-start';
        messageDiv.innerHTML = `
            <div class="bg-gray-200 text-gray-800 px-4 py-2 rounded-lg max-w-xs lg:max-w-md">
                <strong>${data.username}:</strong> ${data.message}
            </div>
        `;
        messagesDiv.appendChild(messageDiv);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }

    addAdminMessage(data) {
        const messagesDiv = document.getElementById('admin-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `flex ${data.from_admin ? 'justify-start' : 'justify-end'}`;
        messageDiv.innerHTML = `
            <div class="bg-${data.from_admin ? 'red' : 'blue'}-200 text-gray-800 px-4 py-2 rounded-lg max-w-xs lg:max-w-md">
                ${data.message}
            </div>
        `;
        messagesDiv.appendChild(messageDiv);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }

    async sendGlobalMessage() {
        const input = document.getElementById('global-input');
        const message = input.value.trim();

        if (!message || !this.webSockets.global) return;

        this.webSockets.global.send(JSON.stringify({
            type: 'message',
            content: message
        }));

        input.value = '';
    }

    async sendAdminMessage() {
        const input = document.getElementById('admin-input');
        const message = input.value.trim();

        if (!message) return;

        // Admin messaging would be implemented here
        input.value = '';
    }

    async loadWalletData() {
        await Promise.all([
            this.updateCoinBalance(),
            this.loadTransactionHistory()
        ]);
    }

    async updateCoinBalance() {
        try {
            const response = await this.apiRequest('/wallet');
            if (response.ok) {
                const data = await response.json();
                document.getElementById('coin-balance').textContent = data.balance;
                document.getElementById('wallet-balance').textContent = `${data.balance} Coins`;
            }
        } catch (error) {
            console.error('Failed to load coin balance:', error);
        }
    }

    async loadTransactionHistory() {
        try {
            const response = await this.apiRequest('/wallet');
            if (response.ok) {
                const data = await response.json();
                const historyDiv = document.getElementById('transaction-history');
                historyDiv.innerHTML = '';

                if (data.transactions && data.transactions.length > 0) {
                    data.transactions.forEach(transaction => {
                        const transactionDiv = document.createElement('div');
                        transactionDiv.className = 'flex justify-between items-center p-2 border-b';
                        transactionDiv.innerHTML = `
                            <div>
                                <p class="text-sm font-medium">${transaction.description}</p>
                                <p class="text-xs text-gray-500">${new Date(transaction.created_at).toLocaleDateString()}</p>
                            </div>
                            <span class="text-sm ${transaction.amount > 0 ? 'text-green-600' : 'text-red-600'}">
                                ${transaction.amount > 0 ? '+' : ''}${transaction.amount}
                            </span>
                        `;
                        historyDiv.appendChild(transactionDiv);
                    });
                } else {
                    historyDiv.innerHTML = '<p class="text-gray-500 text-sm">No transactions yet</p>';
                }
            }
        } catch (error) {
            console.error('Failed to load transaction history:', error);
        }
    }

    async buyCoins(button) {
        const amount = button.dataset.amount;
        const price = button.dataset.price;

        // In a real app, this would integrate with a payment gateway
        // For now, simulate the purchase
        try {
            this.showLoading();
            const response = await this.apiRequest('/buy_coins', {
                method: 'POST',
                body: JSON.stringify({
                    amount: parseInt(amount),
                    payment_method: 'simulated'
                })
            });

            if (response.ok) {
                await this.loadWalletData();
                this.showSuccess(`Successfully purchased ${amount} coins!`);
            } else {
                throw new Error('Purchase failed');
            }
        } catch (error) {
            this.showError('Purchase failed. Please try again.');
        } finally {
            this.hideLoading();
        }
    }

    async apiRequest(endpoint, options = {}) {
        const url = `${this.apiBase}${endpoint}`;
        const config = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        };

        if (this.token) {
            config.headers['Authorization'] = `Bearer ${this.token}`;
        }

        const response = await fetch(url, config);

        if (response.status === 401) {
            this.logout();
            throw new Error('Authentication required');
        }

        return response;
    }

    disconnectWebSockets() {
        Object.values(this.webSockets).forEach(ws => {
            ws.close();
        });
        this.webSockets = {};
    }

    showLoading() {
        document.getElementById('loading-overlay').classList.remove('hidden');
    }

    hideLoading() {
        document.getElementById('loading-overlay').classList.add('hidden');
    }

    showError(message) {
        const errorModal = document.getElementById('error-modal');
        document.getElementById('error-message').textContent = message;
        errorModal.classList.remove('hidden');
    }

    showSuccess(message) {
        // Simple success notification - in a real app, use a proper notification system
        alert(message);
    }
}

function closeErrorModal() {
    document.getElementById('error-modal').classList.add('hidden');
}

function removeFile(fileId) {
    // Implement file removal
    console.log('Remove file:', fileId);
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.alMennApp = new AlMennApp();
});

// Handle browser back/forward buttons
window.addEventListener('hashchange', () => {
    const page = window.location.hash.substring(1) || 'home';
    if (window.alMennApp) {
        window.alMennApp.showPage(page);
    }
});
