/* Native browser UI: no build step, and no provider credentials in the browser. */
const chatLog = document.getElementById('chat-log');
const chatInput = document.getElementById('chat-input');
const sendButton = document.getElementById('send-button');
const requestStatus = document.getElementById('request-status');
const elapsed = document.getElementById('request-elapsed');
const main = document.querySelector('main');
const converter = window.showdown ? new showdown.Converter({ tables: true }) : null;
let busy = false;

function scrollToLatest() {
    main.scrollTop = main.scrollHeight;
}

function renderContent(element, content, format = 'text') {
    if (format === 'markdown' && converter && window.DOMPurify?.isSupported) {
        element.classList.add('markdown-content');
        element.innerHTML = DOMPurify.sanitize(converter.makeHtml(content), {
            ALLOWED_TAGS: ['p', 'br', 'strong', 'em', 'del', 'code', 'pre', 'ul', 'ol', 'li',
                'blockquote', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'a', 'table', 'thead', 'tbody', 'tr', 'th', 'td', 'hr'],
            ALLOWED_ATTR: ['href', 'title'],
        });
        element.querySelectorAll('pre').forEach(pre => pre.classList.add('codeboxcontainer'));
    } else {
        element.classList.add('plain-message');
        element.textContent = content;
    }
}

function addMessage(message, sources = [], isUser = false, status = '') {
    const entry = document.createElement('section');
    entry.className = 'message';
    const heading = document.createElement('h4');
    heading.textContent = isUser ? 'You' : 'eLi';
    entry.appendChild(heading);
    const body = document.createElement('div');
    renderContent(body, message.replace(/\xa0/g, ' '), isUser ? 'text' : 'markdown');
    entry.appendChild(body);
    if (!isUser && Array.isArray(sources) && sources.length) {
        const label = document.createElement('p');
        label.className = 'sources-label';
        label.textContent = `${sources.length} reading location${sources.length === 1 ? '' : 's'}`;
        entry.appendChild(label);
        const list = document.createElement('ol');
        list.className = 'reading-sources';
        for (const source of sources) {
            const item = document.createElement('li');
            const link = document.createElement('a');
            link.textContent = source.file + (source.page ? ` · PDF page ${source.page}` : '');
            // Only the local document route is a valid reading location.
            try {
                const url = new URL(source.url, window.location.origin);
                if (url.origin === window.location.origin && url.pathname.startsWith('/documents/')) {
                    link.href = url.href;
                    link.target = '_blank';
                    link.rel = 'noopener noreferrer';
                }
            } catch { /* Keep the filename as plain text if its URL is malformed. */ }
            item.appendChild(link);
            const hasContent = typeof source.content === 'string' && source.content.trim();
            if (!hasContent || source.format !== 'markdown') {
                const preview = document.createElement('p');
                preview.className = 'source-preview';
                preview.textContent = source.preview || '';
                item.appendChild(preview);
            }
            if (hasContent) {
                const excerpt = document.createElement('details');
                excerpt.className = 'source-excerpt';
                excerpt.open = source.format === 'markdown';
                const summary = document.createElement('summary');
                summary.textContent = 'Read excerpt';
                excerpt.appendChild(summary);
                const content = document.createElement('div');
                content.className = 'source-content';
                renderContent(content, source.content, source.format);
                excerpt.appendChild(content);
                item.appendChild(excerpt);
            }
            list.appendChild(item);
        }
        entry.appendChild(list);
    } else if (!isUser && status === 'no_match') {
        const empty = document.createElement('p');
        empty.className = 'sources-label';
        empty.textContent = 'No relevant reading locations found.';
        entry.appendChild(empty);
    }
    chatLog.appendChild(entry);
    scrollToLatest();
}

async function sendMessage(event) {
    event.preventDefault();
    const message = chatInput.value.trim();
    if (busy || !message) return;
    busy = true;
    addMessage(message, [], true);
    chatInput.value = '';
    chatInput.disabled = sendButton.disabled = true;
    chatLog.setAttribute('aria-busy', 'true');
    requestStatus.hidden = false;
    elapsed.textContent = '0s';
    scrollToLatest();
    const start = Date.now();
    const ticker = setInterval(() => { elapsed.textContent = `${Math.floor((Date.now() - start) / 1000)}s`; }, 1000);
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 90000);
    try {
        const response = await fetch('/api/ask', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message }), signal: controller.signal,
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.error || `Request failed (HTTP ${response.status}). Please try again.`);
        if (typeof data.response !== 'string') throw new Error('The server returned an invalid answer. Please try again.');
        addMessage(data.response, data.sources, false, data.status);
    } catch (error) {
        const description = error.name === 'AbortError'
            ? 'The request took too long. Please try again.'
            : error.message || 'Unable to reach the server. Please try again.';
        addMessage(`Error: ${description}`);
    } finally {
        clearInterval(ticker);
        clearTimeout(timeout);
        requestStatus.hidden = true;
        chatLog.setAttribute('aria-busy', 'false');
        chatInput.disabled = sendButton.disabled = false;
        busy = false;
        chatInput.focus({ preventScroll: true });
        scrollToLatest();
    }
}

document.getElementById('chat-form').addEventListener('submit', sendMessage);
