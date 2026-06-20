'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { api, CacheStatus, EvalReport, QueryResponse, SourceReference } from '@/lib/api';
import ConfidenceBadge from '@/components/ConfidenceBadge';
import DocumentList from '@/components/DocumentList';
import EvalModal from '@/components/EvalModal';
import UploadZone from '@/components/UploadZone';
import CacheModal from '@/components/CacheModal';

// ── Types ─────────────────────────────────────────────────────────────────────

interface Toast {
  id: number;
  message: string;
  type: 'success' | 'error' | 'info';
}

interface Message {
  id: number;
  role: 'user' | 'assistant';
  content: string;
  sources?: SourceReference[];
  confidence?: number;
  thinking?: boolean;
  cancelled?: boolean;
}

const EXAMPLE_QUERIES = [
  'What is the notice period mentioned in this document?',
  'What are the key skills listed on the resume?',
  'Who are the main characters in the story?',
  'What is the capital of France?',
  'Summarize the key points from page 2.',
];

// ── Main Page ──────────────────────────────────────────────────────────────────

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [documents, setDocuments] = useState<string[]>([]);
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [evalReport, setEvalReport] = useState<EvalReport | null>(null);
  const [evalRunning, setEvalRunning] = useState(false);
  const [evalPairs, setEvalPairs] = useState(10);
  const [cacheStatus, setCacheStatus] = useState<CacheStatus | null>(null);
  const [showCacheModal, setShowCacheModal] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  // AbortController ref — holds the controller for the in-flight query
  const abortControllerRef = useRef<AbortController | null>(null);
  const sessionIdRef = useRef<string>('');
  let toastId = useRef(0);
  let msgId = useRef(0);

  const [isLoaded, setIsLoaded] = useState(false);

  // Load documents on mount
  useEffect(() => {
    const existing = window.localStorage.getItem('rag_session_id');
    const sessionId = existing || crypto.randomUUID();
    window.localStorage.setItem('rag_session_id', sessionId);
    sessionIdRef.current = sessionId;

    const savedMsgs = window.localStorage.getItem('rag_chat_messages');
    if (savedMsgs) {
      try {
        const parsed = JSON.parse(savedMsgs);
        setMessages(parsed);
        if (parsed.length > 0) {
          msgId.current = Math.max(...parsed.map((m: any) => m.id));
        }
      } catch (e) {
        console.error('Failed to load chat history', e);
      }
    }
    setIsLoaded(true);

    api.listDocuments()
      .then(setDocuments)
      .catch(() => addToast('Could not connect to backend', 'error'));

    // Check cache status
    const checkCache = async () => {
      try {
        const status = await api.getCacheStatus();
        setCacheStatus(status);
        if (!status.is_ready) {
          setTimeout(checkCache, 2000); // poll every 2s
        }
      } catch (err) {
        console.error('Failed to get cache status', err);
      }
    };
    checkCache();
  }, []);

  const saveMessagesToStorage = (msgs: Message[]) => {
    const savableMessages = msgs.filter(m => !m.thinking);
    if (savableMessages.length > 0) {
      window.localStorage.setItem('rag_chat_messages', JSON.stringify(savableMessages));
    } else {
      window.localStorage.removeItem('rag_chat_messages');
    }
  };

  // Auto-scroll chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Auto-resize textarea
  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setQuestion(e.target.value);
    e.target.style.height = 'auto';
    e.target.style.height = `${Math.min(e.target.scrollHeight, 120)}px`;
  };

  const addToast = (message: string, type: Toast['type']) => {
    const id = ++toastId.current;
    setToasts(t => [...t, { id, message, type }]);
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), 4000);
  };

  const addMessage = (msg: Omit<Message, 'id'>) => {
    const id = ++msgId.current;
    setMessages(prev => {
      const next = [...prev, { ...msg, id }];
      saveMessagesToStorage(next);
      return next;
    });
    return id;
  };

  const updateMessage = (id: number, updates: Partial<Message>) => {
    setMessages(prev => {
      const next = prev.map(m => m.id === id ? { ...m, ...updates } : m);
      saveMessagesToStorage(next);
      return next;
    });
  };

  // ── Stop handler ────────────────────────────────────────────────────────────
  const handleStop = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
  }, []);

  // ── Submit handler ──────────────────────────────────────────────────────────
  const handleSubmit = useCallback(async () => {
    const q = question.trim();
    if (!q || loading) return;

    setQuestion('');
    if (textareaRef.current) { textareaRef.current.style.height = 'auto'; }

    // Add user message
    addMessage({ role: 'user', content: q });

    // Add thinking placeholder
    const thinkId = addMessage({ role: 'assistant', content: '', thinking: true });
    setLoading(true);

    // Create a new AbortController for this request
    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const response: QueryResponse = await api.query(q, undefined, controller.signal, sessionIdRef.current);
      updateMessage(thinkId, {
        content: response.answer,
        sources: response.sources,
        confidence: response.confidence,
        thinking: false,
      });
    } catch (err: unknown) {
      // DOMException name 'AbortError' = user cancelled
      if (err instanceof Error && err.name === 'AbortError') {
        updateMessage(thinkId, {
          content: '⏹ *Query stopped by user.*',
          thinking: false,
          cancelled: true,
          confidence: undefined,
        });
        addToast('Query cancelled', 'info');
      } else {
        updateMessage(thinkId, {
          content: `⚠️ Error: ${err instanceof Error ? err.message : 'Backend unreachable. Is the server running?'}`,
          thinking: false,
          confidence: 0,
        });
      }
    } finally {
      abortControllerRef.current = null;
      setLoading(false);
    }
  }, [question, loading]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleUploadSuccess = (docName: string) => {
    setDocuments(prev => prev.includes(docName) ? prev : [...prev, docName]);
  };

  const handleDeleteDoc = (name: string) => {
    setDocuments(prev => prev.filter(d => d !== name));
  };

  const handleSummarizeDoc = async (name: string) => {
    const thinkId = addMessage({ role: 'assistant', content: '', thinking: true });
    try {
      const info = await api.getDocumentInfo(name);
      updateMessage(thinkId, {
        content: info.summary
          ? `## Summary: ${info.document}\n\n${info.summary}`
          : `No generated summary is available for "${info.document}". Re-upload the PDF to generate one automatically.`,
        confidence: 1,
        thinking: false,
      });
    } catch (err: unknown) {
      updateMessage(thinkId, {
        content: `Error loading summary: ${err instanceof Error ? err.message : 'Unknown error'}`,
        thinking: false,
        confidence: 0,
      });
    }
  };

  const handleRunEval = async () => {
    setEvalRunning(true);
    addToast(`Running evaluation with ${evalPairs} questions…`, 'info');
    try {
      const report = await api.runEvaluation(evalPairs);
      setEvalReport(report);
    } catch (err: unknown) {
      addToast(`Evaluation failed: ${err instanceof Error ? err.message : 'Unknown error'}`, 'error');
    } finally {
      setEvalRunning(false);
    }
  };

  const handleExampleClick = (q: string) => {
    setQuestion(q);
    textareaRef.current?.focus();
  };

  const handleResetAll = async () => {
    if (!confirm('⚠️ This will delete ALL ingested documents from the vector store. Are you sure?')) return;
    try {
      const result = await api.resetAllDocuments();
      setDocuments([]);
      addToast(`Reset complete — removed ${result.documents_removed.length} document(s)`, 'success');
    } catch (err: unknown) {
      addToast(`Reset failed: ${err instanceof Error ? err.message : 'Unknown error'}`, 'error');
    }
  };

  const handleClearCache = async () => {
    try {
      await api.clearCache();
      addToast('Cache cleared successfully', 'success');
    } catch (err: unknown) {
      addToast(`Failed to clear cache: ${err instanceof Error ? err.message : 'Unknown error'}`, 'error');
    }
  };

  const handleClearChat = async () => {
    try {
      await api.clearChat();
      setMessages([]);
      window.localStorage.removeItem('rag_chat_messages');
      addToast('Chat history cleared', 'success');
    } catch (err: unknown) {
      addToast(`Failed to clear chat: ${err instanceof Error ? err.message : 'Unknown error'}`, 'error');
    }
  };

  const isCacheLoading = Boolean(cacheStatus && !cacheStatus.is_ready);
  const cacheLoadingPercent = cacheStatus?.cache_loaded_percent ?? 0;

  return (
    <div className="app-shell">
      {/* ── Sidebar ─────────────────────────────────────────────────────────── */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="sidebar-logo">
            <div className="logo-icon">📄</div>
            <div className="logo-text">DocRAG</div>
          </div>
          <div className="logo-tag">Ask Questions from Any PDF</div>
        </div>

        <div className="sidebar-body">
          <UploadZone onUploadSuccess={handleUploadSuccess} onToast={addToast} />

          <div className="section-label">Corpus ({documents.length})</div>
          <DocumentList
            documents={documents}
            onDelete={handleDeleteDoc}
            onSummarize={handleSummarizeDoc}
            onToast={addToast}
          />
        </div>

        <div className="sidebar-footer">
          {/* Eval pairs slider */}
          <div style={{ marginBottom: 10 }}>
            <div style={{
              display: 'flex', justifyContent: 'space-between',
              fontSize: 11, color: 'var(--text-muted)', marginBottom: 4,
            }}>
              <span>Eval questions</span>
              <span style={{ fontWeight: 700, color: 'var(--accent)' }}>{evalPairs}</span>
            </div>
            <input
              id="eval-pairs-slider"
              type="range"
              min={3}
              max={20}
              value={evalPairs}
              onChange={e => setEvalPairs(Number(e.target.value))}
              style={{ width: '100%', accentColor: 'var(--accent)', cursor: 'pointer' }}
            />
            <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>
              More questions = slower but more accurate evaluation
            </div>
          </div>

          <button
            id="run-evaluation-btn"
            className="eval-btn"
            onClick={handleRunEval}
            disabled={evalRunning || documents.length === 0}
          >
            {evalRunning ? `⏳ Generating ${evalPairs} questions…` : `📊 Run Evaluation (${evalPairs} Q&A)`}
          </button>
          <button
            id="reset-all-btn"
            className="eval-btn"
            onClick={handleResetAll}
            disabled={documents.length === 0}
            style={{ marginTop: '8px', background: 'rgba(220,53,69,0.15)', borderColor: 'rgba(220,53,69,0.4)', color: '#ff6b7a' }}
          >
            🗑️ Reset All Documents
          </button>
          
          <div style={{ display: 'flex', gap: '8px', marginTop: '8px' }}>
            <button
              className="eval-btn"
              onClick={handleClearCache}
              style={{ flex: 1, padding: '6px' }}
            >
              🧹 Clear Cache
            </button>
            <button
              className="eval-btn"
              onClick={handleClearChat}
              style={{ flex: 1, padding: '6px' }}
            >
              💬 Clear Chat
            </button>
          </div>
        </div>
      </aside>

      {/* ── Main Panel ──────────────────────────────────────────────────────── */}
      <main className="main-panel">
        {/* Top Bar */}
        <div className="topbar">
          <div>
            <div className="topbar-title">Document Q&A</div>
            <div className="topbar-sub">
              {documents.length === 0
                ? 'Upload any PDF to begin'
                : `${documents.length} document${documents.length !== 1 ? 's' : ''} indexed`}
            </div>
          </div>
          <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
            {cacheStatus && (
              <div 
                className="status-dot" 
                onClick={() => setShowCacheModal(true)}
                style={{ 
                  background: cacheStatus.is_ready ? 'rgba(16, 185, 129, 0.1)' : 'rgba(245, 158, 11, 0.1)', 
                  color: cacheStatus.is_ready ? '#10b981' : '#f59e0b',
                  padding: '4px 10px', 
                  borderRadius: '12px',
                  cursor: 'pointer',
                  transition: 'background 0.2s'
                }}
                title="Click to view cache entries"
              >
                <div className="dot" style={{ background: cacheStatus.is_ready ? '#10b981' : '#f59e0b' }} />
                Cache: {cacheStatus.is_ready ? 'Ready' : `Loading (${cacheStatus.cache_loaded_percent}%)`}
              </div>
            )}
            <div className="status-dot">
              <div className="dot" />
              nvidia · nvidia-2.5-flash
            </div>
          </div>
        </div>

        {/* Chat Area */}
        <div className="chat-area" id="chat-area">
          {!isLoaded ? null : messages.length === 0 ? (
            <div className="welcome">
              <div className="welcome-glow">📄</div>
              <h2>Ask anything from your documents</h2>
              <p>
                Upload any PDF — contracts, resumes, stories, reports, textbooks —
                and ask questions. Every answer is cited to the exact document and page.
                General questions are answered directly by the AI.
              </p>
              <div className="example-queries">
                {EXAMPLE_QUERIES.map(q => (
                  <button key={q} className="example-query" onClick={() => handleExampleClick(q)}>
                    {q}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map(msg => (
              <div key={msg.id} className="message">
                {msg.role === 'user' ? (
                  <div className="msg-user">{msg.content}</div>
                ) : msg.thinking ? (
                  <div className="thinking">
                    <div className="thinking-dots">
                      <span /><span /><span />
                    </div>
                    Retrieving and generating answer…
                  </div>
                ) : msg.cancelled ? (
                  <div className="msg-assistant" style={{ opacity: 0.6 }}>
                    <div className="msg-answer-card" style={{
                      borderColor: 'rgba(99,102,241,0.2)',
                      background: 'rgba(99,102,241,0.05)',
                    }}>
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                    </div>
                  </div>
                ) : (
                  <div className="msg-assistant">
                    <div className="msg-answer-card">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {msg.content}
                      </ReactMarkdown>
                    </div>

                    <div className="msg-meta">
                      {msg.confidence !== undefined && (
                        <ConfidenceBadge confidence={msg.confidence} />
                      )}
                    </div>
                  </div>
                )}
              </div>
            ))
          )}
          <div ref={chatEndRef} />
        </div>

        {/* Input Bar */}
        <div className="input-bar">
          <div className="input-wrap">
            <textarea
              ref={textareaRef}
              id="question-input"
              className="chat-input"
              placeholder={isCacheLoading ? `Cache loading... (${cacheLoadingPercent}%)` : "Ask a question about your documents… or any general question"}
              value={question}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              rows={1}
              disabled={loading || isCacheLoading}
            />
            {loading ? (
              /* ── Stop button (replaces send while loading) ── */
              <button
                id="stop-query-btn"
                className="send-btn stop-btn"
                onClick={handleStop}
                title="Stop query (Esc)"
                aria-label="Stop query"
              >
                ⏹
              </button>
            ) : (
              <button
                id="submit-question-btn"
                className="send-btn"
                onClick={handleSubmit}
                disabled={loading || isCacheLoading || !question.trim()}
                title="Send (Enter)"
              >
                ➤
              </button>
            )}
          </div>
          <div className="input-hint">
            {isCacheLoading ? (
              <span style={{ color: '#f59e0b' }}>⏳ Cache is warming up. Please wait…</span>
            ) : loading ? (
              <span style={{ color: '#f59e0b' }}>⏳ Generating… click ⏹ to stop</span>
            ) : (
              <>Press <kbd style={{ fontFamily: 'monospace', fontSize: 10 }}>Enter</kbd> to send · Shift+Enter for newline</>
            )}
          </div>
        </div>
      </main>

      {/* ── Toasts ──────────────────────────────────────────────────────────── */}
      <div className="toast-container">
        {toasts.map(t => (
          <div key={t.id} className={`toast ${t.type}`}>
            <span>
              {t.type === 'success' ? '✓' : t.type === 'error' ? '✗' : 'ℹ'}
            </span>
            {t.message}
          </div>
        ))}
      </div>

      {/* ── Eval Modal ──────────────────────────────────────────────────────── */}
      {evalReport && (
        <EvalModal report={evalReport} onClose={() => setEvalReport(null)} />
      )}

      {/* ── Cache Modal ─────────────────────────────────────────────────────── */}
      {showCacheModal && (
        <CacheModal onClose={() => setShowCacheModal(false)} />
      )}
    </div>
  );
}
