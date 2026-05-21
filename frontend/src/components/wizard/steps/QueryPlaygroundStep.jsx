import { useEffect, useRef, useState, useCallback } from 'react';
import { marked } from 'marked';
import './QueryPlaygroundStep.css';

marked.setOptions({ breaks: true, gfm: true });

const getConfidenceLevel = (results) => {
    if (!results || results.length === 0) return null;
    const topScores = results.slice(0, 5).map(r => r.score ?? 0);
    const avg = topScores.reduce((a, b) => a + b, 0) / topScores.length;
    if (avg >= 0.72) return { level: 'High', color: '#047857', bg: '#ecfdf5', border: '#a7f3d0', icon: '●●●' };
    if (avg >= 0.50) return { level: 'Medium', color: '#b45309', bg: '#fffbeb', border: '#fde68a', icon: '●●○' };
    return { level: 'Low', color: '#b91c1c', bg: '#fef2f2', border: '#fecaca', icon: '●○○' };
};

const scoreColor = (score) => {
    if (score >= 0.75) return { bg: '#ecfdf5', text: '#047857', border: '#a7f3d0' };
    if (score >= 0.5)  return { bg: '#fffbeb', text: '#b45309', border: '#fde68a' };
    return                    { bg: '#fef2f2', text: '#b91c1c', border: '#fecaca' };
};

const IconCopy = () => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
    </svg>
);
const IconThumbUp = () => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3H14z"/>
        <path d="M7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/>
    </svg>
);
const IconThumbDown = () => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3H10z"/>
        <path d="M17 2h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"/>
    </svg>
);

const QueryPlaygroundStep = ({
    query, topK, results, loading, summaryLoading, summary, thinkContent = '',
    prompt, maxTokens, modelName, enableSummary = false,
    onQueryChange, onTopKChange, onSearch, onPromptChange,
    onMaxTokensChange, onModelNameChange, onEnableSummaryChange,
    batchesCompleted, totalBatches, activeConfigName, onSubmitFeedback,
    searchError,
}) => {
    const [searchHistory, setSearchHistory] = useState([]);
    const [showHistory, setShowHistory]     = useState(false);
    const [feedbackMap, setFeedbackMap]     = useState({});
    const [copiedMap, setCopiedMap]         = useState({});
    const lastBatchesRef = useRef(0);
    const hasSearchedRef = useRef(false);
    const historyRef     = useRef(null);

    useEffect(() => { if (results && results.length > 0) hasSearchedRef.current = true; }, [results]);

    useEffect(() => {
        if (batchesCompleted > lastBatchesRef.current && batchesCompleted > 0 && hasSearchedRef.current) {
            lastBatchesRef.current = batchesCompleted;
            if (results && results.length > 0 && query && query.trim()) {
                const id = setTimeout(() => onSearch && onSearch(), 1000);
                return () => clearTimeout(id);
            }
        } else if (batchesCompleted > 0) { lastBatchesRef.current = batchesCompleted; }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [batchesCompleted, totalBatches]);

    useEffect(() => {
        const h = (e) => { if (historyRef.current && !historyRef.current.contains(e.target)) setShowHistory(false); };
        document.addEventListener('mousedown', h);
        return () => document.removeEventListener('mousedown', h);
    }, []);

    const handleSearch = useCallback(() => {
        if (!query || !query.trim()) return;
        onSearch && onSearch();
        setSearchHistory(prev => {
            const filtered = prev.filter(h => h.query !== query.trim());
            return [{ query: query.trim(), timestamp: Date.now(), resultCount: 0 }, ...filtered].slice(0, 30);
        });
        setFeedbackMap({});
        setCopiedMap({});
    }, [query, onSearch]);

    useEffect(() => {
        if (results && results.length > 0 && query) {
            setSearchHistory(prev => prev.map((h, i) => i === 0 ? { ...h, resultCount: results.length } : h));
        }
    }, [results]);

    const handleKeyDown = (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSearch(); } };

    const handleVote = (result, idx, vote) => {
        const key = result.id || `r-${idx}`;
        if (feedbackMap[key] === vote) return;
        setFeedbackMap(prev => ({ ...prev, [key]: vote }));
        onSubmitFeedback && onSubmitFeedback({ query, resultId: key, contentSnippet: result.content?.slice(0, 200), vote, configName: activeConfigName });
    };

    const handleCopy = (result, idx) => {
        const key = result.id || `r-${idx}`;
        navigator.clipboard.writeText(result.content || '').then(() => {
            setCopiedMap(prev => ({ ...prev, [key]: true }));
            setTimeout(() => setCopiedMap(prev => ({ ...prev, [key]: false })), 2000);
        });
    };

    const confidence = getConfidenceLevel(results);
    const fmtTime = (ts) => new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    return (
        <div className={`qp-layout${enableSummary ? '' : ' qp-layout--full'}`}>

            <div className="qp-mode-toggle-row">
                <span className="qp-mode-label">AI Summarisation</span>
                <label className="qp-toggle-switch">
                    <input type="checkbox" checked={enableSummary} onChange={(e) => onEnableSummaryChange?.(e.target.checked)} />
                    <span className="qp-toggle-track"><span className="qp-toggle-thumb" /></span>
                </label>
                <span className="qp-mode-hint">{enableSummary ? 'Search + Summarise' : 'Search Only'}</span>
            </div>

            <div className="qp-left">
                <div className="qp-search-card">
                    <div className="qp-search-top-row">
                        <textarea
                            className="qp-textarea"
                            placeholder="Enter your semantic search query…"
                            value={query}
                            onChange={(e) => onQueryChange(e.target.value)}
                            onKeyDown={handleKeyDown}
                            disabled={loading}
                            rows={3}
                        />
                        {searchHistory.length > 0 && (
                            <div className="qp-history-wrapper" ref={historyRef}>
                                <button className="qp-history-btn" onClick={() => setShowHistory(s => !s)} title="Search history">
                                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                        <polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 .49-4.34"/>
                                    </svg>
                                    <span>{searchHistory.length}</span>
                                </button>
                                {showHistory && (
                                    <div className="qp-history-dropdown">
                                        <div className="qp-history-header">
                                            <span>Search History</span>
                                            <button className="qp-history-clear" onClick={() => { setSearchHistory([]); setShowHistory(false); }}>Clear</button>
                                        </div>
                                        {searchHistory.map((h, i) => (
                                            <button key={i} className="qp-history-item" onClick={() => { onQueryChange && onQueryChange(h.query); setShowHistory(false); }}>
                                                <span className="qp-history-query">{h.query}</span>
                                                <span className="qp-history-meta">
                                                    {fmtTime(h.timestamp)}
                                                    {h.resultCount > 0 && <span className="qp-history-count">{h.resultCount} results</span>}
                                                </span>
                                            </button>
                                        ))}
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                    <div className="qp-search-controls">
                        <div className="qp-topk">
                            <label className="qp-topk-label">Top K</label>
                            <input type="number" min="1" max="50" value={topK}
                                onChange={(e) => { const v = parseInt(e.target.value); if (v > 0 && v <= 50) onTopKChange(v); }}
                                className="qp-topk-input" disabled={loading} />
                        </div>
                        <button className="btn-primary qp-search-btn" onClick={handleSearch} disabled={!query.trim() || loading}>
                            {loading ? (<><span className="loading-spinner" />Searching…</>) : (
                                <><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35" strokeLinecap="round"/></svg>Search</>
                            )}
                        </button>
                    </div>
                </div>

                <div className="qp-results-section">
                    <div className="qp-results-heading">
                        <span>Results</span>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                            {confidence && !loading && (
                                <span className="qp-confidence-badge" style={{ background: confidence.bg, color: confidence.color, borderColor: confidence.border }} title="Confidence based on similarity scores">
                                    <span className="qp-conf-icon">{confidence.icon}</span>{confidence.level} Confidence
                                </span>
                            )}
                            {totalBatches > 0 && (
                                <span className={`qp-batch-badge ${batchesCompleted < totalBatches ? 'partial' : 'done'}`}>
                                    {batchesCompleted < totalBatches ? `⏳ ${batchesCompleted}/${totalBatches} batches` : `✅ ${batchesCompleted}/${totalBatches} complete`}
                                </span>
                            )}
                            {results && results.length > 0 && <span className="qp-count-badge">{results.length} found</span>}
                        </div>
                    </div>

                    {loading ? (
                        <div className="qp-loading-state"><div className="qp-loading-spinner" /><p>Searching semantic space…</p></div>
                    ) : searchError ? (
                        <div className="qp-search-error">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
                            </svg>
                            <div>
                                <strong>Search failed</strong>
                                <p>{searchError}</p>
                            </div>
                        </div>
                    ) : results && results.length > 0 ? (
                        <div className="qp-results-list">
                            {results.map((result, idx) => {
                                const score = result.score ?? 0;
                                const c = scoreColor(score);
                                const key = result.id || `r-${idx}`;
                                const voted = feedbackMap[key];
                                const copied = copiedMap[key];
                                return (
                                    <div key={key} className="qp-result-card">
                                        <div className="qp-result-header">
                                            <div className="qp-result-rank">#{idx + 1}</div>
                                            <span className="qp-result-label">Document Fragment</span>
                                            <span className="qp-score-badge" style={{ background: c.bg, color: c.text, borderColor: c.border }}>
                                                {Math.round(score * 100)}% match
                                            </span>
                                        </div>
                                        <div className="qp-result-text" dangerouslySetInnerHTML={{ __html: result.content }} />
                                        {result.metadata?.source && (
                                            <div className="qp-result-meta">
                                                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                                    <path d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" strokeLinecap="round" strokeLinejoin="round"/>
                                                </svg>
                                                <span>{result.metadata.source}</span>
                                            </div>
                                        )}
                                        {/* ── Result feedback actions ── */}
                                        <div className="qp-result-actions">
                                            <button className={`qp-action-btn qp-action-copy${copied ? ' active' : ''}`} onClick={() => handleCopy(result, idx)} title="Copy text">
                                                <IconCopy /><span>{copied ? 'Copied!' : 'Copy'}</span>
                                            </button>
                                            <button className={`qp-action-btn qp-action-up${voted === 'up' ? ' active' : ''}`} onClick={() => handleVote(result, idx, 'up')} title="Relevant">
                                                <IconThumbUp />
                                            </button>
                                            <button className={`qp-action-btn qp-action-down${voted === 'down' ? ' active' : ''}`} onClick={() => handleVote(result, idx, 'down')} title="Not relevant">
                                                <IconThumbDown />
                                            </button>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    ) : (
                        <div className="qp-empty-state">
                            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                                <circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35" strokeLinecap="round"/>
                            </svg>
                            <p>Enter a query above to search your documents.</p>
                        </div>
                    )}
                </div>
            </div>

            {enableSummary && <div className="qp-right">
                <div className="qp-ai-card">
                    <div className="qp-ai-header">
                        <div className="qp-ai-icon">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                                <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" strokeLinecap="round" strokeLinejoin="round"/>
                            </svg>
                        </div>
                        <h3 className="qp-ai-title">AI Summarisation</h3>
                        {confidence && !loading && results && results.length > 0 && (
                            <span className="qp-confidence-badge qp-conf-sm" style={{ background: confidence.bg, color: confidence.color, borderColor: confidence.border, marginLeft: 'auto' }} title="Answer confidence">
                                {confidence.icon} {confidence.level}
                            </span>
                        )}
                    </div>
                    <div className="qp-ai-note">Summary is always generated from the top 5 results, regardless of Top K.</div>
                    <div className="qp-field">
                        <label className="qp-label">Summarisation Model</label>
                        <select value={modelName} onChange={(e) => onModelNameChange?.(e.target.value)} className="qp-select" disabled={loading}>
                            <option value="qwen3-0.6b">Qwen 0.6B (Local GPU/CPU)</option>
                        </select>
                    </div>
                    <div className="qp-field">
                        <label className="qp-label">Max Output Tokens</label>
                        <input type="number" min="10" max="2000" value={maxTokens} onChange={(e) => onMaxTokensChange?.(parseInt(e.target.value) || 150)} className="qp-select" disabled={loading} />
                    </div>
                    <div className="qp-field">
                        <label className="qp-label">System Prompt / Instructions</label>
                        <textarea value={prompt} onChange={(e) => onPromptChange?.(e.target.value)} className="qp-prompt-textarea" placeholder="e.g. Summarise these results in 3–4 lines…" disabled={loading} rows={5} />
                    </div>
                    {(thinkContent || (summaryLoading && !summary)) && (
                        <details className="qp-think-details" open>
                            <summary className="qp-think-summary">
                                <span className="qp-think-label">
                                    {summaryLoading && !summary ? <><span className="qp-think-dot" />Model thinking…</> : 'Model thinking'}
                                </span>
                            </summary>
                            <pre className="qp-think-content">{thinkContent || '…'}</pre>
                        </details>
                    )}
                    {summary ? (
                        <div className="qp-summary-output">
                            <div className="qp-summary-output-label">Generated Answer</div>
                            <div className="qp-summary-text qp-summary-markdown" dangerouslySetInnerHTML={{ __html: marked.parse(summary) }} />
                        </div>
                    ) : summaryLoading ? (
                        <div className="qp-summary-loading"><div className="qp-summary-spinner" /><span>Generating answer…</span></div>
                    ) : results && results.length > 0 && !loading ? (
                        <div className="qp-summary-placeholder">Summary will appear here after the model finishes…</div>
                    ) : null}
                </div>
            </div>}
        </div>
    );
};

export default QueryPlaygroundStep;
