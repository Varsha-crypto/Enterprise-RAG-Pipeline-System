import { useState, useEffect, useRef, useCallback } from 'react';
import { fetchLogs } from '../services/api';

const LogViewer = ({ autoRefresh = false }) => {
    const [logs, setLogs]         = useState([]);
    const [open, setOpen]         = useState(false);
    const [loading, setLoading]   = useState(false);
    const [lines, setLines]       = useState(200);
    const bodyRef                 = useRef(null);
    const intervalRef             = useRef(null);

    const load = useCallback(async () => {
        setLoading(true);
        try {
            const data = await fetchLogs(lines);
            setLogs(data.logs || []);
        } catch {
            setLogs(['[Error] Could not reach the log API — is the backend running?']);
        } finally {
            setLoading(false);
        }
    }, [lines]);

    useEffect(() => {
        if (!open) return;
        load();
        if (autoRefresh) {
            intervalRef.current = setInterval(load, 4000);
        }
        return () => clearInterval(intervalRef.current);
    }, [open, load, autoRefresh]);

    // Auto-scroll to bottom when new logs arrive
    useEffect(() => {
        if (bodyRef.current && open) {
            bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
        }
    }, [logs, open]);

    const classifyLine = (line) => {
        const l = line.toLowerCase();
        if (l.includes('error') || l.includes('exception') || l.includes('traceback')) return 'log-line-error';
        if (l.includes('warn')) return 'log-line-warn';
        if (l.includes('info')) return 'log-line-info';
        return '';
    };

    return (
        <div className="log-panel">
            <div className="log-panel-header" onClick={() => setOpen(o => !o)}>
                <span className="log-panel-title">
                    <span className="log-dot" style={{ background: open ? '#22c55e' : '#94a3b8' }} />
                    Backend Log Viewer
                    {loading && <span style={{ color: '#94a3b8', fontWeight: 400, fontSize: '0.7rem' }}> loading…</span>}
                </span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                    {open && (
                        <>
                            <select
                                value={lines}
                                onChange={(e) => setLines(Number(e.target.value))}
                                onClick={(e) => e.stopPropagation()}
                                style={{ fontSize: '0.72rem', padding: '0.2rem 0.4rem', borderRadius: '6px', border: '1px solid #334155', background: '#1e293b', color: '#94a3b8' }}
                            >
                                <option value={50}>Last 50</option>
                                <option value={200}>Last 200</option>
                                <option value={500}>Last 500</option>
                            </select>
                            <button
                                onClick={(e) => { e.stopPropagation(); load(); }}
                                style={{ fontSize: '0.72rem', padding: '0.2rem 0.6rem', borderRadius: '6px', border: '1px solid #334155', background: '#1e293b', color: '#94a3b8', cursor: 'pointer' }}
                            >↻ Refresh</button>
                        </>
                    )}
                    <span style={{ color: '#475569', fontSize: '0.85rem' }}>{open ? '▲' : '▼'}</span>
                </div>
            </div>
            {open && (
                <div className="log-panel-body" ref={bodyRef}>
                    {logs.length === 0 && !loading && <span style={{ color: '#475569' }}>No log entries yet.</span>}
                    {logs.map((line, i) => (
                        <div key={i} className={`log-line ${classifyLine(line)}`}>{line}</div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default LogViewer;
