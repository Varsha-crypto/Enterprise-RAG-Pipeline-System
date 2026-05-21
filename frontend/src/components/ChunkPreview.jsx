import { useState } from 'react';
import { fetchChunkPreview } from '../services/api';

const ChunkPreview = ({ sampleText = '', chunkSize = 500, chunkOverlap = 50, strategy = 'fixed_size' }) => {
    const [open, setOpen]         = useState(false);
    const [chunks, setChunks]     = useState([]);
    const [total, setTotal]       = useState(0);
    const [loading, setLoading]   = useState(false);
    const [error, setError]       = useState('');

    const handlePreview = async () => {
        if (!sampleText.trim()) {
            setError('No source text available for preview.');
            setOpen(true);
            return;
        }
        setLoading(true);
        setError('');
        try {
            const data = await fetchChunkPreview({
                text: sampleText.slice(0, 8000), // cap for preview
                chunkSize,
                chunkOverlap,
                strategy,
            });
            setChunks(data.chunks || []);
            setTotal(data.total_chunks || 0);
            setOpen(true);
        } catch (e) {
            setError('Chunk preview unavailable: ' + e.message);
            setOpen(true);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div>
            <button
                type="button"
                className="qp-history-btn"
                style={{ fontSize: '0.78rem', padding: '0.4rem 0.85rem', gap: '0.4rem' }}
                onClick={open ? () => setOpen(false) : handlePreview}
                disabled={loading}
            >
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/>
                </svg>
                {loading ? 'Loading…' : open ? 'Hide Chunk Preview' : 'Preview Chunks'}
            </button>

            {open && (
                <div className="chunk-preview-panel" style={{ marginTop: '0.75rem' }}>
                    <div className="chunk-preview-header">
                        <span>Chunk Preview — {strategy} · size {chunkSize} · overlap {chunkOverlap}</span>
                        <span style={{ color: '#94a3b8', fontWeight: 400 }}>
                            Showing {chunks.length} of {total} chunk{total !== 1 ? 's' : ''}
                        </span>
                    </div>
                    <div className="chunk-preview-list">
                        {error && <div style={{ color: '#ef4444', fontSize: '0.8rem' }}>{error}</div>}
                        {chunks.length === 0 && !error && <div style={{ color: '#94a3b8', fontSize: '0.8rem' }}>No chunks generated.</div>}
                        {chunks.map((ch) => (
                            <div key={ch.index} className="chunk-preview-item">
                                {ch.text}
                                <div className="chunk-preview-item-meta">Chunk #{ch.index + 1} · {ch.length} chars</div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
};

export default ChunkPreview;
