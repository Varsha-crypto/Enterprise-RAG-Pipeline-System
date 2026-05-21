import React from 'react';
import ChunkPreview from '../../ChunkPreview';
import './PipelineDesignStep.css';

/**
 * PipelineDesignStep Component
 * 
 * Screen 2: Display backend-recommended AI pipeline configuration with override options.
 * 
 * @param {Object} props
 * @param {Object} props.recommendations - { key: { value, reason } }
 * @param {Object} props.overrides - { key: { enabled, value } }
 * @param {Object} props.availableOptions - { key: [options] }
 * @param {function} props.onOverrideToggle - (key) => void
 * @param {function} props.onOverrideChange - (key, value) => void
 * @param {function} props.onContinue - () => void
 * @param {function} props.onBack - () => void
 */
const PipelineDesignStep = ({
    recommendations,
    overrides,
    availableOptions,
    onOverrideToggle,
    onOverrideChange,
    onExecute,
    onBack,
    incrementalMode = false,
    onIncrementalModeChange,
    sampleText = '',
    chunkSize = 500,
    chunkOverlap = 50,
}) => {
    const categories = [
        { key: 'chunkingStrategy', title: 'Chunking Strategy' },
        { key: 'embeddingModel',   title: 'Embedding Model' },
        { key: 'retrievalMethod',  title: 'Index Type' },
    ];

    return (
        <div className="pipeline-design-step">

            <div className="pipeline-cards-grid">
                {categories.map(({ key, title }) => {
                    const rec = recommendations[key] || { value: 'Loading...', reason: '' };
                    const override = overrides[key] || { enabled: false, value: '' };
                    const options = availableOptions[key] || [];

                    return (
                        <div key={key} className="pipeline-card">
                            <div className="card-header">
                                <h3 className="card-title">{title}</h3>
                                {!override.enabled && (
                                    <span className="card-badge">Recommended</span>
                                )}
                                {override.enabled && (
                                    <span className="card-badge" style={{ backgroundColor: '#fff7ed', color: '#c2410c' }}>Custom</span>
                                )}
                            </div>

                            <div className="recommendation-section">
                                <span className="rec-label">Recommendation</span>
                                <div className="rec-value">
                                    {rec.value}
                                </div>
                            </div>

                            <div className="override-section">
                                <label className="override-toggle">
                                    <input
                                        type="checkbox"
                                        className="hidden"
                                        style={{ display: 'none' }}
                                        checked={override.enabled}
                                        onChange={() => onOverrideToggle(key)}
                                    />
                                    <div className="checkbox-visual"></div>
                                    <span className="override-label">Override Recommendation</span>
                                </label>

                                <div className={`override-dropdown-container ${override.enabled ? 'open' : ''}`}>
                                    <select
                                        className="override-select"
                                        value={override.value || rec.value}
                                        onChange={(e) => onOverrideChange(key, e.target.value)}
                                        disabled={!override.enabled}
                                    >
                                        {!override.value && <option value="" disabled>Select option</option>}
                                        {options.map(opt => (
                                            <option key={opt} value={opt}>
                                                {opt}
                                            </option>
                                        ))}
                                        {key === 'embeddingModel' && <option value="other">Other (Manual Download)</option>}
                                    </select>

                                    {key === 'embeddingModel' && override.enabled && (
                                        <div className="custom-model-inputs" style={{ marginTop: '1rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                                            {override.value === 'other' && (
                                                <div className="input-group">
                                                    <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Model Name (HuggingFace ID)</label>
                                                    <input
                                                        type="text"
                                                        className="override-select"
                                                        placeholder="e.g. sentence-transformers/all-MiniLM-L6-v2"
                                                        onChange={(e) => onOverrideChange('embeddingModelManual', e.target.value)}
                                                    />
                                                </div>
                                            )}
                                            <div className="input-group">
                                                <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Embedding Dimensions</label>
                                                <input
                                                    type="number"
                                                    className="override-select"
                                                    placeholder="e.g. 768"
                                                    min="64"
                                                    max="4096"
                                                    value={overrides.embeddingDimensions?.value || ''}
                                                    onChange={(e) => {
                                                        const v = parseInt(e.target.value);
                                                        if (e.target.value === '' || (v >= 64 && v <= 4096)) {
                                                            onOverrideChange('embeddingDimensions', e.target.value);
                                                        }
                                                    }}
                                                />
                                            </div>
                                            <div className="input-group">
                                                <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Hugging Face Token (Optional)</label>
                                                <input
                                                    type="password"
                                                    className="override-select"
                                                    placeholder="hf_..."
                                                    value={overrides.huggingfaceToken?.value || ''}
                                                    onChange={(e) => onOverrideChange('huggingfaceToken', e.target.value)}
                                                />
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    );
                })}
            </div>

            <div className="pipeline-action-bar" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <button
                    onClick={onBack}
                    className="px-4 py-2 rounded border border-gray-300 text-gray-700 bg-white hover:bg-gray-50 font-medium"
                >
                    Back
                </button>

                <div className="execution-controls" style={{ display: 'flex', alignItems: 'center', gap: '2rem' }}>
                    <div className="batch-toggle-container" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                        <span className="toggle-label" style={{ fontSize: '0.9rem', fontWeight: 500, color: 'var(--text-secondary)' }}>Batch Processing</span>
                        <div className="toggle-tooltip" style={{ position: 'relative', display: 'inline-block', cursor: 'help' }}>
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" style={{ opacity: 0.6 }}>
                                <circle cx="12" cy="12" r="10" />
                                <path d="M9.09 9a3 3 0 015.83 1c0 2-3 3-3 3" />
                                <line x1="12" y1="17" x2="12.01" y2="17" />
                            </svg>
                        </div>
                        <label className="switch">
                            <input
                                type="checkbox"
                                checked={incrementalMode}
                                onChange={(e) => onIncrementalModeChange(e.target.checked)}
                            />
                            <span className="slider round"></span>
                        </label>
                    </div>
                    <button
                        onClick={onExecute}
                        className="btn-primary"
                    >
                        Execute Pipeline
                    </button>
                    <div style={{ marginTop: '0.75rem' }}>
                        <ChunkPreview
                            sampleText={sampleText}
                            chunkSize={chunkSize}
                            chunkOverlap={chunkOverlap}
                            strategy={overrides?.chunkingStrategy?.enabled ? overrides.chunkingStrategy.value : (recommendations?.chunkingStrategy?.value || 'fixed_size')}
                        />
                    </div>
                </div>
            </div>
        </div>
    );
};

export default PipelineDesignStep;
