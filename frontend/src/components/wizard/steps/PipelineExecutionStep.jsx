import React, { useEffect } from 'react';
import './PipelineExecutionStep.css';
import QueryPlaygroundStep from './QueryPlaygroundStep';
import BatchProgressBar from './BatchProgressBar';

const PipelineExecutionStep = ({
    pipelineStatus,
    errorMessage,
    progress,
    metrics,
    onExecute,
    onExport,
    onCancelPipeline,
    onBack,
    canViewResults = false,
    pipelineMode,
    batchesCompleted = 0,
    totalBatches = 0,
    stage,
    // Query Playground Props
    query,
    topK,
    results,
    loading,
    summaryLoading,
    summary,
    thinkContent = '',
    prompt,
    maxTokens,
    modelName,
    activeConfigName,
    onQueryChange,
    onTopKChange,
    onSearch,
    onRestart,
    onPromptChange,
    onMaxTokensChange,
    onModelNameChange,
    enableSummary = false,
    onEnableSummaryChange,
    incrementalMode = false,
    onIncrementalModeChange,
    isImported = false,
    executionTime = null,
    nodeTimings = {},
    device = 'cpu',
    onSubmitFeedback,
    searchError,
}) => {
    const isRunning = pipelineStatus === 'running' || pipelineStatus === 'uploading';
    const isCompleted = pipelineStatus === 'completed';
    const playgroundRef = React.useRef(null);

    const scrollToPlayground = () => {
        playgroundRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    };

    const getNodeState = (nodeIndex) => {
        if (pipelineStatus === 'idle') return '';
        const stepSize = 100 / 6;
        const threshold = (nodeIndex + 1) * stepSize;
        const prevThreshold = nodeIndex * stepSize;
        if (isCompleted) return 'completed';
        if (progress >= threshold - 0.5) return 'completed';
        if (progress > prevThreshold && progress < threshold) return 'running';
        return '';
    };

    const nodes = [
        { label: 'Connect DB', icon: 'M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z' },
        { label: 'Data Ingestion', icon: 'M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12' },
        { label: 'Chunking', icon: 'M4 6h16M4 12h16M4 18h7' },
        { label: 'Embedding', icon: 'M13 10V3L4 14h7v7l9-11h-7z' },
        { label: 'Vector Store', icon: 'M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4' },
        { label: 'Retrieval Ready', icon: 'M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z' }
    ];

    const getStatusText = () => {
        switch (pipelineStatus) {
            case 'idle': return 'Ready to Execute';
            case 'uploading': return 'Uploading File...';
            case 'running': return 'Processing Pipeline...';
            case 'completed': return 'Pipeline Execution Successful';
            case 'cleaning': return 'Cleaning Database...';
            case 'cancelled_and_cleaned': return 'Cancelled and Cleaned';
            case 'cancelled': return 'Pipeline Cancelled';
            case 'failed':
            case 'error': return 'Execution Failed';
            default: return 'Unknown Status';
        }
    };

    const playgroundReady = isCompleted || (incrementalMode && batchesCompleted > 0);
    const resultsToShow = results || [];

    const renderPlaygroundCard = () => {
        if (playgroundReady) {
            return (
                <div className="playground-section">
                    <QueryPlaygroundStep
                        query={query}
                        topK={topK}
                        results={resultsToShow}
                        loading={loading}
                        summaryLoading={summaryLoading}
                        summary={summary}
                        thinkContent={thinkContent}
                        prompt={prompt}
                        maxTokens={maxTokens}
                        modelName={modelName}
                        onQueryChange={onQueryChange}
                        onTopKChange={onTopKChange}
                        onPromptChange={onPromptChange}
                        onMaxTokensChange={onMaxTokensChange}
                        onModelNameChange={onModelNameChange}
                        enableSummary={enableSummary}
                        onEnableSummaryChange={onEnableSummaryChange}
                        onSearch={onSearch}
                        batchesCompleted={batchesCompleted}
                        totalBatches={totalBatches}
                        activeConfigName={activeConfigName}
                        onSubmitFeedback={onSubmitFeedback}
                        searchError={searchError}
                    />
                </div>
            );
        }
        return null;
    };

    return (
        <div className="pipeline-execution-step">
            {(!incrementalMode && pipelineMode !== 'batch' && !isImported) ? (
                <div className="pipeline-diagram">
                    {nodes.map((node, index) => (
                        <React.Fragment key={index}>
                            <div className={`pipeline-node ${getNodeState(index)}`}>
                                {nodeTimings[index] && (
                                    <div className="node-step-timing" style={{
                                        position: 'absolute',
                                        top: '-25px',
                                        fontSize: '0.7rem',
                                        fontWeight: '700',
                                        color: getNodeState(index) === 'completed' ? '#10b981' : '#6b7280',
                                        backgroundColor: 'rgba(255,255,255,0.9)',
                                        padding: '2px 6px',
                                        borderRadius: '4px',
                                        boxShadow: '0 1px 2px rgba(0,0,0,0.05)',
                                        whiteSpace: 'nowrap',
                                        animation: 'fadeIn 0.3s ease-out'
                                    }}>
                                        {nodeTimings[index]}
                                    </div>
                                )}
                                <div className="node-circle">
                                    {getNodeState(index) === 'completed' ? (
                                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                                            <path d="M20 6L9 17L4 12" strokeLinecap="round" strokeLinejoin="round" />
                                        </svg>
                                    ) : (
                                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                            <path d={node.icon} strokeLinecap="round" strokeLinejoin="round" />
                                        </svg>
                                    )}
                                </div>
                                <span className="node-label">{node.label}</span>
                            </div>
                            {index < nodes.length - 1 && (
                                <div className={`pipeline-connector ${getNodeState(index) === 'completed' ? 'active' : ''}`}></div>
                            )}
                        </React.Fragment>
                    ))}
                </div>
            ) : null}

            {!isImported && (
                <div className="status-panel">
                    <div className="status-header">
                        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                            <span className="status-text">{getStatusText()}</span>
                            {isRunning && (
                                <span style={{
                                    fontSize: '0.7rem',
                                    fontWeight: '700',
                                    textTransform: 'uppercase',
                                    padding: '2px 8px',
                                    borderRadius: '4px',
                                    backgroundColor: device?.toLowerCase() === 'cuda' ? '#eef2ff' : '#f8fafc',
                                    color: device?.toLowerCase() === 'cuda' ? '#4f46e5' : '#64748b',
                                    border: `1px solid ${device?.toLowerCase() === 'cuda' ? '#c7d2fe' : '#e2e8f0'}`,
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '4px'
                                }}>
                                    <span style={{ fontSize: '10px' }}>{device?.toLowerCase() === 'cuda' ? '⚡' : '💻'}</span>
                                    {device?.toLowerCase() === 'cuda' ? 'GPU Accelerated' : 'CPU Processing'}
                                </span>
                            )}
                        </div>
                        <span className={`status-badge ${pipelineStatus}`}>
                            {pipelineStatus.charAt(0).toUpperCase() + pipelineStatus.slice(1)}
                        </span>
                    </div>

                    {pipelineStatus !== 'idle' && !incrementalMode && (
                        <div style={{ marginTop: '1rem', marginBottom: '0.5rem' }}>
                            <div style={{
                                display: 'flex',
                                justifyContent: 'space-between',
                                fontSize: '0.8rem',
                                color: '#6b7280',
                                marginBottom: '0.4rem',
                                fontWeight: 500
                            }}>
                                <span>{stage || (isRunning ? 'Processing...' : isCompleted ? 'Complete' : 'Ready')}</span>
                                <span>{Math.round(progress || 0)}%</span>
                            </div>
                            <div style={{
                                height: '8px',
                                background: '#e5e7eb',
                                borderRadius: '999px',
                                overflow: 'hidden'
                            }}>
                                <div style={{
                                    height: '100%',
                                    width: `${Math.min(Math.max(progress || 0, 0), 100)}%`,
                                    background: isCompleted
                                        ? 'linear-gradient(90deg, #10b981, #34d399)'
                                        : pipelineStatus === 'error'
                                            ? '#ef4444'
                                            : 'linear-gradient(100deg, #4f46e5, #818cf8)',
                                    borderRadius: '999px',
                                    transition: 'width 0.5s cubic-bezier(0.4, 0, 0.2, 1)'
                                }} />
                            </div>
                        </div>
                    )}

                    {(pipelineStatus === 'error' || pipelineStatus === 'failed') && errorMessage && (
                        <div style={{
                            marginTop: '1rem',
                            padding: '0.875rem 1rem',
                            background: '#fef2f2',
                            border: '1px solid #fecaca',
                            borderRadius: '0.5rem',
                            color: '#991b1b',
                            fontSize: '0.85rem',
                            display: 'flex',
                            gap: '0.5rem',
                            alignItems: 'flex-start'
                        }}>
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ flexShrink: 0, marginTop: '1px' }}>
                                <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
                            </svg>
                            <div>
                                <strong>Pipeline failed</strong>
                                <div style={{ marginTop: '2px', color: '#b91c1c', wordBreak: 'break-word' }}>{errorMessage}</div>
                            </div>
                        </div>
                    )}

                    {(incrementalMode || pipelineMode === 'batch') && (
                        <BatchProgressBar
                            progress={progress}
                            completedBatches={batchesCompleted}
                            totalBatches={totalBatches}
                            stage={stage}
                            pipelineStatus={pipelineStatus}
                        />
                    )}

                    <div className={`execution-metrics ${incrementalMode ? 'batch-mode' : 'single-mode'}`}>
                        <div className="metric-card">
                            <span className="metric-value">{metrics?.chunks || 0}</span>
                            <span className="metric-label">Chunks Processed</span>
                        </div>
                        <div className="metric-card">
                            <span className="metric-value">{metrics?.embeddings || 0}</span>
                            <span className="metric-label">Embeddings Generated</span>
                        </div>
                        {incrementalMode && (
                            <div className="metric-card">
                                <span className="metric-value">{batchesCompleted || 0}{totalBatches ? ` / ${totalBatches}` : ''}</span>
                                <span className="metric-label">Batches Completed</span>
                            </div>
                        )}
                        {isCompleted && executionTime && (
                            <div className="metric-card">
                                <span className="metric-value">{executionTime}</span>
                                <span className="metric-label">Total Execution Time</span>
                            </div>
                        )}
                    </div>

                    {(incrementalMode && batchesCompleted > 0 && isRunning) && (
                        <div style={{
                            marginTop: '1.25rem',
                            padding: '1rem',
                            background: 'linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%)',
                            border: '1px solid #bfdbfe',
                            borderRadius: '0.75rem',
                            fontSize: '0.95rem',
                            color: '#1e40af',
                            boxShadow: '0 1px 2px rgba(0,0,0,0.05)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            gap: '0.75rem'
                        }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                                <span style={{ fontSize: '1.2rem' }}>⚡</span>
                                <div>
                                    <strong>Real-time Results Available!</strong><br />
                                    Processing batch {batchesCompleted} of {totalBatches}.
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            )}

            {(playgroundReady) && (
                <div className="results-navigation-header" style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '1rem 2rem',
                    margin: '0 -2rem 2rem -2rem',
                    borderBottom: '1px solid var(--border-color)',
                    position: 'sticky',
                    top: '-2.5rem',
                    backgroundColor: 'rgba(255, 255, 255, 0.98)',
                    backdropFilter: 'blur(10px)',
                    zIndex: 40,
                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.05)'
                }}>
                    <div style={{ display: 'flex', gap: '1rem' }}>
                        {playgroundReady && (
                            <button onClick={scrollToPlayground} className="btn-secondary" style={{ padding: '0.5rem 1.25rem', fontSize: '0.9rem' }}>
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" style={{ marginRight: '8px' }}>
                                    <path d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                                </svg>
                                Query Playground
                            </button>
                        )}
                    </div>
                {isCompleted && (
                    <div style={{ display: 'flex', gap: '1rem' }}>
                        <button onClick={onRestart} className="btn-secondary" style={{ padding: '0.5rem 1.25rem', fontSize: '0.9rem' }}>
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" style={{ marginRight: '8px' }}>
                                <path d="M3 12a9 9 0 1 1 9 9 9 9 0 0 1-9-9zM12 8v8M8 12h8" strokeLinecap="round" strokeLinejoin="round" />
                                <path d="M9 12l3-3 3 3" strokeLinecap="round" strokeLinejoin="round" />
                            </svg>
                            Start New Pipeline
                        </button>
                        <button onClick={onExport} className="btn-primary" style={{ padding: '0.5rem 1.5rem', fontSize: '0.9rem', boxShadow: 'var(--shadow-sm)' }}>
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" style={{ marginRight: '8px' }}>
                                <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3" />
                            </svg>
                            Export Configuration
                        </button>
                    </div>
                )}
            </div>
            )}

            <div className="pipeline-results-area">
                <div ref={playgroundRef} style={{ scrollMarginTop: '100px' }}>
                    {renderPlaygroundCard()}
                </div>
            </div>

            <div className="execution-action-bar">
                {!isRunning && (
                    <button
                        onClick={onBack}
                        className="btn-secondary"
                        style={{ padding: '0.5rem 1.5rem', fontSize: '0.9rem' }}
                    >
                        Back
                    </button>
                )}

                {!isCompleted && !isImported && isRunning && (
                    <button
                        onClick={() => {
                            if (window.confirm("Are you sure you want to cancel the pipeline? All ingested data for this run will be deleted.")) {
                                onCancelPipeline && onCancelPipeline();
                            }
                        }}
                        style={{
                            padding: '0.5rem 1.5rem',
                            fontSize: '0.9rem',
                            backgroundColor: '#ef4444',
                            color: 'white',
                            border: '1px solid #dc2626',
                            borderRadius: '0.375rem',
                            cursor: 'pointer',
                            fontWeight: '500',
                            transition: 'background-color 0.2s',
                            boxShadow: '0 1px 2px rgba(0,0,0,0.05)'
                        }}
                        onMouseOver={(e) => e.target.style.backgroundColor = '#dc2626'}
                        onMouseOut={(e) => e.target.style.backgroundColor = '#ef4444'}
                    >
                        Cancel Pipeline
                    </button>
                )}

                {!isCompleted && !isImported && !isRunning && (
                    <button
                        onClick={onExecute}
                        className="btn-primary"
                        style={{ padding: '0.5rem 1.5rem', fontSize: '0.9rem' }}
                    >
                        {pipelineStatus === 'error' || pipelineStatus === 'failed' ? 'Retry Pipeline' : 'Run Pipeline'}
                    </button>
                )}

                {isCompleted && (
                    <button
                        onClick={onRestart}
                        className="btn-secondary"
                        style={{ padding: '0.5rem 1.5rem', fontSize: '0.9rem' }}
                    >
                        Return to Start
                    </button>
                )}
            </div>
        </div>
    );
};

export default PipelineExecutionStep;
