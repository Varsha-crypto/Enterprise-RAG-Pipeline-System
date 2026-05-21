import React, { useState, useEffect, useRef } from 'react';

const BatchProgressBar = ({
    progress,
    completedBatches,
    totalBatches,
    stage,
    pipelineStatus = 'idle'
}) => {
    const isCompleted = pipelineStatus === 'completed';
    const isError = pipelineStatus === 'error';
    const isRunning = pipelineStatus === 'running';

    const [eta, setEta] = useState(null);
    const startTimeRef = useRef(null);
    const lastBatchRef = useRef(0);

    useEffect(() => {
        if (isRunning && !startTimeRef.current && completedBatches === 0) {
            startTimeRef.current = Date.now();
        }

        if (isRunning && completedBatches > lastBatchRef.current) {
            const now = Date.now();
            const elapsed = now - startTimeRef.current;
            const timePerBatch = elapsed / completedBatches;
            const remainingBatches = totalBatches - completedBatches;
            const remainingTime = timePerBatch * remainingBatches;
            
            setEta(remainingTime);
            lastBatchRef.current = completedBatches;
        }

        if (isCompleted || !isRunning) {
            setEta(null);
            if (!isRunning) {
                startTimeRef.current = null;
                lastBatchRef.current = 0;
            }
        }
    }, [completedBatches, totalBatches, isRunning, isCompleted]);

    const formatEta = (ms) => {
        if (!ms || ms < 0) return null;
        const seconds = Math.round(ms / 1000);
        if (seconds < 60) return `${seconds}s remaining`;
        const minutes = Math.floor(seconds / 60);
        const remSeconds = seconds % 60;
        return `${minutes}m ${remSeconds}s remaining`;
    };

    // Strict percentage based on batches
    const calculatePercent = () => {
        if (!totalBatches || totalBatches <= 0) return 0;
        const percent = (completedBatches / totalBatches) * 100;
        return Math.min(Math.max(percent, 0), 100);
    };

    const percent = calculatePercent();
    const displayPercent = Math.round(Math.max(percent, progress || 0));

    return (
        <div style={{ marginTop: '1.5rem', marginBottom: '1.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', fontSize: '0.85rem', color: '#4b5563', fontWeight: '600' }}>
                <span style={{ color: '#1e293b' }}>
                    {isCompleted ? 'All Batches Processed' : `Processing batches: ${completedBatches}${totalBatches ? ` / ${totalBatches}` : ''}`}
                </span>
                <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                    {isRunning && eta && (
                        <span style={{ color: '#6b7280', fontWeight: 'normal', fontSize: '0.8rem' }}>
                            {formatEta(eta)}
                        </span>
                    )}
                    <span style={{ color: isCompleted ? '#10b981' : '#4f46e5' }}>{displayPercent}%</span>
                </div>
            </div>
            <div style={{ height: '12px', background: '#f1f5f9', border: '1px solid #e2e8f0', borderRadius: '6px', overflow: 'hidden' }}>
                <div
                    className="batch-progress-bar-fill"
                    style={{
                        height: '100%',
                        width: `${displayPercent}%`,
                        background: isCompleted
                            ? 'linear-gradient(90deg, #10b981, #34d399)'
                            : isError
                                ? '#ef4444'
                                : 'linear-gradient(90deg, #4f46e5, #818cf8)',
                        borderRadius: '6px',
                        transition: 'width 0.6s cubic-bezier(0.34, 1.56, 0.64, 1)'
                    }}
                />
            </div>
            {(stage || pipelineStatus !== 'idle') && (
                <p style={{ marginTop: '0.6rem', fontSize: '0.8rem', color: '#64748b', fontWeight: '500' }}>
                    {isCompleted ? '✓ Pipeline execution finished successfully' : stage || 'Initializing batch processing...'}
                </p>
            )}
        </div>
    );
};

export default React.memo(BatchProgressBar);
