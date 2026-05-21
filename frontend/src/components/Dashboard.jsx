import React from 'react';
import './Dashboard.css';

const Dashboard = ({ onStartPipeline, savedConfigs = [], onRerunConfig }) => {
    return (
        <div className="dashboard-container">
            <header className="dashboard-header text-center">
                <h1 className="dashboard-title">Welcome to Shakti DB AI</h1>
                <p className="dashboard-subtitle">Select a module to get started with your data intelligence journey.</p>
            </header>

            <div className="dashboard-grid">
                {/* AI Pipeline Card */}
                <div className="dashboard-card action-card" onClick={onStartPipeline}>
                    <div className="card-bg-gradient"></div>
                    <div className="card-content">
                        <div className="card-icon-wrapper">
                            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                                <path d="M2 17L12 22L22 17" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                                <path d="M2 12L12 17L22 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                            </svg>
                        </div>
                        <h2 className="card-title">AI Pipeline</h2>
                        <p className="card-description">Configure your data ingestion, database settings, and design your AI pipeline through our intuitive wizard.</p>
                        <div className="card-footer">
                            <span className="btn-start">Launch →</span>
                        </div>
                    </div>
                </div>

                {/* Test Module (Sandbox) Card */}
                <div className="dashboard-card static-card">
                    <div className="card-bg-gradient secondary"></div>
                    <div className="card-content">
                        <div className="card-icon-wrapper secondary">
                            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.77 3.77z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                            </svg>
                        </div>
                        <h2 className="card-title">Test Module (Sandbox)</h2>
                        <p className="card-description">Experiment with different configurations in a controlled environment. (Coming soon)</p>
                        <div className="card-footer">
                            <span className="badge">Soon</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* ── Saved Configurations — Re-run Shortcut ── */}
            {savedConfigs.length > 0 && (
                <div className="dashboard-saved-configs">
                    <h3 className="saved-configs-title">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/>
                        </svg>
                        Saved Configurations
                    </h3>
                    <p className="saved-configs-subtitle">Re-run a saved pipeline without stepping through the wizard again.</p>
                    <div className="saved-configs-list">
                        {savedConfigs.map((cfg) => {
                            const name = typeof cfg === 'string' ? cfg : (cfg.config_name || cfg.name || cfg);
                            return (
                                <div key={name} className="saved-config-row">
                                    <div className="saved-config-info">
                                        <span className="saved-config-icon">⚡</span>
                                        <div>
                                            <div className="saved-config-name">{name}</div>
                                            {cfg.created_at && <div className="saved-config-meta">Created {new Date(cfg.created_at).toLocaleDateString()}</div>}
                                        </div>
                                    </div>
                                    <button
                                        className="btn-rerun"
                                        onClick={() => onRerunConfig && onRerunConfig(name)}
                                        title="Re-run this pipeline"
                                    >
                                        ↻ Re-run
                                    </button>
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}
        </div>
    );
};

export default Dashboard;
