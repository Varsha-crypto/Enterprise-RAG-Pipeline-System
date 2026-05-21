import React, { useState, useEffect, useRef } from 'react';
import WizardLayout from './components/wizard/WizardLayout';
import DataIngestionStep from './components/wizard/steps/DataIngestionStep';
import DatabaseConfigurationStep from './components/wizard/steps/DatabaseConfigurationStep';
import PipelineDesignStep from './components/wizard/steps/PipelineDesignStep';
import PipelineExecutionStep from './components/wizard/steps/PipelineExecutionStep';
import Navbar from './components/Navbar';
import Dashboard from './components/Dashboard';
import LogViewer from './components/LogViewer';
import DuplicateWarningModal from './components/DuplicateWarningModal';
import './components/wizard/WizardLayout.css';
import './components/wizard/WizardStepper.css';
import './index.css';

// Import Real Service
import {
    fetchPipelineRecommendations,
    executePipeline,
    executePreembeddedPipeline,
    searchData,
    generateSummaryStream,
    getAvailableOptions,
    ingestData,
    saveDatabaseConfig,
    exportConfiguration,
    importConfiguration,
    getConfigDetails,
    getSourceDbColumns,
    cancelPipeline,
    listConfigs,
    checkTableExists,
    submitFeedback,
    checkDuplicates,
} from './services/api';
// executePreembeddedPipeline is used internally when a database source has a vector column set

const App = () => {
    const [view, setView] = useState('dashboard'); // 'dashboard' | 'wizard'
    const [currentStep, setCurrentStep] = useState(1);
    const [completedSteps, setCompletedSteps] = useState([]);

    // Guard against concurrent search requests
    const searchInProgress = useRef(false);

    // Config Name tracking for search — persisted across refresh via localStorage
    const [activeConfigName, setActiveConfigName] = useState(
        () => localStorage.getItem('shaktidb_active_config') || null
    );
    const [savedConfigs, setSavedConfigs] = useState([]);

    // ── Duplicate detection state ────────────────────────────────────────────
    const [dupWarning, setDupWarning] = useState(null); // null | { rowCount, tableName, onProceed }
    useEffect(() => {
        if (activeConfigName) localStorage.setItem('shaktidb_active_config', activeConfigName);
    }, [activeConfigName]);

    // Load saved configs list on mount
    useEffect(() => {
        listConfigs()
            .then(data => setSavedConfigs(data.configs || data || []))
            .catch(() => {});
    }, []);

    // --- Step 1: Data Ingestion State ---
    const [ingestionData, setIngestionData] = useState({
        dataSourceType: 'manual', // 'manual' | 'database'
        dataFormat: 'TXT',
        domain: '',
        dataSize: '',
        file: null,
        // Schema discovery (structured files)
        textColumn: '',
        embeddingColumn: '',
        // Database Source Fields
        dbHost: '',
        dbPort: '',
        dbUser: '',
        dbPassword: '',
        dbName: '',
        chunkColumn: '',
        tableName: '', // Used for manual upload as well
        tableNameExists: false,
        idColumn: '',
        schema: {} // { colName: "TEXT" | "INTEGER" | "FLOAT" | etc }
    });
    const [ingestionErrors, setIngestionErrors] = useState({});

    // --- Step 2: Database Configuration State ---
    const [databaseConfig, setDatabaseConfig] = useState({
        dbHost: '',
        dbPort: '',
        dbUser: '',
        dbPassword: '',
        dbName: '',
        tableName: ''
    });

    // --- Step 3: Pipeline Design State ---
    const [recommendations, setRecommendations] = useState({});
    const [pipelineOverrides, setPipelineOverrides] = useState({
        chunkingStrategy: { enabled: false, value: '' },
        embeddingModel: { enabled: false, value: '' },
        embeddingDimensions: { enabled: false, value: '' },
        huggingfaceToken: { enabled: false, value: '' },
        retrievalMethod: { enabled: false, value: '' }
    });
    const [availableOptions, setAvailableOptions] = useState({
        chunkingStrategy: [],
        embeddingModel: [],
        retrievalMethod: []
    });

    useEffect(() => {
        const loadOptions = async () => {
            const defaults = {
                chunkingStrategy: ['fixed_size', 'sentence_based'],
                embeddingModel: ['BAAI/bge-m3', 'BAAI/bge-small-en-v1.5'],
                retrievalMethod: ['hnsw', 'flat']
            };

            setAvailableOptions(defaults);

            try {
                const options = await getAvailableOptions();
                if (options) {
                    setAvailableOptions(prev => ({
                        ...prev,
                        embeddingModel: options.embedding_models?.length ? options.embedding_models : prev.embeddingModel,
                        retrievalMethod: options.index_types?.length ? options.index_types : prev.retrievalMethod
                    }));
                }
            } catch (error) {
                console.warn("Using fallback options. Backend options fetch failed:", error);
            }
        };
        loadOptions();
    }, []);

    // --- Step 4: Pipeline Execution State ---
    const [executionState, setExecutionState] = useState({
        status: 'idle', // idle, running, completed, error
        errorMessage: null,
        progress: 0,
        metrics: { chunks: 0, embeddings: 0 },
        batchesCompleted: 0,
        totalBatches: null,
        isPartialReady: false,
        canViewResults: false,
        pipelineMode: null,
        incrementalMode: false,
        isImported: false,
        startTime: null,
        executionTime: null,
        nodeTimings: {}
    });

    const [queryState, setQueryState] = useState({
        query: '',
        topK: 5,
        results: [],
        summary: null,
        thinkContent: '',
        prompt: 'You are a helpful assistant. Answer thoroughly and in detail based only on the context provided. Use all relevant information from the context to give a complete answer.',
        maxTokens: 1200,
        modelName: 'qwen3-0.6b',
        enableSummary: false,
        loading: false,
        summaryLoading: false,
    });

    const lastBatchRef = React.useRef(0);
    useEffect(() => {
        if (executionState.batchesCompleted > lastBatchRef.current) {
            lastBatchRef.current = executionState.batchesCompleted;
            if (activeConfigName && queryState.query && queryState.results && queryState.results.length > 0) {
                searchData(queryState.query, activeConfigName, queryState.topK, queryState.prompt, queryState.maxTokens, queryState.modelName)
                    .then(response => {
                        setQueryState(prev => {
                            if (prev.query === queryState.query) {
                                return { ...prev, results: response.results, summary: response.summary };
                            }
                            return prev;
                        });
                    })
                    .catch(err => console.error("Auto-refresh search failed", err));
            }
        }
    }, [executionState.batchesCompleted, activeConfigName, queryState.query, queryState.results, queryState.topK]);


    const handleStepChange = (step) => {
        if (step > currentStep && !completedSteps.includes(step - 1)) {
            return;
        }
        setCurrentStep(step);
    };

    const markStepComplete = (step) => {
        if (!completedSteps.includes(step)) {
            setCompletedSteps(prev => [...prev, step]);
        }
    };

    const handleIngestionChange = (field, value) => {
        setIngestionData(prev => ({
            ...prev,
            [field]: value
        }));

        if (field === 'tableName' && value) {
            handleTableCheck(value);
        }

        // Extract sample text when a file is uploaded (for chunk preview)
        if (field === 'file' && value instanceof File && value.type !== 'application/pdf') {
            const reader = new FileReader();
            reader.onload = (e) => {
                const text = e.target.result?.slice(0, 8000) || '';
                setIngestionData(prev => ({ ...prev, _sampleText: text }));
            };
            reader.readAsText(value);
        }
    };

    const handleTableCheck = async (tableName) => {
        try {
            // Use current databaseConfig or defaults
            const result = await checkTableExists(databaseConfig, tableName);
            setIngestionData(prev => ({ ...prev, tableNameExists: result.exists }));
        } catch (error) {
            console.warn("Table check failed:", error);
        }
    };

    const handleFetchColumns = async () => {
        try {
            const result = await getSourceDbColumns({
                dbHost: ingestionData.dbHost,
                dbPort: ingestionData.dbPort,
                dbUser: ingestionData.dbUser,
                dbPassword: ingestionData.dbPassword,
                dbName: ingestionData.dbName,
                tableName: ingestionData.tableName
            });
            if (result && result.columns) {
                const columnNamesStr = result.columns.map(col => col.name).join(', ');
                handleIngestionChange('columnNames', columnNamesStr);
            }
        } catch (error) {
            console.error("Failed to fetch columns:", error);
            alert("Failed to fetch columns from database: " + error.message);
        }
    };

    const handleIngestionSubmit = async () => {
        try {
            setIngestionErrors({});
            const errors = {};

            if (ingestionData.dataSourceType === 'manual') {
                if (!ingestionData.file) {
                    errors.file = "Please upload a file for manual ingestion.";
                }
            } else {
                const requiredFields = {
                    dbHost: 'Database Host', dbPort: 'Port', dbUser: 'Username',
                    dbPassword: 'Password', dbName: 'Database Name', tableName: 'Table Name',
                    chunkColumn: 'Text Column', idColumn: 'ID Column'
                };
                Object.keys(requiredFields).forEach(field => {
                    if (!ingestionData[field]) errors[field] = `${requiredFields[field]} is required.`;
                });
            }

            if (Object.keys(errors).length > 0) {
                setIngestionErrors(errors);
                return;
            }

            if (ingestionData.tableNameExists) {
                const confirmed = window.confirm(
                    `Table "${ingestionData.tableName}" already exists.\n\nProceeding will overwrite it and all existing data will be lost.\n\nContinue?`
                );
                if (!confirmed) return;
            }

            await ingestData(ingestionData);
            const recs = await fetchPipelineRecommendations(ingestionData);
            setRecommendations(recs);

            markStepComplete(1);

            if (ingestionData.dataSourceType === 'database') {
                // DB source skips the database-config step
                setDatabaseConfig({
                    dbHost: ingestionData.dbHost,
                    dbPort: ingestionData.dbPort,
                    dbUser: ingestionData.dbUser,
                    dbPassword: ingestionData.dbPassword,
                    dbName: ingestionData.dbName,
                    tableName: ingestionData.tableName
                });
                markStepComplete(2);
                setCurrentStep(3);
            } else {
                // Pre-populate database target table if user entered it in manual step
                if (ingestionData.tableName) {
                    setDatabaseConfig(prev => ({ ...prev, tableName: ingestionData.tableName }));
                }
                setCurrentStep(2);
            }
        } catch (error) {
            console.error("Ingestion failed", error);
            alert("Ingestion failed: " + error.message);
        }
    };

    const handleDatabaseChange = (field, value) => {
        setDatabaseConfig(prev => ({
            ...prev,
            [field]: value
        }));
    };

    const handleDatabaseSubmit = async () => {
        try {
            await saveDatabaseConfig(databaseConfig);
            markStepComplete(2);
            setCurrentStep(3);
        } catch (error) {
            console.error("DB Save failed", error);
        }
    };

    const handleOverrideToggle = (key) => {
        setPipelineOverrides(prev => ({
            ...prev,
            [key]: { ...prev[key], enabled: !prev[key].enabled }
        }));
    };

    const handleOverrideChange = (key, value) => {
        setPipelineOverrides(prev => ({
            ...prev,
            [key]: { ...prev[key], value: value }
        }));
    };

    const handlePipelineSubmit = () => {
        markStepComplete(3);
        setCurrentStep(4);
    };

    const handleExecutePipeline = async () => {
        // ── Duplicate detection ──────────────────────────────────────────────
        if (databaseConfig.dbHost && databaseConfig.tableName) {
            try {
                const dupResult = await checkDuplicates({
                    dbHost: databaseConfig.dbHost,
                    dbPort: databaseConfig.dbPort,
                    dbUser: databaseConfig.dbUser,
                    dbPassword: databaseConfig.dbPassword,
                    dbName: databaseConfig.dbName,
                    tableName: databaseConfig.tableName,
                });
                if (dupResult.exists && dupResult.row_count > 0) {
                    // Show warning modal and pause until user confirms
                    await new Promise((resolve) => {
                        setDupWarning({
                            rowCount: dupResult.row_count,
                            tableName: databaseConfig.tableName,
                            onProceed: () => { setDupWarning(null); resolve(true); },
                            onCancel:  () => { setDupWarning(null); resolve(false); },
                        });
                    }).then((proceed) => {
                        if (!proceed) throw new Error('__USER_CANCELLED__');
                    });
                }
            } catch (e) {
                if (e.message === '__USER_CANCELLED__') return;
                // Non-fatal: if dup check fails, proceed anyway
            }
        }
        // ────────────────────────────────────────────────────────────────────

        setExecutionState(prev => ({
            ...prev,
            status: 'running',
            errorMessage: null,
            progress: 0,
            metrics: { chunks: 0, embeddings: 0 },
            pipelineMode: null,
            batchesCompleted: 0,
            totalBatches: null,
            startTime: Date.now(),
            executionTime: null,
            nodeTimings: {}
        }));

        try {
            // Pre-embedded path: database source with an existing vector column — skip chunk/embed
            const isPreembedded = ingestionData.dataSourceType === 'database' && !!ingestionData.embeddingColumn;

            const payload = {
                sourceType: ingestionData.dataSourceType,
                database: databaseConfig,
                ...ingestionData,
                file: ingestionData.dataSourceType === 'manual' ? ingestionData.file : null,
                textColumn: ingestionData.textColumn || null,
                embeddingColumn: ingestionData.embeddingColumn || null,
                pipeline: {
                    recommendations,
                    overrides: pipelineOverrides,
                    incremental_mode: executionState.incrementalMode,
                    top_k: queryState.topK
                }
            };

            if (isPreembedded) {
                const result = await executePreembeddedPipeline(
                    payload,
                    (progress, metrics) => {
                        if (metrics.config_name) setActiveConfigName(metrics.config_name);
                        setExecutionState(prev => ({
                            ...prev,
                            progress: progress != null ? progress : prev.progress,
                            status: metrics.status === 'completed' ? 'completed' : 'running',
                            stage: metrics.stage || prev.stage || '',
                            batchesCompleted: 1,
                            totalBatches: 1,
                            canViewResults: metrics.status === 'completed',
                        }));
                    },
                    ({ configName }) => setActiveConfigName(configName)
                );
                setActiveConfigName(result.config_name);
                setExecutionState(prev => ({
                    ...prev,
                    status: result.error ? 'error' : 'completed',
                    progress: 100,
                    canViewResults: true,
                    batchesCompleted: 1,
                    totalBatches: 1,
                }));
                if (!result.error) markStepComplete(4);
                return;
            }

            let lastBatchesCompleted = 0;

            const result = await executePipeline(
                payload,
                (progress, metrics) => {
                    // Handle upload progress events from XHR
                    if (progress && typeof progress === 'object' && progress.type === 'upload') {
                        setExecutionState(prev => ({
                            ...prev,
                            status: 'uploading',
                            progress: progress.percent,
                            stage: `Uploading file… ${progress.percent}%`,
                        }));
                        return;
                    }

                    if (metrics.config_name) {
                        setActiveConfigName(metrics.config_name);
                    }

                    setExecutionState(prev => {
                        if (metrics.status === 'cancelled' || metrics.cancelled) {
                            return { ...prev, status: 'cancelled_and_cleaned', progress: 100 };
                        }

                        const newBatchesCompleted = metrics.batches_completed !== undefined ? metrics.batches_completed : prev.batchesCompleted;
                        const newTotalBatches = metrics.total_batches !== undefined ? metrics.total_batches : prev.totalBatches;
                        const isPartialReady = metrics.mode === 'partially_ready' || metrics.mode === 'ready' || newBatchesCompleted > 0;

                        let nextMode = prev.pipelineMode;
                        if (!nextMode && metrics.mode) {
                            if (metrics.mode === 'single' || metrics.mode === 'batch') {
                                nextMode = metrics.mode;
                            } else if (newTotalBatches > 1) {
                                nextMode = 'batch';
                            } else if (newTotalBatches === 1) {
                                nextMode = 'single';
                            }
                        }

                        const nextNodeTimings = { ...prev.nodeTimings };
                        const stepSize = 100 / 6;
                        const now = Date.now();

                        for (let i = 0; i < 6; i++) {
                            const threshold = (i + 1) * stepSize;
                            if (!nextNodeTimings[i] && (progress >= threshold - 0.5 || metrics.pipeline_completed)) {
                                const startTime = prev.startTime || now;
                                let totalAccumulatedTime = 0;
                                for (let j = 0; j < i; j++) {
                                    totalAccumulatedTime += parseFloat(nextNodeTimings[j] || 0);
                                }
                                const elapsedTillNow = (now - startTime) / 1000;
                                const nodeDuration = Math.max(0, elapsedTillNow - totalAccumulatedTime);
                                nextNodeTimings[i] = `${nodeDuration.toFixed(1)}s`;
                            }
                        }

                        return {
                            ...prev,
                            progress: progress != null ? progress : prev.progress,
                            pipelineMode: nextMode,
                            nodeTimings: nextNodeTimings,
                            metrics: {
                                chunks: (metrics.chunks !== undefined && !isNaN(metrics.chunks)) ? Math.max(metrics.chunks, prev.metrics.chunks) : prev.metrics.chunks,
                                embeddings: (metrics.embeddings !== undefined && !isNaN(metrics.embeddings)) ? Math.max(metrics.embeddings, prev.metrics.embeddings) : prev.metrics.embeddings
                            },
                            batchesCompleted: metrics.batches_completed ?? prev.batchesCompleted,
                            totalBatches: metrics.total_batches ?? prev.totalBatches,
                            stage: metrics.stage || prev.stage || '',
                            isPartialReady,
                            canViewResults: isPartialReady && (metrics.batches_completed ?? prev.batchesCompleted) > 0
                        };
                    });

                    if (metrics.partial_results && metrics.partial_results.length > 0) {
                        setQueryState(prev => {
                            if (prev.results.length === 0 && !prev.loading) {
                                return {
                                    ...prev,
                                    results: metrics.partial_results.map(r => ({
                                        id: r.id,
                                        score: r.score || r.similarity_score,
                                        content: r.text || r.content,
                                        metadata: { model: r.embedding_model || r.metadata?.source }
                                    }))
                                };
                            }
                            return prev;
                        });
                    }
                },
                ({ configName }) => {
                    setActiveConfigName(configName);
                }
            );

            setActiveConfigName(result.config_name);
            const finalStatus = (result.status === 'failed' || result.error)
                ? 'failed'
                : (result.status === 'cancelled' || result.cancelled) ? 'cancelled' : 'completed';

            if (finalStatus === 'cancelled') {
                setExecutionState(prev => ({ ...prev, status: 'cancelled_and_cleaned' }));
                await new Promise(r => setTimeout(r, 2000));
                handleResetApp();
                return;
            }

            setExecutionState(prev => {
                if (prev.status === 'cleaning' || prev.status === 'cancelled_and_cleaned') {
                    return prev;
                }
                const endTime = Date.now();
                const diffMs = prev.startTime ? (endTime - prev.startTime) : 0;
                const totalSeconds = diffMs / 1000;
                const minutes = Math.floor(totalSeconds / 60);
                const seconds = (totalSeconds % 60).toFixed(1);
                const durationStr = minutes > 0 ? `${minutes}m ${seconds}s` : `${seconds}s`;

                return {
                    ...prev,
                    status: finalStatus,
                    progress: finalStatus === 'completed' ? 100 : prev.progress,
                    batchesCompleted: result.batches_completed ?? prev.batchesCompleted,
                    totalBatches: result.total_batches ?? prev.totalBatches,
                    executionTime: durationStr
                };
            });

            if (finalStatus === 'completed') {
                markStepComplete(4);
            }

        } catch (error) {
            console.error("Pipeline Execution Failed:", error);
            const isCancelled = error.message && error.message.toLowerCase().includes('cancel');
            if (!isCancelled) {
                setExecutionState(prev => ({ ...prev, status: 'error', errorMessage: error.message || 'Unknown error' }));
            } else {
                handleResetApp();
            }
        }
    };

    const handleResetApp = () => {
        setExecutionState({
            status: 'idle', progress: 0, metrics: { chunks: 0, embeddings: 0 }, canViewResults: false, pipelineMode: null,
            batchesCompleted: 0, totalBatches: 0, stage: '', incrementalMode: false, isImported: false,
            executionTime: null, nodeTimings: {}
        });
        setIngestionData({
            dataSourceType: 'manual', dataFormat: 'TXT', domain: '', dataSize: '', file: null,
            textColumn: '', embeddingColumn: '',
            dbHost: '', dbPort: '', dbUser: '', dbPassword: '', dbName: '', tableName: '', columnNames: '', chunkColumn: '', idColumn: ''
        });
        setDatabaseConfig({ dbHost: '', dbPort: '', dbUser: '', dbPassword: '', dbName: '', tableName: '' });
        setPipelineOverrides({
            chunkingStrategy: { enabled: false, value: '' },
            embeddingModel: { enabled: false, value: '' },
            embeddingDimensions: { enabled: false, value: '' },
            huggingfaceToken: { enabled: false, value: '' },
            retrievalMethod: { enabled: false, value: '' }
        });
        setRecommendations({});
        setQueryState({ query: '', topK: 5, results: [], summary: null, prompt: 'You are a helpful assistant. Answer thoroughly and in detail based only on the context provided. Use all relevant information from the context to give a complete answer.', maxTokens: 1200, modelName: 'qwen3-0.6b', loading: false });
        setCurrentStep(1);
        setCompletedSteps([]);
        setActiveConfigName(null);
    };

    const handleExportConfig = async () => {
        if (activeConfigName) {
            try {
                const blob = await exportConfiguration(activeConfigName);
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `${activeConfigName}.zip`;
                a.click();
                URL.revokeObjectURL(url);
            } catch (error) {
                console.error("Export failed", error);
                alert("Failed to export configuration from backend.");
            }
        }
    };

    const handleImportConfig = async (file) => {
        if (!file) return;
        try {
            const result = await importConfiguration(file);
            if (result.config) {
                const config = result.config;
                if (config.ingestion) {
                    setIngestionData(prev => ({
                        ...prev,
                        dataSourceType: config.ingestion.source_type || prev.dataSourceType,
                        dataFormat: config.ingestion.data_format || prev.dataFormat,
                        domain: config.ingestion.domain || prev.domain,
                        dataSize: config.ingestion.data_size || prev.dataSize,
                        ...(config.ingestion.source_db ? {
                            dbHost: config.ingestion.source_db.host || '',
                            dbPort: config.ingestion.source_db.port || '',
                            dbUser: config.ingestion.source_db.user || '',
                            dbPassword: config.ingestion.source_db.password || '',
                            dbName: config.ingestion.source_db.dbname || '',
                            tableName: config.ingestion.source_db.table || '',
                            chunkColumn: config.ingestion.source_db.chunk_column || '',
                            idColumn: config.ingestion.source_db.id_column || ''
                        } : {})
                    }));
                }
                if (config.database) {
                    setDatabaseConfig({
                        dbHost: config.database.host || '',
                        dbPort: config.database.port || '',
                        dbUser: config.database.user || '',
                        dbPassword: config.database.password || '',
                        dbName: config.database.dbname || '',
                        tableName: config.database.table || 'documents'
                    });
                }
                if (config.pipeline) {
                    if (config.pipeline.recommendations) setRecommendations(config.pipeline.recommendations);
                    if (config.pipeline.overrides) setPipelineOverrides(config.pipeline.overrides);
                }
                setActiveConfigName(config.config_name);
                if (result.ready) {
                    setExecutionState(prev => ({
                        ...prev,
                        status: 'completed',
                        progress: 100,
                        canViewResults: true,
                        isPartialReady: true,
                        isImported: true
                    }));
                    markStepComplete(1);
                    markStepComplete(2);
                    markStepComplete(3);
                    setCurrentStep(4);
                }
            }
        } catch (error) {
            console.error("Import failed", error);
            alert("Failed to import configuration: " + error.message);
        }
    };

    const handleCancelPipeline = async () => {
        if (!activeConfigName) {
            handleResetApp();
            return;
        }
        setExecutionState(prev => ({ ...prev, status: 'cleaning' }));
        try {
            await cancelPipeline(activeConfigName);
            setTimeout(() => {
                setExecutionState(prev => {
                    if (prev.status === 'cleaning') {
                        handleResetApp();
                    }
                    return prev;
                });
            }, 5000);
        } catch (error) {
            console.error("Cancellation API failed", error);
            setTimeout(handleResetApp, 2000);
        }
    };

    // ── Feature: Result Feedback ─────────────────────────────────────────────
    const handleSubmitFeedback = async (payload) => {
        try { await submitFeedback(payload); } catch (e) { console.warn('Feedback submit failed:', e); }
    };

    // ── Feature: Pipeline Re-run from Dashboard ──────────────────────────────
    const handleRerunConfig = async (configName) => {
        try {
            const cfg = await getConfigDetails(configName);
            if (cfg && cfg.database) {
                setDatabaseConfig({
                    dbHost: cfg.database.host || '',
                    dbPort: cfg.database.port || '',
                    dbUser: cfg.database.user || '',
                    dbPassword: cfg.database.password || '',
                    dbName: cfg.database.dbname || '',
                    tableName: cfg.database.table || 'documents',
                });
            }
            if (cfg && cfg.pipeline) {
                if (cfg.pipeline.recommendations) setRecommendations(cfg.pipeline.recommendations);
                if (cfg.pipeline.overrides) setPipelineOverrides(cfg.pipeline.overrides);
            }
            setActiveConfigName(configName);
            markStepComplete(1);
            markStepComplete(2);
            markStepComplete(3);
            setCurrentStep(4);
            setView('wizard');
        } catch (e) {
            alert('Failed to load saved config for re-run: ' + e.message);
        }
    };

    const handleSearch = async () => {
        if (!activeConfigName) {
            setQueryState(prev => ({ ...prev, searchError: 'No active configuration found. Please run the pipeline first.' }));
            return;
        }
        if (searchInProgress.current) return;
        searchInProgress.current = true;

        // Step 1: fetch results immediately — no LLM involved
        setQueryState(prev => ({ ...prev, loading: true, summary: null, summaryLoading: false, results: [], searchError: null }));
        let retrievedChunks = [];
        try {
            const response = await searchData(queryState.query, activeConfigName, queryState.topK);
            retrievedChunks = (response.results || []).map(r => r.content);
            setQueryState(prev => ({ ...prev, results: response.results, loading: false }));
        } catch (error) {
            console.error("Search failed", error);
            // Show error inline — no alert popup
            setQueryState(prev => ({
                ...prev,
                loading: false,
                searchError: error.message || 'Search failed. Please check the backend is running.',
            }));
            searchInProgress.current = false;
            return;
        }

        // Results are displayed — release the lock so the user can search again
        // independently of whether summarisation is still running.
        searchInProgress.current = false;

        // Step 2: stream summary — only when user opted in, fires independently
        if (retrievedChunks.length > 0 && queryState.modelName && queryState.enableSummary) {
            setQueryState(prev => ({ ...prev, summaryLoading: true, summary: null, thinkContent: '' }));
            const summaryChunks = retrievedChunks.slice(0, 10);
            generateSummaryStream(
                queryState.query,
                summaryChunks,
                queryState.modelName,
                queryState.prompt,
                queryState.maxTokens,
                {
                    onToken: (type, token) => {
                        if (type === 'think') {
                            setQueryState(prev => ({ ...prev, thinkContent: prev.thinkContent + token }));
                        } else if (type === 'answer') {
                            setQueryState(prev => ({ ...prev, summary: (prev.summary || '') + token }));
                        }
                    },
                    onDone: () => {
                        setQueryState(prev => ({ ...prev, summaryLoading: false }));
                    },
                    onError: (err) => {
                        console.warn("Summary stream failed (non-fatal):", err);
                        setQueryState(prev => ({ ...prev, summaryLoading: false }));
                    }
                }
            );
        }
    };

    const renderStepContent = () => {
        switch (currentStep) {
            case 1:
                return (
                    <DataIngestionStep
                        dataSourceType={ingestionData.dataSourceType}
                        dataFormat={ingestionData.dataFormat}
                        domain={ingestionData.domain}
                        dataSize={ingestionData.dataSize}
                        file={ingestionData.file}
                        textColumn={ingestionData.textColumn}
                        embeddingColumn={ingestionData.embeddingColumn}
                        schema={ingestionData.schema}
                        dbHost={ingestionData.dbHost}
                        dbPort={ingestionData.dbPort}
                        dbUser={ingestionData.dbUser}
                        dbPassword={ingestionData.dbPassword}
                        dbName={ingestionData.dbName}
                        tableName={ingestionData.tableName}
                        columnNames={ingestionData.columnNames}
                        chunkColumn={ingestionData.chunkColumn}
                        idColumn={ingestionData.idColumn}
                        errors={ingestionErrors}
                        onChange={handleIngestionChange}
                        onFetchColumns={handleFetchColumns}
                        onSubmit={handleIngestionSubmit}
                        onImportConfig={handleImportConfig}
                    />
                );
            case 2:
                return (
                    <DatabaseConfigurationStep
                        dbHost={databaseConfig.dbHost}
                        dbPort={databaseConfig.dbPort}
                        dbUser={databaseConfig.dbUser}
                        dbPassword={databaseConfig.dbPassword}
                        dbName={databaseConfig.dbName}
                        tableName={databaseConfig.tableName}
                        onChange={handleDatabaseChange}
                        onSubmit={handleDatabaseSubmit}
                        onBack={() => setCurrentStep(1)}
                    />
                );
            case 3:
                return (
                    <PipelineDesignStep
                        recommendations={recommendations}
                        overrides={pipelineOverrides}
                        availableOptions={availableOptions}
                        onOverrideToggle={handleOverrideToggle}
                        onOverrideChange={handleOverrideChange}
                        onExecute={() => {
                            handlePipelineSubmit();
                            handleExecutePipeline();
                        }}
                        onBack={() => setCurrentStep(ingestionData.dataSourceType === 'database' ? 1 : 2)}
                        incrementalMode={executionState.incrementalMode}
                        onIncrementalModeChange={(val) => setExecutionState(prev => ({ ...prev, incrementalMode: val }))}
                        sampleText={ingestionData._sampleText || ''}
                        chunkSize={500}
                        chunkOverlap={50}
                    />
                );
            case 4:
                return (
                    <PipelineExecutionStep
                        pipelineStatus={executionState.status}
                        errorMessage={executionState.errorMessage}
                        progress={executionState.progress}
                        metrics={executionState.metrics}
                        onExecute={handleExecutePipeline}
                        onExport={handleExportConfig}
                        onCancelPipeline={handleCancelPipeline}
                        onBack={() => setCurrentStep(3)}
                        canViewResults={executionState.canViewResults}
                        pipelineMode={executionState.pipelineMode}
                        batchesCompleted={executionState.batchesCompleted}
                        totalBatches={executionState.totalBatches}
                        stage={executionState.stage}
                        incrementalMode={executionState.incrementalMode}
                        isImported={executionState.isImported}
                        executionTime={executionState.executionTime}
                        nodeTimings={executionState.nodeTimings}
                        query={queryState.query}
                        topK={queryState.topK}
                        results={queryState.results}
                        loading={queryState.loading}
                        summaryLoading={queryState.summaryLoading}
                        summary={queryState.summary}
                        prompt={queryState.prompt}
                        maxTokens={queryState.maxTokens}
                        modelName={queryState.modelName}
                        enableSummary={queryState.enableSummary}
                        searchError={queryState.searchError}
                        onQueryChange={(q) => setQueryState(prev => ({ ...prev, query: q, searchError: null }))}
                        onTopKChange={(k) => setQueryState(prev => ({ ...prev, topK: k }))}
                        onPromptChange={(p) => setQueryState(prev => ({ ...prev, prompt: p }))}
                        onMaxTokensChange={(t) => setQueryState(prev => ({ ...prev, maxTokens: t }))}
                        onModelNameChange={(m) => setQueryState(prev => ({ ...prev, modelName: m }))}
                        thinkContent={queryState.thinkContent}
                        onEnableSummaryChange={(v) => setQueryState(prev => ({ ...prev, enableSummary: v, summary: v ? prev.summary : null, thinkContent: v ? prev.thinkContent : '' }))}
                        onSearch={handleSearch}
                        onRestart={handleResetApp}
                        activeConfigName={activeConfigName}
                        onSubmitFeedback={handleSubmitFeedback}
                    />
                );
            default:
                return <div>Unknown Step</div>;
        }
    };

    return (
        <div className="app-main-wrapper">
            <Navbar onLogoClick={() => setView('dashboard')} />
            <main className="main-content">
                {view === 'dashboard' ? (
                    <Dashboard
                        onStartPipeline={() => setView('wizard')}
                        savedConfigs={savedConfigs}
                        onRerunConfig={handleRerunConfig}
                    />
                ) : (
                    <WizardLayout
                        currentStep={currentStep}
                        completedSteps={completedSteps}
                        onStepChange={handleStepChange}
                        showExecutionStep={executionState.status !== 'idle'}
                        skippedSteps={ingestionData.dataSourceType === 'database' ? [2] : []}
                        footer={null}
                    >
                        {renderStepContent()}
                    </WizardLayout>
                )}
                {/* ── Log Viewer (always visible at bottom of wizard) ── */}
                {view === 'wizard' && (
                    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '0 1.5rem 2rem' }}>
                        <LogViewer autoRefresh={executionState.status === 'running'} />
                    </div>
                )}
            </main>
            {/* ── Duplicate Warning Modal ── */}
            {dupWarning && (
                <DuplicateWarningModal
                    rowCount={dupWarning.rowCount}
                    tableName={dupWarning.tableName}
                    onProceed={dupWarning.onProceed}
                    onCancel={dupWarning.onCancel}
                />
            )}
        </div>
    );
};

export default App;
