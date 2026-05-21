const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

/**
 * Executes the pipeline with correct SSE-first orchestration.
 */
export const executePipeline = async (payload, onProgress, onStart = null) => {
    const {
        file, database, pipeline, sourceType,
        dbHost, dbPort, dbUser, dbPassword, dbName,
        tableName, chunkColumn, idColumn,
        textColumn, embeddingColumn
    } = payload;

    const incrementalMode = pipeline?.incremental_mode || false;

    const formData = new FormData();
    if (file && sourceType === 'manual') {
        formData.append('file', file);
    }

    const params = new URLSearchParams();
    if (sourceType) {
        params.append('source_type', sourceType === 'manual' ? 'file' : sourceType);
    }
    if (sourceType === 'database') {
        params.append('source_db_host', dbHost || '');
        params.append('source_db_port', parseInt(dbPort) || 0);
        params.append('source_db_user', dbUser || '');
        params.append('source_db_password', dbPassword || '');
        params.append('source_db_name', dbName || '');
        params.append('source_db_table', tableName || '');
        params.append('chunk_column', chunkColumn || '');
        params.append('id_column', idColumn || 'id');
    }
    if (database) {
        params.append('db_host', database.dbHost || '');
        params.append('db_port', parseInt(database.dbPort) || 0);
        params.append('db_user', database.dbUser || '');
        params.append('db_password', database.dbPassword || '');
        params.append('db_name', database.dbName || '');
        params.append('db_table', database.tableName || 'documents');
    }
    if (pipeline.overrides.embeddingModel.enabled) {
        const modelValue = pipeline.overrides.embeddingModel.value === 'other'
            ? pipeline.overrides.embeddingModelManual?.value
            : pipeline.overrides.embeddingModel.value;
        params.append('embedding_model', modelValue);
    }
    if (pipeline.overrides.embeddingDimensions?.enabled && pipeline.overrides.embeddingDimensions.value) {
        params.append('vector_dim', pipeline.overrides.embeddingDimensions.value);
    }
    if (pipeline.overrides.huggingfaceToken?.enabled && pipeline.overrides.huggingfaceToken.value) {
        params.append('hf_token', pipeline.overrides.huggingfaceToken.value);
    }
    if (pipeline.overrides.chunkingStrategy.enabled) {
        params.append('chunking_strategy', pipeline.overrides.chunkingStrategy.value);
    }
    if (pipeline.overrides.retrievalMethod.enabled) {
        params.append('index_type', pipeline.overrides.retrievalMethod.value);
    }

    params.append('incremental_mode', incrementalMode);
    params.append('top_k', pipeline?.top_k || 3);
    if (textColumn) params.append('text_column', textColumn);
    if (embeddingColumn) params.append('embedding_column', embeddingColumn);
    
    // Add custom schema for manual uploads
    if (sourceType === 'manual' && payload.schema) {
        params.append('schema', JSON.stringify(payload.schema));
    }

    let configName = "";
    let executeEndpoint = "";

    if (sourceType === 'database') {
        const dbPayload = {
            source: {
                dbHost, dbPort, dbUser, dbPassword, dbName, dbTable: tableName, chunkColumn, idColumn
            },
            // Always let the backend auto-create a fresh target DB on the same server.
            // Passing the source as target would INSERT into the source table (wrong schema).
            target: null,
            pipeline: {
                embeddingModel: pipeline.overrides.embeddingModel.enabled ? pipeline.overrides.embeddingModel.value : "BAAI/bge-m3",
                embeddingModelManual: pipeline.overrides.embeddingModelManual?.value,
                embeddingDimensions: pipeline.overrides.embeddingDimensions?.enabled ? pipeline.overrides.embeddingDimensions.value : null,
                huggingfaceToken: pipeline.overrides.huggingfaceToken?.enabled ? pipeline.overrides.huggingfaceToken.value : null,
                indexType: pipeline.overrides.retrievalMethod.enabled ? pipeline.overrides.retrievalMethod.value : "hnsw",
                chunkingStrategy: pipeline.overrides.chunkingStrategy.enabled ? pipeline.overrides.chunkingStrategy.value : "fixed_size",
                incrementalMode: incrementalMode,
                topK: pipeline?.top_k || 3
            }
        };
        const configResult = await configureDbPipeline(dbPayload);
        configName = configResult.config_name;
        executeEndpoint = '/execute-pipeline-from-db';
    } else {
        const uploadResult = await new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            xhr.open('POST', `${API_BASE_URL}/api/upload-file-for-pipeline?${params.toString()}`);
            xhr.upload.onprogress = (e) => {
                if (e.lengthComputable && onProgress) {
                    onProgress({ type: 'upload', percent: Math.round((e.loaded / e.total) * 100) });
                }
            };
            xhr.onload = () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                    try { resolve(JSON.parse(xhr.responseText)); }
                    catch { reject(new Error('Invalid response from server')); }
                } else {
                    let detail = 'Upload failed';
                    try { detail = JSON.parse(xhr.responseText).detail || detail; } catch {}
                    reject(new Error(detail));
                }
            };
            xhr.onerror = () => reject(new Error('Network error during upload'));
            xhr.send(formData);
        });
        configName = uploadResult.config_name;
        executeEndpoint = '/execute-pipeline-from-file';
    }

    if (onStart) onStart({ configName });

    let progressId = null;
    try {
        const trackerResponse = await fetch(`${API_BASE_URL}/create-pipeline-tracker`, { method: 'POST' });
        if (trackerResponse.ok) {
            const trackerResult = await trackerResponse.json();
            progressId = trackerResult.progress_id;
        } else {
            throw new Error('tracker endpoint error');
        }
    } catch (trackerErr) {
        console.warn('[SSE] create-pipeline-tracker failed, using legacy execute-first path:', trackerErr.message);
        return _legacyExecuteAndTrack(configName, onProgress);
    }

    return new Promise((resolve, reject) => {
        let executionTriggered = false;

        _streamSSEProgress(
            progressId,
            configName,
            onProgress,
            async () => {
                if (executionTriggered) return;
                executionTriggered = true;

                try {
                    const execResponse = await fetch(
                        `${API_BASE_URL}${executeEndpoint}?config_name=${configName}&progress_id=${progressId}`,
                        { method: 'POST' }
                    );
                    if (!execResponse.ok) {
                        const err = await execResponse.json().catch(() => ({}));
                        throw new Error(err.detail || 'Execution trigger failed');
                    }
                } catch (e) {
                    reject(e);
                }
            }
        ).then(resolve).catch(reject);
    });
};

function _streamSSEProgress(progressId, configName, onProgress, onOpen = null) {
    return new Promise((resolve, reject) => {
        const sseUrl = `${API_BASE_URL}/pipeline-progress-stream/${progressId}`;
        let settled = false;
        let reconnectCount = 0;
        const MAX_RECONNECTS = 5;
        const STALE_TIMEOUT_MS = 600000;

        let es = null;
        let staleTimer = null;

        const done = (fn, val) => {
            if (settled) return;
            settled = true;
            clearTimeout(staleTimer);
            if (es) { es.close(); es = null; }
            fn(val);
        };

        const resetStaleTimer = () => {
            if (settled) return;
            clearTimeout(staleTimer);
            staleTimer = setTimeout(() => {
                done(reject, new Error(`SSE stalled: no updates for ${STALE_TIMEOUT_MS / 1000}s. Pipeline may have crashed silently.`));
            }, STALE_TIMEOUT_MS);
        };

        const openStream = () => {
            if (settled) return;
            if (es) { es.close(); es = null; }

            es = new EventSource(sseUrl);
            resetStaleTimer();

            es.onopen = () => {
                if (onOpen) onOpen();
            };

            es.onmessage = (event) => {
                resetStaleTimer();
                if (!event.data || !event.data.trim()) return;

                try {
                    const status = JSON.parse(event.data);
                    if (status.heartbeat) return;
                    if (status.error && !status.pipeline_completed) {
                        done(reject, new Error(status.error));
                        return;
                    }

                    if (onProgress) {
                        const progressValue = status.progress ?? status.overall_progress ?? status.percent ?? 0;
                        onProgress(progressValue, {
                            chunks: status.chunks_processed !== undefined ? Number(status.chunks_processed) : undefined,
                            embeddings: status.embeddings_generated !== undefined ? Number(status.embeddings_generated) : undefined,
                            status: (status.status === 'failed' || status.status === 'error' || status.error)
                                ? 'failed'
                                : (status.pipeline_completed ? 'completed' : (status.status || 'running')),
                            batches_completed: status.batches_completed !== undefined ? Number(status.batches_completed) : undefined,
                            total_batches: status.total_batches !== undefined ? Number(status.total_batches) : undefined,
                            mode: status.mode || null,
                            stage: status.stage || status.current_step || '',
                            model_loading: status.model_loading || false,
                            config_name: configName,
                            partial_results: status.partial_results || []
                        });
                    }

                    if (status.pipeline_completed === true) {
                        done(resolve, { ...status, config_name: configName });
                    } else if (status.status === 'cancelled') {
                        done(resolve, { ...status, config_name: configName, cancelled: true });
                    } else if (status.status === 'failed' || status.status === 'error') {
                        done(reject, new Error(status.error || 'Pipeline execution failed'));
                    }

                } catch (e) {
                    console.error('[SSE] JSON parse error:', e);
                }
            };

            es.onerror = (_err) => {
                if (settled) return;
                const state = es ? es.readyState : -1;
                if (state === EventSource.CLOSED || state === EventSource.CONNECTING) {
                    if (reconnectCount < MAX_RECONNECTS) {
                        reconnectCount++;
                        const delay = Math.min(1000 * reconnectCount, 5000);
                        clearTimeout(staleTimer);
                        if (es) { es.close(); es = null; }
                        setTimeout(openStream, delay);
                    } else {
                        done(reject, new Error('SSE failed after max reconnects.'));
                    }
                }
            };
        };

        openStream();
    });
}

async function _legacyExecuteAndTrack(configName, onProgress) {
    const execResponse = await fetch(
        `${API_BASE_URL}/execute-pipeline-from-file?config_name=${configName}`,
        { method: 'POST' }
    );
    if (!execResponse.ok) {
        const err = await execResponse.json();
        throw new Error(err.detail || 'Execution trigger failed');
    }
    const { progress_id: progressId } = await execResponse.json();
    return _streamSSEProgress(progressId, configName, onProgress);
}

export const searchData = async (query, configName, topK = 5) => {
    const params = new URLSearchParams();
    params.append('query', query);
    params.append('config_name', configName);
    params.append('top_k', topK);

    const response = await fetch(`${API_BASE_URL}/unified-search?${params.toString()}`, {
        method: 'POST'
    });
    if (!response.ok) {
        let detail = 'Search failed';
        try { detail = (await response.json()).detail || detail; } catch {}
        throw new Error(detail);
    }
    const data = await response.json();
    return {
        results: (data.results || []).map(r => ({
            id: r.id,
            score: r.score ?? r.similarity_score,
            content: r.content ?? r.text,
            metadata: { source: r.embedding_model, model: r.embedding_model }
        }))
    };
};

/**
 * Stream summary tokens from the backend.
 * onToken(type, text) — type is 'think' | 'answer' | 'think_start' | 'think_end'
 * Returns a cancel function.
 */
export const generateSummaryStream = (query, chunks, modelName = 'qwen3-0.6b', systemPrompt = '', maxTokens = 1200, { onToken, onDone, onError } = {}) => {
    const params = new URLSearchParams({ query, model_name: modelName, max_tokens: maxTokens });
    if (systemPrompt) params.append('system_prompt', systemPrompt);

    let cancelled = false;
    const controller = new AbortController();

    fetch(`${API_BASE_URL}/generate-summary-stream?${params.toString()}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(chunks),
        signal: controller.signal,
    }).then(async (res) => {
        if (!res.ok) { onError?.(new Error('Stream request failed')); return; }
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buf = '';
        while (!cancelled) {
            const { done, value } = await reader.read();
            if (done) { onDone?.(); break; }
            buf += decoder.decode(value, { stream: true });
            const lines = buf.split('\n');
            buf = lines.pop();
            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                try {
                    const evt = JSON.parse(line.slice(6));
                    if (evt.type === 'done') { onDone?.(); return; }
                    if (evt.type === 'error') { onError?.(new Error(evt.message)); return; }
                    onToken?.(evt.type, evt.token ?? '');
                } catch (_) {}
            }
        }
    }).catch((err) => { if (err.name !== 'AbortError') onError?.(err); });

    return () => { cancelled = true; controller.abort(); };
};

export const generateSummary = async (query, chunks, modelName = 'qwen3-0.6b', systemPrompt = '', maxTokens = 150) => {
    const params = new URLSearchParams();
    params.append('query', query);
    params.append('model_name', modelName);
    if (systemPrompt) params.append('system_prompt', systemPrompt);
    params.append('max_tokens', maxTokens);

    const response = await fetch(`${API_BASE_URL}/generate-summary?${params.toString()}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(chunks),
    });
    if (!response.ok) throw new Error('Summary generation failed');
    const data = await response.json();
    return data.summary || null;
};

export const getAvailableOptions = async () => {
    const response = await fetch(`${API_BASE_URL}/api/config-options`);
    if (!response.ok) throw new Error("Failed to fetch options");
    return await response.json();
};

export const configureDbPipeline = async (payload) => {
    const params = new URLSearchParams();
    params.append('source_db_host', payload.source.dbHost);
    params.append('source_db_port', payload.source.dbPort);
    params.append('source_db_user', payload.source.dbUser);
    params.append('source_db_password', payload.source.dbPassword);
    params.append('source_db_name', payload.source.dbName);
    params.append('source_db_table', payload.source.dbTable);
    params.append('source_chunk_column', payload.source.chunkColumn);
    params.append('source_id_column', payload.source.idColumn);

    if (payload.target) {
        params.append('target_db_host', payload.target.dbHost);
        params.append('target_db_port', payload.target.dbPort);
        params.append('target_db_user', payload.target.dbUser);
        params.append('target_db_password', payload.target.dbPassword);
        params.append('target_db_name', payload.target.dbName);
        params.append('target_db_table', payload.target.dbTable);
    }

    const modelValue = payload.pipeline.embeddingModel === 'other'
        ? payload.pipeline.embeddingModelManual
        : payload.pipeline.embeddingModel;
    params.append('embedding_model', modelValue || "BAAI/bge-m3");

    if (payload.pipeline.embeddingDimensions) {
        params.append('vector_dim', payload.pipeline.embeddingDimensions);
    }
    if (payload.pipeline.huggingfaceToken) {
        params.append('hf_token', payload.pipeline.huggingfaceToken);
    }

    params.append('index_type', payload.pipeline.indexType || "hnsw");

    if (payload.pipeline.incrementalMode !== undefined) {
        params.append('incremental_mode', payload.pipeline.incrementalMode);
    }

    const response = await fetch(`${API_BASE_URL}/configure-db-source-pipeline?${params.toString()}`, {
        method: 'POST'
    });
    if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'DB Configuration failed');
    }
    return await response.json();
};

/**
 * Configure + execute the pre-embedded pipeline (source table already has vectors).
 * Skips chunking and re-embedding — builds a FAISS index only.
 */
export const executePreembeddedPipeline = async (payload, onProgress, onStart = null) => {
    const { dbHost, dbPort, dbUser, dbPassword, dbName, tableName,
            chunkColumn, embeddingColumn, idColumn, pipeline } = payload;

    const embeddingModel = pipeline.overrides.embeddingModel.enabled
        ? (pipeline.overrides.embeddingModel.value === 'other'
            ? pipeline.overrides.embeddingModelManual?.value
            : pipeline.overrides.embeddingModel.value)
        : 'BAAI/bge-m3';

    const indexType = pipeline.overrides.retrievalMethod.enabled
        ? pipeline.overrides.retrievalMethod.value
        : 'hnsw';

    const params = new URLSearchParams({
        source_db_host: dbHost,
        source_db_port: dbPort,
        source_db_user: dbUser,
        source_db_password: dbPassword,
        source_db_name: dbName,
        source_db_table: tableName,
        text_column: chunkColumn,
        vector_column: embeddingColumn,
        id_column: idColumn || 'id',
        embedding_model: embeddingModel || 'BAAI/bge-m3',
        index_type: indexType,
    });

    const configResponse = await fetch(
        `${API_BASE_URL}/configure-preembedded-pipeline?${params.toString()}`,
        { method: 'POST' }
    );
    if (!configResponse.ok) {
        const err = await configResponse.json().catch(() => ({}));
        throw new Error(err.detail || 'Pre-embedded configuration failed');
    }
    const configResult = await configResponse.json();
    const configName = configResult.config_name;
    const progressId = configResult.progress_id;

    if (onStart) onStart({ configName });

    return new Promise((resolve, reject) => {
        let executionTriggered = false;

        _streamSSEProgress(
            progressId,
            configName,
            onProgress,
            async () => {
                if (executionTriggered) return;
                executionTriggered = true;
                try {
                    const execResponse = await fetch(
                        `${API_BASE_URL}/execute-preembedded-pipeline?config_name=${configName}&progress_id=${progressId}`,
                        { method: 'POST' }
                    );
                    if (!execResponse.ok) {
                        const err = await execResponse.json().catch(() => ({}));
                        throw new Error(err.detail || 'Pre-embedded execution trigger failed');
                    }
                } catch (e) {
                    reject(e);
                }
            }
        ).then(resolve).catch(reject);
    });
};

export const getSourceDbColumns = async (dbConfig) => {
    // Credentials sent in JSON body — never in query string (would appear in server logs)
    const response = await fetch(`${API_BASE_URL}/get-source-db-columns`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            db_host: dbConfig.dbHost || '',
            db_port: parseInt(dbConfig.dbPort) || 5432,
            db_user: dbConfig.dbUser || '',
            db_password: dbConfig.dbPassword || '',
            db_name: dbConfig.dbName || '',
            db_table: dbConfig.tableName || '',
        }),
    });
    if (!response.ok) {
        let errStr = "Failed to fetch columns";
        try {
            const errJson = await response.json();
            const detail = errJson.detail;
            if (Array.isArray(detail)) {
                // FastAPI validation error — detail is [{loc, msg, type}, ...]
                errStr = detail.map(e => e.msg || JSON.stringify(e)).join('; ');
            } else if (typeof detail === 'string') {
                // Strip psycopg2 tuple wrapper e.g. "('message',)"
                const match = detail.match(/'([^']+)'/);
                errStr = (match && match[1]) ? match[1] : detail;
            }
        } catch (e) { /* response was not JSON */ }
        throw new Error(errStr);
    }
    return await response.json();
};

export const checkTableExists = async (dbConfig, tableName) => {
    const params = new URLSearchParams({
        db_host: dbConfig.dbHost || 'localhost',
        db_port: parseInt(dbConfig.dbPort) || 5433,
        db_user: dbConfig.dbUser || 'postgres',
        db_password: dbConfig.dbPassword || 'postgres',
        db_name: dbConfig.dbName || 'appdb',
        table_name: tableName
    });

    const response = await fetch(`${API_BASE_URL}/api/check-table-exists?${params.toString()}`, {
        method: 'POST'
    });
    if (!response.ok) throw new Error("Table check failed");
    return await response.json();
};

export const inspectFile = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch(`${API_BASE_URL}/api/inspect-file`, {
        method: 'POST',
        body: formData
    });
    if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || 'File inspection failed');
    }
    return await response.json();
};

export const cancelPipeline = async (configName) => {
    const response = await fetch(`${API_BASE_URL}/api/cancel-pipeline?config_name=${encodeURIComponent(configName)}`, {
        method: 'POST'
    });
    if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || 'Failed to cancel pipeline');
    }
    return await response.json();
};

export const exportConfiguration = async (configName) => {
    const response = await fetch(`${API_BASE_URL}/export-config?config_name=${configName}`, {
        method: 'POST'
    });
    if (!response.ok) throw new Error("Failed to export configuration");
    return response.blob();
};

export const importConfiguration = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch(`${API_BASE_URL}/import-config`, {
        method: 'POST',
        body: formData
    });
    if (!response.ok) throw new Error("Failed to import configuration");
    return await response.json();
};

export const listConfigs = async () => {
    const response = await fetch(`${API_BASE_URL}/list-configs`);
    if (!response.ok) throw new Error("Failed to list configurations");
    return await response.json();
};

export const getConfigDetails = async (configName) => {
    try {
        const response = await fetch(`${API_BASE_URL}/export-config?config_name=${configName}`, {
            method: 'POST'
        });
        if (!response.ok) throw new Error("Failed to get config details");
        const blob = await response.blob();
        const text = await blob.text();
        return JSON.parse(text);
    } catch (error) {
        return null;
    }
};

export const ingestData = async (_data) => {
    return { success: true };
};

export const saveDatabaseConfig = async (_config) => {
    return { success: true };
};

export const fetchPipelineRecommendations = async (ingestionData) => {
    const { dataSourceType, file, dataFormat } = ingestionData;

    // Determine chunking strategy
    let chunkingStrategy = { value: 'fixed_size', reason: 'Default for general text data.' };
    if (dataFormat === 'json' || dataFormat === 'csv') {
        chunkingStrategy = { value: 'none', reason: 'Structured data — each row is already a discrete chunk.' };
    } else if (dataFormat === 'pdf' || dataFormat === 'docx') {
        chunkingStrategy = { value: 'sentence', reason: 'Document format benefits from sentence-aware splitting.' };
    } else if (dataSourceType === 'database') {
        chunkingStrategy = { value: 'none', reason: 'DB rows are pre-chunked — no further splitting needed.' };
    }

    // Determine embedding model based on file size
    let embeddingModel = { value: 'BAAI/bge-m3', reason: 'Best general-purpose multilingual embeddings.' };
    if (file && file.size < 5 * 1024 * 1024) {
        // Small file — lightweight model is fine
        embeddingModel = { value: 'nomic-ai/nomic-embed-text-v1', reason: 'Fast and efficient for small datasets.' };
    } else if (file && file.size > 50 * 1024 * 1024) {
        // Large file — stick with high-quality model
        embeddingModel = { value: 'BAAI/bge-m3', reason: 'High quality needed for large diverse datasets.' };
    }

    // Determine retrieval method based on estimated row count / file size
    let retrievalMethod = { value: 'hnsw', reason: 'Efficient approximate search for most dataset sizes.' };
    if (file && file.size < 1 * 1024 * 1024) {
        retrievalMethod = { value: 'flat', reason: 'Small dataset — exact flat search is fast enough.' };
    }

    return { chunkingStrategy, embeddingModel, retrievalMethod };
};

// ── NEW FEATURE APIs ─────────────────────────────────────────────────────────

export const fetchLogs = async (lines = 200) => {
    const response = await fetch(`${API_BASE_URL}/api/logs?lines=${lines}`);
    if (!response.ok) throw new Error('Failed to fetch logs');
    return response.json();
};

export const submitFeedback = async ({ query, resultId, contentSnippet, vote, configName }) => {
    const response = await fetch(`${API_BASE_URL}/api/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            query,
            result_id: resultId,
            content_snippet: contentSnippet,
            vote,
            config_name: configName,
        }),
    });
    if (!response.ok) throw new Error('Failed to submit feedback');
    return response.json();
};

export const fetchChunkPreview = async ({ text, chunkSize, chunkOverlap, strategy }) => {
    const response = await fetch(`${API_BASE_URL}/api/chunk-preview`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            text,
            chunk_size: chunkSize,
            chunk_overlap: chunkOverlap,
            strategy,
        }),
    });
    if (!response.ok) throw new Error('Failed to fetch chunk preview');
    return response.json();
};

export const checkDuplicates = async ({ dbHost, dbPort, dbUser, dbPassword, dbName, tableName }) => {
    const response = await fetch(`${API_BASE_URL}/api/check-duplicates`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            db_host: dbHost,
            db_port: dbPort,
            db_user: dbUser,
            db_password: dbPassword,
            db_name: dbName,
            table_name: tableName,
        }),
    });
    if (!response.ok) throw new Error('Failed to check duplicates');
    return response.json();
};
