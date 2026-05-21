import React, { useCallback, useRef, useState, useEffect } from 'react';
import { inspectFile, getSourceDbColumns } from '../../../services/api';
import './DataIngestionStep.css';

const STRUCTURED_FORMATS = ['.csv', '.xlsx', '.parquet', '.json'];

const FORMAT_ACCEPT = {
    'TXT':     '.txt',
    'CSV':     '.csv',
    'JSON':    '.json',
    'Parquet': '.parquet',
    'Excel':   '.xlsx',
    'Word':    '.docx',
};

const FORMAT_EXT = {
    'TXT':     ['.txt'],
    'CSV':     ['.csv'],
    'JSON':    ['.json'],
    'Parquet': ['.parquet'],
    'Excel':   ['.xlsx'],
    'Word':    ['.docx'],
};

const DataIngestionStep = ({
    dataSourceType,
    dataFormat,
    domain,
    dataSize,
    file,
    textColumn,
    embeddingColumn,
    dbHost,
    dbPort,
    dbUser,
    dbPassword,
    dbName,
    tableName,
    columnNames,
    chunkColumn,
    idColumn,
    tableNameExists,
    schema,
    errors = {},
    onChange,
    onFetchColumns,
    onSubmit,
    onImportConfig
}) => {
    const fileInputRef = useRef(null);
    const importInputRef = useRef(null);

    // Schema discovery state — local to this component
    const [schemaLoading, setSchemaLoading] = useState(false);
    const [schemaInfo, setSchemaInfo] = useState(null); // { columns, potential_embeddings, sample, has_structure }
    const [schemaError, setSchemaError] = useState(null);

    // DB column fetch state
    const [dbFetchLoading, setDbFetchLoading] = useState(false);
    const [dbFetchError, setDbFetchError] = useState(null);
    const [fetchedDbColumns, setFetchedDbColumns] = useState([]); // [{name, type}, ...]

    // Auto-inspect whenever a structured file is selected
    useEffect(() => {
        if (!file) {
            setSchemaInfo(null);
            setSchemaError(null);
            return;
        }
        const ext = '.' + file.name.split('.').pop().toLowerCase();
        if (!STRUCTURED_FORMATS.includes(ext)) {
            setSchemaInfo(null);
            setSchemaError(null);
            return;
        }
        // Clear old selections
        onChange('textColumn', '');
        onChange('embeddingColumn', '');
        runInspect(file);
    }, [file]);

    const runInspect = async (f) => {
        setSchemaLoading(true);
        setSchemaError(null);
        setSchemaInfo(null);
        try {
            const result = await inspectFile(f);
            setSchemaInfo(result);
            // Auto-select first non-embedding text column
            if (result.has_structure && result.columns.length > 0) {
                const firstText = result.columns.find(c => !result.potential_embeddings.includes(c));
                if (firstText) onChange('textColumn', firstText);
            }
        } catch (err) {
            setSchemaError(err.message || 'Failed to inspect file schema.');
        } finally {
            setSchemaLoading(false);
        }
    };

    // ── DB Column Fetch ───────────────────────────────────
    const handleFetchDbColumns = async () => {
        setDbFetchLoading(true);
        setDbFetchError(null);
        try {
            const result = await getSourceDbColumns({ dbHost, dbPort, dbUser, dbPassword, dbName, tableName });
            const cols = result.columns || [];
            setFetchedDbColumns(cols);
            // Store as text for parent
            onChange('columnNames', cols.map(c => c.name).join(', '));
            // Auto-select 'id' for idColumn if it exists
            const idCol = cols.find(c => c.name === 'id');
            if (idCol && !idColumn) onChange('idColumn', 'id');
        } catch (err) {
            setDbFetchError(err.message || 'Failed to fetch columns.');
            setFetchedDbColumns([]);
        } finally {
            setDbFetchLoading(false);
        }
    };

    // ── Drag & Drop handlers ──────────────────────────────
    const handleDragOver = (e) => {
        e.preventDefault(); e.stopPropagation();
        e.currentTarget.classList.add('drag-active');
    };
    const handleDragLeave = (e) => {
        e.preventDefault(); e.stopPropagation();
        e.currentTarget.classList.remove('drag-active');
    };
    const validateFileFormat = (uploadedFile) => {
        if (!dataFormat) return true; // no format selected yet — allow any
        const allowed = FORMAT_EXT[dataFormat] || [];
        if (allowed.length === 0) return true;
        const ext = '.' + uploadedFile.name.split('.').pop().toLowerCase();
        return allowed.includes(ext);
    };

    const handleDrop = (e) => {
        e.preventDefault(); e.stopPropagation();
        e.currentTarget.classList.remove('drag-active');
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            const uploadedFile = e.dataTransfer.files[0];
            if (uploadedFile.size > 100 * 1024 * 1024) { alert('File size exceeds 100MB limit.'); return; }
            if (!validateFileFormat(uploadedFile)) {
                alert(`Selected format is "${dataFormat}". Please upload a ${FORMAT_ACCEPT[dataFormat]} file.`);
                return;
            }
            onChange('file', uploadedFile);
        }
    };
    const handleFileSelect = (e) => {
        if (e.target.files && e.target.files[0]) {
            const uploadedFile = e.target.files[0];
            if (uploadedFile.size > 100 * 1024 * 1024) {
                alert('File size exceeds 100MB limit.');
                if (fileInputRef.current) fileInputRef.current.value = '';
                return;
            }
            if (!validateFileFormat(uploadedFile)) {
                alert(`Selected format is "${dataFormat}". Please upload a ${FORMAT_ACCEPT[dataFormat]} file.`);
                if (fileInputRef.current) fileInputRef.current.value = '';
                return;
            }
            onChange('file', uploadedFile);
        }
    };
    const handleImportFileSelect = (e) => {
        if (e.target.files && e.target.files[0]) onImportConfig(e.target.files[0]);
    };
    const triggerFileInput = () => fileInputRef.current.click();
    const removeFile = (e) => {
        e.stopPropagation();
        onChange('file', null);
        if (fileInputRef.current) fileInputRef.current.value = '';
    };
    const formatFileSize = (bytes) => {
        if (bytes === 0) return '0 Bytes';
        const k = 1024, sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    };

    const isDatabaseValid = dbHost && dbPort && dbUser && dbPassword && dbName && tableName && columnNames && chunkColumn && idColumn;
    const isFormValid = dataSourceType === 'manual'
        ? (dataFormat && domain && dataSize && file && (!schemaInfo?.has_structure || textColumn))
        : (domain && dataSize && isDatabaseValid);

    // column type badge helper
    const getColBadge = (col) => {
        if (schemaInfo?.potential_embeddings?.includes(col)) return 'vector';
        return 'text';
    };

    return (
        <div className="data-ingestion-step">
            {/* ── Data Source Type ── */}
            <div className="form-group">
                <label htmlFor="dataSourceType">Data Source</label>
                <select
                    id="dataSourceType"
                    className="form-select"
                    value={dataSourceType}
                    onChange={(e) => onChange('dataSourceType', e.target.value)}
                >
                    <option value="manual">Manual Upload</option>
                    <option value="database">Existing Database</option>
                </select>
            </div>

            {/* ── FILE UPLOAD PATH ── */}
            {dataSourceType === 'manual' && (
                <>
                    <div className="form-group">
                        <label htmlFor="dataFormat">Data Format</label>
                        <select
                            id="dataFormat"
                            className="form-select"
                            value={dataFormat}
                            onChange={(e) => {
                                const newFormat = e.target.value;
                                onChange('dataFormat', newFormat);
                                // Clear file if it no longer matches the new format
                                if (file && newFormat && FORMAT_EXT[newFormat]) {
                                    const ext = '.' + file.name.split('.').pop().toLowerCase();
                                    if (!FORMAT_EXT[newFormat].includes(ext)) {
                                        onChange('file', null);
                                        if (fileInputRef.current) fileInputRef.current.value = '';
                                    }
                                }
                            }}
                        >
                            <option value="">Select Format</option>
                            <option value="TXT">TXT</option>
                            <option value="CSV">CSV</option>
                            <option value="JSON">JSON</option>
                            <option value="Parquet">Parquet</option>
                            <option value="Excel">Excel (XLSX)</option>
                            <option value="Word">Word (DOCX)</option>
                        </select>
                    </div>

                    <div className="form-group">
                        <label htmlFor="manualTableName">Target Table Name <span className="required-star">*</span></label>
                        <p className="schema-selector-hint">The database table where your embeddings will be stored</p>
                        <input
                            type="text"
                            id="manualTableName"
                            className={`form-input ${errors.tableName || tableNameExists ? 'error' : ''}`}
                            placeholder="e.g. my_documents_table"
                            value={tableName}
                            onChange={(e) => onChange('tableName', e.target.value)}
                        />
                        {tableNameExists && (
                            <p className="warning-message" style={{ color: '#f59e0b', fontSize: '0.85rem', marginTop: '4px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0zM12 9v4M12 17h.01" />
                                </svg>
                                Warning: A table with this name already exists in the database.
                            </p>
                        )}
                        {errors.tableName && <p className="error-message">{errors.tableName}</p>}
                    </div>

                    <div className="form-group">
                        <label>Upload Dataset</label>
                        {!file ? (
                            <>
                                <div
                                    className={`drop-zone ${errors.file ? 'error' : ''}`}
                                    onDragOver={handleDragOver}
                                    onDragLeave={handleDragLeave}
                                    onDrop={handleDrop}
                                    onClick={triggerFileInput}
                                >
                                    <div className="drop-zone-content">
                                        <div className="upload-icon-wrapper">
                                            <svg className="upload-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                                            </svg>
                                        </div>
                                        <div className="upload-text-container">
                                            <p className="upload-main-text">
                                                <span>Click to upload</span> or drag and drop
                                            </p>
                                            <p className="upload-sub-text">Supported: TXT, CSV, JSON, Parquet, XLSX, DOCX (Max 100MB)</p>
                                        </div>
                                    </div>
                                    <input
                                        type="file"
                                        ref={fileInputRef}
                                        style={{ display: 'none' }}
                                        accept={dataFormat && FORMAT_ACCEPT[dataFormat] ? FORMAT_ACCEPT[dataFormat] : '.txt,.csv,.json,.parquet,.xlsx,.docx'}
                                        onChange={handleFileSelect}
                                    />
                                </div>
                                {errors.file && <p className="error-message">{errors.file}</p>}
                            </>
                        ) : (
                            <div className="file-info">
                                <svg className="file-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                </svg>
                                <div className="file-details">
                                    <span className="file-name">{file.name}</span>
                                    <span className="file-size">{formatFileSize(file.size)}</span>
                                </div>
                                <button className="remove-file" onClick={removeFile} aria-label="Remove file">
                                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <path d="M6 18L18 6M6 6l12 12" strokeLinecap="round" strokeLinejoin="round" />
                                    </svg>
                                </button>
                            </div>
                        )}
                    </div>

                    {/* ── SCHEMA DISCOVERY PANEL ── */}
                    {file && schemaLoading && (
                        <div className="schema-discovery-panel schema-loading">
                            <div className="schema-loading-inner">
                                <div className="schema-spinner" />
                                <span>Detecting file schema…</span>
                            </div>
                        </div>
                    )}

                    {file && schemaError && (
                        <div className="schema-discovery-panel schema-error">
                            <div className="schema-error-header">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" />
                                </svg>
                                Schema detection failed — file will be ingested as raw text.
                            </div>
                        </div>
                    )}

                    {file && schemaInfo && schemaInfo.has_structure && !schemaLoading && (
                        <div className="schema-discovery-panel">
                            {/* Header */}
                            <div className="schema-panel-header">
                                <div className="schema-panel-title">
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <rect x="3" y="3" width="18" height="18" rx="2" /><path d="M3 9h18M9 21V9" />
                                    </svg>
                                    Schema Detected — {schemaInfo.format} · {schemaInfo.columns.length} columns
                                </div>
                                {schemaInfo.potential_embeddings?.length > 0 && (
                                    <div className="schema-embed-badge">
                                        ⚡ {schemaInfo.potential_embeddings.length} embedding column{schemaInfo.potential_embeddings.length > 1 ? 's' : ''} found
                                    </div>
                                )}
                            </div>

                            {/* Column Configuration Grid */}
                            <div className="schema-cols-label">Column Configuration</div>
                            <p className="schema-selector-hint" style={{ marginBottom: '12px' }}>
                                Define the data types for columns you wish to ingest. Primary key and embeddings are handled automatically.
                            </p>
                            
                            <div className="schema-config-table-wrapper">
                                <table className="schema-config-table">
                                    <thead>
                                        <tr>
                                            <th>Column Name</th>
                                            <th>Data Type</th>
                                            <th>Action</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {schemaInfo.columns.map(col => {
                                            const isIngested = schema[col] !== undefined;
                                            const currentType = schema[col] || 'TEXT';
                                            const isEmbedSource = textColumn === col;
                                            const isExistingEmbed = embeddingColumn === col;

                                            return (
                                                <tr key={col} className={`${isIngested ? 'active-row' : ''} ${isEmbedSource ? 'source-row' : ''}`}>
                                                    <td>
                                                        <div className="col-name-wrapper">
                                                            <span className={`col-indicator ${getColBadge(col)}`} />
                                                            <span className="col-name-text">{col}</span>
                                                            {isEmbedSource && <span className="col-tag-content">CONTENT</span>}
                                                            {isExistingEmbed && <span className="col-tag-vector">VECTOR</span>}
                                                        </div>
                                                    </td>
                                                    <td>
                                                        <select 
                                                            className="schema-type-select"
                                                            value={currentType}
                                                            onChange={(e) => {
                                                                const newSchema = { ...schema, [col]: e.target.value };
                                                                onChange('schema', newSchema);
                                                            }}
                                                            disabled={!isIngested}
                                                        >
                                                            <option value="TEXT">TEXT</option>
                                                            <option value="INTEGER">INTEGER</option>
                                                            <option value="FLOAT">FLOAT</option>
                                                            <option value="BOOLEAN">BOOLEAN</option>
                                                            <option value="TIMESTAMP">TIMESTAMP</option>
                                                            <option value="JSONB">JSONB</option>
                                                        </select>
                                                    </td>
                                                    <td>
                                                        <label className="switch">
                                                            <input 
                                                                type="checkbox" 
                                                                checked={isIngested}
                                                                onChange={(e) => {
                                                                    const newSchema = { ...schema };
                                                                    if (e.target.checked) {
                                                                        newSchema[col] = 'TEXT';
                                                                    } else {
                                                                        delete newSchema[col];
                                                                    }
                                                                    onChange('schema', newSchema);
                                                                }}
                                                            />
                                                            <span className="slider round"></span>
                                                        </label>
                                                    </td>
                                                </tr>
                                            );
                                        })}
                                    </tbody>
                                </table>
                            </div>

                            {/* Column selectors */}
                            <div className="schema-selectors">
                                <div className="schema-selector-group">
                                    <label htmlFor="textColumnSelect" className="schema-selector-label">
                                        <span className="selector-dot text-dot" />
                                        Text Column <span className="required-star">*</span>
                                    </label>
                                    <p className="schema-selector-hint">Column to embed and index for search</p>
                                    <select
                                        id="textColumnSelect"
                                        className={`form-select schema-select ${!textColumn ? 'highlight' : ''}`}
                                        value={textColumn || ''}
                                        onChange={(e) => onChange('textColumn', e.target.value)}
                                    >
                                        <option value="">— Select Text Column —</option>
                                        {schemaInfo.columns.filter(c => !schemaInfo.potential_embeddings?.includes(c)).map(col => (
                                            <option key={col} value={col}>{col}</option>
                                        ))}
                                        {schemaInfo.columns.filter(c => schemaInfo.potential_embeddings?.includes(c)).length > 0 && (
                                            <optgroup label="Vector Columns (not recommended)">
                                                {schemaInfo.columns.filter(c => schemaInfo.potential_embeddings?.includes(c)).map(col => (
                                                    <option key={col} value={col}>{col} (vector)</option>
                                                ))}
                                            </optgroup>
                                        )}
                                    </select>
                                </div>

                                <div className="schema-selector-group">
                                    <label htmlFor="embeddingColumnSelect" className="schema-selector-label">
                                        <span className="selector-dot embed-dot" />
                                        Pre-existing Embeddings <span className="optional-tag">optional</span>
                                    </label>
                                    <p className="schema-selector-hint">Skip re-embedding by using an existing vector column</p>
                                    <select
                                        id="embeddingColumnSelect"
                                        className="form-select schema-select"
                                        value={embeddingColumn || ''}
                                        onChange={(e) => onChange('embeddingColumn', e.target.value || null)}
                                    >
                                        <option value="">— None (generate new embeddings) —</option>
                                        {schemaInfo.potential_embeddings?.map(col => (
                                            <option key={col} value={col}>{col}</option>
                                        ))}
                                        {schemaInfo.columns.filter(c => !schemaInfo.potential_embeddings?.includes(c)).length > 0 && (
                                            <optgroup label="Other columns">
                                                {schemaInfo.columns.filter(c => !schemaInfo.potential_embeddings?.includes(c)).map(col => (
                                                    <option key={col} value={col}>{col}</option>
                                                ))}
                                            </optgroup>
                                        )}
                                    </select>
                                    {embeddingColumn && (
                                        <div className="embed-reuse-badge">
                                            ✓ Will reuse embeddings from <strong>{embeddingColumn}</strong> — skipping embedding model
                                        </div>
                                    )}
                                </div>
                            </div>

                            {/* Sample preview */}
                            {schemaInfo.sample && schemaInfo.sample.length > 0 && (
                                <details className="schema-sample">
                                    <summary className="schema-sample-title">Preview first row</summary>
                                    <div className="schema-sample-rows">
                                        {Object.entries(schemaInfo.sample[0]).map(([k, v]) => (
                                            <div key={k} className="schema-sample-row">
                                                <span className="sample-key">{k}</span>
                                                <span className="sample-val">{String(v).substring(0, 120)}{String(v).length > 120 ? '…' : ''}</span>
                                            </div>
                                        ))}
                                    </div>
                                </details>
                            )}
                        </div>
                    )}
                </>
            )}

            {/* ── DATABASE PATH ── */}
            {dataSourceType === 'database' && (
                <div className="database-source-fields">
                    <div className="form-group">
                        <label htmlFor="dbHost">Database Host</label>
                        <input type="text" id="dbHost" className={`form-input ${errors.dbHost ? 'error' : ''}`}
                            placeholder="localhost" value={dbHost} onChange={(e) => onChange('dbHost', e.target.value)} />
                        {errors.dbHost && <p className="error-message">{errors.dbHost}</p>}
                    </div>
                    <div className="form-group">
                        <label htmlFor="dbPort">Database Port</label>
                        <input type="text" id="dbPort" className={`form-input ${errors.dbPort ? 'error' : ''}`}
                            placeholder="5432" value={dbPort} onChange={(e) => onChange('dbPort', e.target.value)} />
                        {errors.dbPort && <p className="error-message">{errors.dbPort}</p>}
                    </div>
                    <div className="form-group">
                        <label htmlFor="dbUser">Username</label>
                        <input type="text" id="dbUser" className={`form-input ${errors.dbUser ? 'error' : ''}`}
                            placeholder="postgres" value={dbUser} onChange={(e) => onChange('dbUser', e.target.value)} />
                        {errors.dbUser && <p className="error-message">{errors.dbUser}</p>}
                    </div>
                    <div className="form-group">
                        <label htmlFor="dbPassword">Password</label>
                        <input type="password" id="dbPassword" className={`form-input ${errors.dbPassword ? 'error' : ''}`}
                            placeholder="••••••••" value={dbPassword} onChange={(e) => onChange('dbPassword', e.target.value)} />
                        {errors.dbPassword && <p className="error-message">{errors.dbPassword}</p>}
                    </div>
                    <div className="form-group">
                        <label htmlFor="dbName">Database Name</label>
                        <input type="text" id="dbName" className={`form-input ${errors.dbName ? 'error' : ''}`}
                            placeholder="source_db" value={dbName} onChange={(e) => onChange('dbName', e.target.value)} />
                        {errors.dbName && <p className="error-message">{errors.dbName}</p>}
                    </div>
                    <div className="form-group">
                        <label htmlFor="tableName">Table Name</label>
                        <input type="text" id="tableName" className={`form-input ${errors.tableName ? 'error' : ''}`}
                            placeholder="raw_data" value={tableName} onChange={(e) => onChange('tableName', e.target.value)} />
                        {errors.tableName && <p className="error-message">{errors.tableName}</p>}
                    </div>
                    {/* Fetch Columns */}
                    <div className="form-group">
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                            <label style={{ marginBottom: 0 }}>
                                Table Columns
                                {fetchedDbColumns.length > 0 && (
                                    <span style={{ marginLeft: '8px', fontSize: '0.75rem', color: '#10b981', fontWeight: 500 }}>
                                        ✓ {fetchedDbColumns.length} columns loaded
                                    </span>
                                )}
                            </label>
                            <button
                                type="button"
                                className="btn-primary"
                                onClick={handleFetchDbColumns}
                                disabled={dbFetchLoading}
                                style={{ minWidth: '80px' }}
                            >
                                {dbFetchLoading ? 'Fetching…' : 'Fetch Columns'}
                            </button>
                        </div>
                        {dbFetchError && (
                            <p className="error-message" style={{ marginTop: '4px' }}>{dbFetchError}</p>
                        )}
                        {fetchedDbColumns.length > 0 && (
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', padding: '8px', background: '#f8fafc', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                                {fetchedDbColumns.map(col => (
                                    <span key={col.name} style={{ fontSize: '0.78rem', padding: '2px 8px', borderRadius: '12px', background: '#eef2ff', color: '#4f46e5', border: '1px solid #c7d2fe', fontFamily: 'monospace' }}>
                                        {col.name}
                                        <span style={{ color: '#94a3b8', marginLeft: '4px' }}>({col.type?.split(' ')[0]})</span>
                                    </span>
                                ))}
                            </div>
                        )}
                    </div>

                    {/* ID Column */}
                    <div className="form-group">
                        <label htmlFor="idColumn">ID Column <span style={{ color: '#ef4444' }}>*</span></label>
                        {fetchedDbColumns.length > 0 ? (
                            <select
                                id="idColumn"
                                className={`form-input ${errors.idColumn ? 'error' : ''}`}
                                value={idColumn}
                                onChange={(e) => onChange('idColumn', e.target.value)}
                            >
                                <option value="">— Select ID column —</option>
                                {fetchedDbColumns.map(col => (
                                    <option key={col.name} value={col.name}>{col.name} ({col.type?.split(' ')[0]})</option>
                                ))}
                            </select>
                        ) : (
                            <input type="text" id="idColumn" className={`form-input ${errors.idColumn ? 'error' : ''}`}
                                placeholder="id" value={idColumn} onChange={(e) => onChange('idColumn', e.target.value)} />
                        )}
                        {errors.idColumn && <p className="error-message">{errors.idColumn}</p>}
                    </div>

                    {/* Text Column */}
                    <div className="form-group">
                        <label htmlFor="chunkColumn">Text Column <span style={{ color: '#ef4444' }}>*</span></label>
                        {fetchedDbColumns.length > 0 ? (
                            <select
                                id="chunkColumn"
                                className={`form-input ${errors.chunkColumn ? 'error' : ''}`}
                                value={chunkColumn}
                                onChange={(e) => onChange('chunkColumn', e.target.value)}
                            >
                                <option value="">— Select text column —</option>
                                {fetchedDbColumns.map(col => (
                                    <option key={col.name} value={col.name}>{col.name} ({col.type?.split(' ')[0]})</option>
                                ))}
                            </select>
                        ) : (
                            <input type="text" id="chunkColumn" className={`form-input ${errors.chunkColumn ? 'error' : ''}`}
                                placeholder="content" value={chunkColumn} onChange={(e) => onChange('chunkColumn', e.target.value)} />
                        )}
                        {errors.chunkColumn && <p className="error-message">{errors.chunkColumn}</p>}
                    </div>

                    {/* Vector Column (optional) */}
                    <div className="form-group">
                        <label htmlFor="embeddingColumn">
                            Vector Column <span style={{ fontSize: '0.75rem', color: '#6b7280' }}>(optional — leave blank to generate embeddings)</span>
                        </label>
                        {fetchedDbColumns.length > 0 ? (
                            <select
                                id="embeddingColumn"
                                className="form-input"
                                value={embeddingColumn || ''}
                                onChange={(e) => onChange('embeddingColumn', e.target.value || null)}
                            >
                                <option value="">— None (generate embeddings) —</option>
                                {fetchedDbColumns.map(col => (
                                    <option key={col.name} value={col.name}>{col.name} ({col.type?.split(' ')[0]})</option>
                                ))}
                            </select>
                        ) : (
                            <input type="text" id="embeddingColumn" className="form-input"
                                placeholder="embedding"
                                value={embeddingColumn || ''}
                                onChange={(e) => onChange('embeddingColumn', e.target.value || null)} />
                        )}
                        {embeddingColumn && (
                            <div className="preembedded-notice" style={{ marginTop: '0.5rem', marginBottom: 0 }}>
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <circle cx="12" cy="12" r="10"/><path d="M9 12l2 2 4-4"/>
                                </svg>
                                <span>Table already has vectors — chunking and re-embedding will be skipped. Only the FAISS index will be built.</span>
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* ── Domain & Size (always shown) ── */}
            <div className="form-group">
                <label htmlFor="domain">Domain Type</label>
                <select id="domain" className="form-select" value={domain} onChange={(e) => onChange('domain', e.target.value)}>
                    <option value="">Select Domain</option>
                    <option value="Medical">Medical</option>
                    <option value="Legal">Legal</option>
                    <option value="Finance">Finance</option>
                    <option value="General">General</option>
                </select>
            </div>
            <div className="form-group">
                <label htmlFor="dataSize">Data Size</label>
                <select id="dataSize" className="form-select" value={dataSize} onChange={(e) => onChange('dataSize', e.target.value)}>
                    <option value="">Select Size</option>
                    <option value="Small">Small (&lt; 1GB)</option>
                    <option value="Medium">Medium (1GB - 10GB)</option>
                    <option value="Large">Large (&gt; 10GB)</option>
                </select>
            </div>

            {/* ── Action Bar ── */}
            <div className="ingestion-action-bar">
                <input type="file" ref={importInputRef} style={{ display: 'none' }} accept=".json" onChange={handleImportFileSelect} />
                <button className="btn-secondary" onClick={() => importInputRef.current.click()}>
                    Import Configuration
                </button>
                <button
                    className="btn-primary"
                    disabled={!isFormValid || schemaLoading}
                    onClick={onSubmit}
                >
                    {schemaLoading ? 'Inspecting…' : 'Next Step'}
                </button>
            </div>
        </div>
    );
};

export default DataIngestionStep;
