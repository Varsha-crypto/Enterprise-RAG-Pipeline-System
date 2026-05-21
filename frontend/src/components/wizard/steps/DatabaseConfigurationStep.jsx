import React from 'react';
import './DatabaseConfigurationStep.css';

/**
 * DatabaseConfigurationStep Component
 * 
 * Screen 2: Configure database connection details.
 */
const DatabaseConfigurationStep = ({
    dbHost,
    dbPort,
    dbUser,
    dbPassword,
    dbName,
    tableName,
    onChange,
    onSubmit,
    onBack
}) => {

    // If fields are empty, backend will auto-create a local DB.
    // So we allow empty fields now.
    const isFormValid = true;

    return (
        <div className="database-configuration-step">


            <div className="form-group">
                <label htmlFor="dbHost">Database Host</label>
                <input
                    type="text"
                    id="dbHost"
                    className="form-input"
                    placeholder="localhost"
                    value={dbHost}
                    onChange={(e) => onChange('dbHost', e.target.value)}
                />
            </div>

            <div className="form-group">
                <label htmlFor="dbPort">Database Port</label>
                <input
                    type="text"
                    id="dbPort"
                    className="form-input"
                    placeholder="5432"
                    value={dbPort}
                    onChange={(e) => onChange('dbPort', e.target.value)}
                />
            </div>

            <div className="form-group">
                <label htmlFor="dbUser">Username</label>
                <input
                    type="text"
                    id="dbUser"
                    className="form-input"
                    placeholder="postgres"
                    value={dbUser}
                    onChange={(e) => onChange('dbUser', e.target.value)}
                />
            </div>

            <div className="form-group">
                <label htmlFor="dbPassword">Password</label>
                <input
                    type="password"
                    id="dbPassword"
                    className="form-input"
                    placeholder="••••••••"
                    value={dbPassword}
                    onChange={(e) => onChange('dbPassword', e.target.value)}
                />
            </div>

            <div className="form-group">
                <label htmlFor="dbName">Database Name</label>
                <input
                    type="text"
                    id="dbName"
                    className="form-input"
                    placeholder="rag_db_v1"
                    value={dbName}
                    onChange={(e) => onChange('dbName', e.target.value)}
                />
            </div>

            <div className="form-group">
                <label htmlFor="tableName">Table Name</label>
                <input
                    type="text"
                    id="tableName"
                    className="form-input"
                    placeholder="documents"
                    value={tableName}
                    onChange={(e) => onChange('tableName', e.target.value)}
                />
            </div>

            <div className="action-bar">
                <button
                    className="btn-secondary"
                    onClick={onBack}
                >
                    Back
                </button>
                <button
                    className="btn-primary"
                    disabled={!isFormValid}
                    onClick={onSubmit}
                >
                    Continue to Pipeline Design
                </button>
            </div>
        </div>
    );
};

export default DatabaseConfigurationStep;
