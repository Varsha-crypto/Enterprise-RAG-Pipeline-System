const DuplicateWarningModal = ({ rowCount, tableName, onProceed, onCancel }) => (
    <div className="dup-modal-overlay">
        <div className="dup-modal">
            <div className="dup-modal-icon">⚠️</div>
            <div className="dup-modal-title">Existing Data Detected</div>
            <div className="dup-modal-body">
                The target table <strong style={{ color: '#1e293b' }}>"{tableName}"</strong> already contains{' '}
                <strong style={{ color: '#dc2626' }}>{rowCount.toLocaleString()} rows</strong> from a previous pipeline run.
                <br /><br />
                Proceeding will add more embeddings and may cause <strong>duplicate entries</strong>. Do you want to continue?
            </div>
            <div className="dup-modal-actions">
                <button className="dup-btn-cancel" onClick={onCancel}>Cancel</button>
                <button className="dup-btn-proceed" onClick={onProceed}>Proceed Anyway</button>
            </div>
        </div>
    </div>
);

export default DuplicateWarningModal;
