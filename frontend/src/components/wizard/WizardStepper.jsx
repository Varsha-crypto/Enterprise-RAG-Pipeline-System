import React from 'react';
import './WizardStepper.css';

/**
 * WizardStepper Component
 * 
 * Displays a persistent top stepper with navigation capabilities.
 * 
 * @param {Object} props
 * @param {number} props.currentStep - 1-based index of the current step
 * @param {number[]} props.completedSteps - Array of completed step indices
 * @param {function} props.onStepChange - Callback when a step is clicked
 * @param {string} props.orientation - 'horizontal' or 'vertical'
 */
const WizardStepper = ({ currentStep, completedSteps, skippedSteps = [], onStepChange, steps, orientation = 'horizontal' }) => {
    const calculateProgress = () => {
        if (steps.length <= 1) return 0;
        // Calculate progress percentage based on the highest completed step or current step
        const maxActiveStep = Math.max(currentStep, ...completedSteps);
        // 0% at step 1, 100% at last step
        return ((maxActiveStep - 1) / (steps.length - 1)) * 100;
    };

    const isStepClickable = (index) => {
        const stepNumber = index + 1;
        // Allow clicking if it's a completed step or the current step
        // Also allow clicking if it's the immediate next step and the current step is completed (optional, but standard flow usually restricts this)
        // Requirement says: "Allow navigation only to completed or current steps"
        return completedSteps.includes(stepNumber) || stepNumber === currentStep;
    };

    const handleStepClick = (index) => {
        if (isStepClickable(index)) {
            onStepChange(index + 1);
        }
    };

    return (
        <div className={`wizard-stepper ${orientation}`}>
            <div className="stepper-progress-bar-container">
                <div
                    className="stepper-progress-bar"
                    style={orientation === 'vertical'
                        ? { height: `${calculateProgress()}%` }
                        : { width: `${calculateProgress()}%` }
                    }
                ></div>
            </div>

            {steps.map((label, index) => {
                const stepNumber = index + 1;
                const isActive = stepNumber === currentStep;
                const isCompleted = completedSteps.includes(stepNumber);
                const isSkipped = skippedSteps.includes(stepNumber);
                const clickable = isStepClickable(index);

                return (
                    <div
                        key={index}
                        className={`step-item ${isActive ? 'active' : ''} ${isCompleted ? 'completed' : ''} ${isSkipped ? 'skipped' : ''} ${clickable ? 'clickable' : ''}`}
                        onClick={() => handleStepClick(index)}
                        title={isSkipped ? 'Skipped — not needed for this source type' : undefined}
                    >
                        <div className="step-indicator">
                            {isSkipped ? (
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                    <path d="M5 12h14M13 6l6 6-6 6" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
                                </svg>
                            ) : isCompleted ? (
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                    <path d="M20 6L9 17L4 12" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
                                </svg>
                            ) : (
                                stepNumber
                            )}
                        </div>
                        <span className="step-label">
                            {label}
                            {isSkipped && <span style={{ fontSize: '0.65rem', opacity: 0.6, marginLeft: '0.3rem' }}>(skipped)</span>}
                        </span>
                    </div>
                );
            })}
        </div>
    );
};

export default WizardStepper;
