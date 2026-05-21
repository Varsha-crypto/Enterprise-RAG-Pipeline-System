import React from 'react';
import WizardStepper from './WizardStepper';
import './WizardLayout.css';

/**
 * WizardLayout Component
 * 
 * Global layout wrapper for the 4-step wizard flow.
 * Handles the persistent header with stepper and structured content layout.
 * 
 * @param {Object} props
 * @param {number} props.currentStep - Current step number (1-4)
 * @param {number[]} props.completedSteps - Array of completed step numbers
 * @param {function} props.onStepChange - Handler for step navigation
 * @param {React.ReactNode} props.children - Content for the current step
 * @param {React.ReactNode} [props.footer] - Optional footer actions
 */
const WizardLayout = ({
    currentStep,
    completedSteps,
    onStepChange,
    showExecutionStep = false, // Control visibility of Step 4
    skippedSteps = [],
    children,
    footer
}) => {
    const ALL_STEPS = [
        "Source",
        "Target Database",
        "AI Recommendation",
        "Pipeline Execution"
    ];

    // Filter steps: Hide "Execute Pipeline" unless explicitly requested
    const visibleSteps = showExecutionStep
        ? ALL_STEPS
        : ALL_STEPS.slice(0, 3);

    return (
        <div className="wizard-layout">
            <aside className="wizard-sidebar">
                <WizardStepper
                    steps={visibleSteps}
                    currentStep={currentStep}
                    completedSteps={completedSteps}
                    skippedSteps={skippedSteps}
                    onStepChange={onStepChange}
                    orientation="vertical"
                />
            </aside>

            <div className="wizard-main-container">
                <main className="wizard-content" key={currentStep}>
                    {children}
                </main>

                {footer && (
                    <footer className="wizard-footer">
                        {footer}
                    </footer>
                )}
            </div>
        </div>
    );
};

export default WizardLayout;
