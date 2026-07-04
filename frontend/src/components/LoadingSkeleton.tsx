interface Step {
  label: string;
  icon: string;
}

interface Props {
  steps: Step[];
  currentStep: number;
}

export default function LoadingSkeleton({ steps, currentStep }: Props) {
  return (
    <div className="card loading-card">
      {/* 步骤进度 */}
      <div className="loading-steps">
        {steps.map((step, idx) => (
          <div
            key={step.label}
            className={`loading-step ${
              idx < currentStep ? 'done' : idx === currentStep ? 'active' : 'pending'
            }`}
          >
            <div className="loading-step-indicator">
              {idx < currentStep ? (
                <span className="step-check">✓</span>
              ) : idx === currentStep ? (
                <span className="step-spinner" />
              ) : (
                <span className="step-dot" />
              )}
            </div>
            <span className="loading-step-text">{step.label}</span>
          </div>
        ))}
      </div>

      {/* 骨架卡片 */}
      <div className="skeleton-card">
        <div className="skeleton-line skeleton-title" />
        <div className="skeleton-line skeleton-bar" />
        <div className="skeleton-line skeleton-bar w-60" />
        <div className="skeleton-line skeleton-bar w-80" />
      </div>

      <div className="skeleton-card">
        <div className="skeleton-line skeleton-title" />
        <div className="skeleton-line skeleton-text" />
        <div className="skeleton-line skeleton-text" />
        <div className="skeleton-line skeleton-text w-60" />
      </div>
    </div>
  );
}
