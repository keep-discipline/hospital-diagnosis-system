import type { DiagnosisResult as DiagnosisResultType } from '../types/diagnosis';

interface Props {
  result: DiagnosisResultType;
}

const RISK_COLORS: Record<number, string> = {
  0: '#22c55e',
  1: '#f59e0b',
  2: '#ef4444',
};

export default function DiagnosisResultCard({ result }: Props) {
  return (
    <section className="form-section result-card">
      <h2>📊 AI 诊断结果</h2>

      <div className="primary-diagnosis">
        <span className="disease-label">{result.top_prediction}</span>
        <span className="confidence">
          置信度 {(result.confidence * 100).toFixed(1)}%
        </span>
      </div>

      <div className="top3-list">
        {result.top3.map((item, idx) => (
          <div key={item.disease} className="top3-item">
            <span
              className="risk-dot"
              style={{ background: RISK_COLORS[idx] || '#6b7280' }}
            />
            <span className="disease-name">{item.disease}</span>
            <span className="probability">
              {(item.probability * 100).toFixed(1)}%
            </span>
            <div className="prob-bar">
              <div
                className="prob-bar-fill"
                style={{
                  width: `${(item.probability * 100).toFixed(1)}%`,
                  background: RISK_COLORS[idx] || '#6b7280',
                }}
              />
            </div>
          </div>
        ))}
      </div>

      <div className="treatment-box">
        <h3>💊 建议治疗方案</h3>
        <p>{result.treatment_suggestion}</p>
      </div>
    </section>
  );
}
