import type { DiagnosisResult as DiagnosisResultType } from '../types/diagnosis';

interface Props {
  result: DiagnosisResultType;
}

const BAR_COLORS = ['#0891B2', '#D97706', '#DC2626'];

function getConfidenceLevel(pct: number) {
  if (pct >= 80) return 'high';
  if (pct >= 50) return 'medium';
  return 'low';
}

export default function DiagnosisResultCard({ result }: Props) {
  const confidencePct = result.confidence * 100;

  return (
    <section className="card result-card">
      <div className="card-header">
        <span className="icon">📊</span>
        <h2>AI 诊断结果</h2>
      </div>

      {/* 主要诊断 */}
      <div className="primary-diagnosis">
        <span className="disease-label">{result.top_prediction}</span>
        <span className={`confidence ${getConfidenceLevel(confidencePct)}`}>
          置信度 {confidencePct.toFixed(1)}%
        </span>
      </div>

      {/* Top-3 列表 */}
      <div className="top3-list">
        {result.top3.map((item, idx) => {
          const probPct = item.probability * 100;
          return (
            <div key={item.disease} className="top3-item">
              <div className={`top3-rank r${idx + 1}`}>{idx + 1}</div>
              <span className="top3-disease">{item.disease}</span>
              <span className="top3-prob">{probPct.toFixed(1)}%</span>
              <div className="prob-bar">
                <div
                  className="prob-bar-fill"
                  style={{
                    width: `${probPct}%`,
                    background: BAR_COLORS[idx] || '#6B7280',
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>

      {/* 治疗方案 */}
      <div className="treatment-box">
        <h3>💊 建议治疗方案</h3>
        <p>{result.treatment_suggestion}</p>
      </div>
    </section>
  );
}
