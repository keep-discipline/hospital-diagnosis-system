import type { SimilarCase } from '../types/diagnosis';

interface Props {
  cases: SimilarCase[];
}

export default function SimilarCasesList({ cases }: Props) {
  if (cases.length === 0) {
    return (
      <section className="form-section">
        <h2>📚 相似历史病例 (RAG 检索)</h2>
        <p className="empty-hint">暂无相似病例参考</p>
      </section>
    );
  }

  return (
    <section className="form-section">
      <h2>📚 相似历史病例 (RAG 检索)</h2>
      {cases.map((c) => (
        <div key={c.id} className="similar-case-item">
          <div className="case-header">
            <span className="similarity-badge">
              相似度 {(c.similarity * 100).toFixed(0)}%
            </span>
            <span className="case-diagnosis">确诊: {c.diagnosis}</span>
          </div>
          <p className="case-symptoms">症状: {c.symptom_description}</p>
          <p className="case-treatment">治疗: {c.treatment}</p>
        </div>
      ))}
    </section>
  );
}
