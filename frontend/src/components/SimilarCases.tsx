import type { SimilarCase } from '../types/diagnosis';

interface Props {
  cases: SimilarCase[];
}

export default function SimilarCasesList({ cases }: Props) {
  return (
    <section className="card">
      <div className="card-header">
        <span className="icon">📚</span>
        <h2>相似历史病例 (RAG 检索)</h2>
        <span className="status-badge ready">
          {cases.length > 0 ? `${cases.length} 例匹配` : '无匹配'}
        </span>
      </div>

      {cases.length === 0 ? (
        <div className="empty-hint">暂无相似病例参考</div>
      ) : (
        <div className="similar-cases-list">
          {cases.map((c) => {
            const simPct = c.similarity * 100;
            return (
              <div key={c.id} className="similar-case-item">
                <div className="case-header">
                  <span
                    className={`similarity-badge ${simPct >= 90 ? 'high-match' : ''}`}
                  >
                    相似度 {simPct.toFixed(0)}%
                  </span>
                  <span className="case-diagnosis">确诊: {c.diagnosis}</span>
                </div>
                <div className="case-symptoms">症状: {c.symptom_description}</div>
                <div className="case-treatment">治疗: {c.treatment}</div>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}
