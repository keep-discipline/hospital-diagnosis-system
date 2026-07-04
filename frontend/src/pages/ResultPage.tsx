import { useLocation, useNavigate } from 'react-router-dom';
import DiagnosisResultCard from '../components/DiagnosisResult';
import SimilarCasesList from '../components/SimilarCases';
import type { DiagnoseResponse } from '../types/diagnosis';

export default function ResultPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const data = location.state as DiagnoseResponse | undefined;

  if (!data) {
    return (
      <div className="card">
        <div className="empty-panel">
          <div className="empty-panel-icon">📭</div>
          <h3>暂无诊断结果</h3>
          <p>请先填写诊断表单并提交</p>
          <button className="btn btn-primary mt-2" onClick={() => navigate('/')}>
            前往诊断页面
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="diagnosis-layout">
      <div className="left-panel">
        <DiagnosisResultCard result={data.diagnosis} />
      </div>
      <div className="right-panel">
        <SimilarCasesList cases={data.similar_cases} />
        <button className="btn btn-primary" onClick={() => navigate('/')}>
          🔄 重新诊断
        </button>
      </div>
    </div>
  );
}
