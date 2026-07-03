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
      <div className="empty-state">
        <p>暂无诊断结果</p>
        <button className="btn-link" onClick={() => navigate('/')}>
          返回首页进行诊断
        </button>
      </div>
    );
  }

  return (
    <div>
      <DiagnosisResultCard result={data.diagnosis} />
      <SimilarCasesList cases={data.similar_cases} />
      <button className="btn-submit" onClick={() => navigate('/')}>
        🔄 重新诊断
      </button>
    </div>
  );
}
