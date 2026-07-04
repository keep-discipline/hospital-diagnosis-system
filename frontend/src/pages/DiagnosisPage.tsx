import { useState } from 'react';
import PatientForm from '../components/PatientForm';
import LabReportForm from '../components/LabReportForm';
import DiagnosisResultCard from '../components/DiagnosisResult';
import SimilarCasesList from '../components/SimilarCases';
import { api } from '../services/api';
import type { DiagnoseRequest, LabReport, DiagnoseResponse } from '../types/diagnosis';

const EMPTY_LAB: LabReport = {
  wbc: 0, neutrophil_pct: 0, lymphocyte_pct: 0, crp: 0,
  temperature: 36.8, systolic_bp: 120, diastolic_bp: 80,
  heart_rate: 72, respiratory_rate: 16, spo2: 98, rbc: 0,
  hemoglobin: 0, hematocrit: 0, platelet: 0, glucose: 0,
  creatinine: 0, bun: 0, alt: 0, ast: 0,
  total_cholesterol: 0, triglycerides: 0,
};

const EMPTY_FORM: DiagnoseRequest = {
  name: '',
  age: 0,
  gender: 'male',
  symptom_description: '',
  lab_report: EMPTY_LAB,
};

export default function DiagnosisPage() {
  const [formData, setFormData] = useState<DiagnoseRequest>(EMPTY_FORM);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<DiagnoseResponse | null>(null);

  const handleSubmit = async () => {
    if (!formData.name || !formData.age || !formData.gender) {
      setError('请填写基本信息（姓名、年龄、性别）');
      return;
    }
    if (!formData.symptom_description.trim()) {
      setError('请填写症状描述');
      return;
    }

    setError('');
    setLoading(true);
    try {
      const { data } = await api.diagnose(formData);
      setResult(data);
      // 滚动到结果区域
      setTimeout(() => {
        document.getElementById('result-panel')?.scrollIntoView({
          behavior: 'smooth',
          block: 'start',
        });
      }, 150);
    } catch (err) {
      setError('诊断请求失败，请检查后端服务是否启动');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setError('');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="diagnosis-layout">
      {/* ── 左栏：表单 ── */}
      <div className="left-panel">
        <PatientForm
          formData={formData}
          onChange={(data) => setFormData(data as DiagnoseRequest)}
        />

        <LabReportForm
          labData={formData.lab_report}
          onChange={(lab) => setFormData({ ...formData, lab_report: lab })}
        />

        {error && <div className="alert alert-error">{error}</div>}

        <button
          className="btn btn-primary"
          onClick={handleSubmit}
          disabled={loading}
        >
          {loading ? '⏳ 正在分析中...' : '🔍 开始智能诊断'}
        </button>
      </div>

      {/* ── 右栏：结果 ── */}
      <div className="right-panel" id="result-panel">
        {result ? (
          <>
            <DiagnosisResultCard result={result.diagnosis} />
            <SimilarCasesList cases={result.similar_cases} />
            <button className="btn btn-outline" onClick={handleReset} style={{ width: '100%' }}>
              🔄 重新诊断
            </button>
          </>
        ) : (
          <div className="card">
            <div className="empty-panel">
              <div className="empty-panel-icon">🔍</div>
              <h3>等待诊断</h3>
              <p>请在左侧填写患者信息、症状描述和化验单数据，然后点击「开始智能诊断」</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
