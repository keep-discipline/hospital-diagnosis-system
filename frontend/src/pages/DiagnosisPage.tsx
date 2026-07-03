import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import PatientForm from '../components/PatientForm';
import LabReportForm from '../components/LabReportForm';
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
  const navigate = useNavigate();
  const [formData, setFormData] = useState<DiagnoseRequest>(EMPTY_FORM);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

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
      navigate('/result', { state: data as DiagnoseResponse });
    } catch (err) {
      setError('诊断请求失败，请检查后端服务是否启动');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <PatientForm
        formData={formData}
        onChange={(data) => setFormData(data as DiagnoseRequest)}
      />
      <LabReportForm
        labData={formData.lab_report}
        onChange={(lab) => setFormData({ ...formData, lab_report: lab })}
      />

      {error && <div className="error-message">{error}</div>}

      <button
        className="btn-submit"
        onClick={handleSubmit}
        disabled={loading}
      >
        {loading ? '⏳ 正在分析中...' : '🔍 开始诊断'}
      </button>
    </div>
  );
}
