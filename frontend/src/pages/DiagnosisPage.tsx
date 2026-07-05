import { useState, useEffect, useCallback } from 'react';
import PatientForm from '../components/PatientForm';
import LabReportForm from '../components/LabReportForm';
import DiagnosisResultCard from '../components/DiagnosisResult';
import SimilarCasesList from '../components/SimilarCases';
import LoadingSkeleton from '../components/LoadingSkeleton';
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

/** 诊断步骤文案 */
const DIAGNOSIS_STEPS = [
  { label: '正在分析症状描述...', icon: '🔍' },
  { label: '正在检索相似病例...', icon: '📚' },
  { label: 'AI 模型正在诊断...', icon: '🧠' },
  { label: '正在生成治疗方案...', icon: '💊' },
];

export default function DiagnosisPage() {
  const [formData, setFormData] = useState<DiagnoseRequest>(EMPTY_FORM);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<DiagnoseResponse | null>(null);
  const [currentStep, setCurrentStep] = useState(0);

  // 诊断过程中轮播步骤
  useEffect(() => {
    if (!loading) return;
    const timer = setInterval(() => {
      setCurrentStep((prev) => (prev < DIAGNOSIS_STEPS.length - 1 ? prev + 1 : prev));
    }, 2000);
    return () => clearInterval(timer);
  }, [loading]);

  /** 校验表单，返回第一个错误或 null */
  const validateForm = useCallback((): string | null => {
    if (!formData.name.trim()) return '请输入姓名';
    if (!formData.age || formData.age <= 0 || formData.age > 150) return '请输入有效年龄 (1-150)';
    if (!formData.gender) return '请选择性别';
    if (!formData.symptom_description.trim() || formData.symptom_description.trim().length < 2) {
      return '请输入至少 2 个字符的症状描述';
    }
    return null;
  }, [formData]);

  const handleSubmit = async () => {
    const validationError = validateForm();
    if (validationError) {
      setError(validationError);
      return;
    }

    setError('');
    setLoading(true);
    setCurrentStep(0);
    setResult(null);

    try {
      const { data } = await api.diagnose(formData);
      setResult(data);
      // 等 loading 动画结束再滚动
      setTimeout(() => {
        document.getElementById('result-panel')?.scrollIntoView({
          behavior: 'smooth',
          block: 'start',
        });
      }, 300);
    } catch (err) {
      setError('诊断请求失败，请检查后端服务是否启动');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleExport = async () => {
    if (!result) return;
    try {
      const blob = await api.exportReport({
        name: formData.name,
        age: formData.age,
        gender: formData.gender,
        symptom_description: formData.symptom_description,
        lab_data: formData.lab_report,
        diagnosis: result.diagnosis,
        similar_cases: result.similar_cases,
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `诊断报告_${formData.name}_${new Date().toLocaleDateString('zh-CN')}.pdf`;
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      setError('PDF 导出失败');
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
          onPatientInfo={(info) =>
            setFormData((prev) => ({
              ...prev,
              name: info.name || prev.name,
              age: info.age || prev.age,
              gender: (info.gender as 'male' | 'female' | 'other') || prev.gender,
            }))
          }
        />

        {error && (
          <div className="alert alert-error">
            <span className="alert-icon">⚠️</span>
            <span>{error}</span>
          </div>
        )}

        <button
          className={`btn btn-primary btn-lg ${loading ? 'btn-loading' : ''}`}
          onClick={handleSubmit}
          disabled={loading}
        >
          {loading ? (
            <>
              <span className="spinner" />
              <span>正在分析中...</span>
            </>
          ) : (
            '🔍 开始智能诊断'
          )}
        </button>
      </div>

      {/* ── 右栏：结果 ── */}
      <div className="right-panel" id="result-panel">
        {loading ? (
          <LoadingSkeleton steps={DIAGNOSIS_STEPS} currentStep={currentStep} />
        ) : result ? (
          <>
            <DiagnosisResultCard result={result.diagnosis} />
            <SimilarCasesList cases={result.similar_cases} />
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <button className="btn btn-outline" onClick={handleExport}>
                📥 导出报告
              </button>
              <button className="btn btn-outline" onClick={handleReset} style={{ flex: 1 }}>
                🔄 重新诊断
              </button>
            </div>
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
