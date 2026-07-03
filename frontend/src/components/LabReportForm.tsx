import { useState } from 'react';
import type { LabReport } from '../types/diagnosis';

interface Props {
  labData: LabReport;
  onChange: (data: LabReport) => void;
}

const FIELDS: { key: keyof LabReport; label: string; unit: string }[] = [
  { key: 'wbc', label: '白细胞计数', unit: '×10⁹/L' },
  { key: 'neutrophil_pct', label: '中性粒细胞百分比', unit: '%' },
  { key: 'lymphocyte_pct', label: '淋巴细胞百分比', unit: '%' },
  { key: 'crp', label: 'C反应蛋白', unit: 'mg/L' },
  { key: 'temperature', label: '体温', unit: '°C' },
  { key: 'systolic_bp', label: '收缩压', unit: 'mmHg' },
  { key: 'diastolic_bp', label: '舒张压', unit: 'mmHg' },
  { key: 'heart_rate', label: '心率', unit: '次/分' },
  { key: 'respiratory_rate', label: '呼吸频率', unit: '次/分' },
  { key: 'spo2', label: '血氧饱和度', unit: '%' },
  { key: 'rbc', label: '红细胞计数', unit: '×10¹²/L' },
  { key: 'hemoglobin', label: '血红蛋白', unit: 'g/L' },
  { key: 'hematocrit', label: '血细胞比容', unit: '%' },
  { key: 'platelet', label: '血小板计数', unit: '×10⁹/L' },
  { key: 'glucose', label: '空腹血糖', unit: 'mmol/L' },
  { key: 'creatinine', label: '肌酐', unit: 'μmol/L' },
  { key: 'bun', label: '尿素氮', unit: 'mmol/L' },
  { key: 'alt', label: '谷丙转氨酶', unit: 'U/L' },
  { key: 'ast', label: '谷草转氨酶', unit: 'U/L' },
  { key: 'total_cholesterol', label: '总胆固醇', unit: 'mmol/L' },
  { key: 'triglycerides', label: '甘油三酯', unit: 'mmol/L' },
];

export default function LabReportForm({ labData, onChange }: Props) {
  const [expanded, setExpanded] = useState(false);

  const updateField = (key: keyof LabReport, value: string) => {
    onChange({ ...labData, [key]: parseFloat(value) || 0 });
  };

  const visibleFields = expanded ? FIELDS : FIELDS.slice(0, 8);

  return (
    <section className="form-section">
      <h2>🔬 化验单数据</h2>
      <div className="lab-grid">
        {visibleFields.map(({ key, label, unit }) => (
          <label key={key}>
            {label}
            <div className="input-with-unit">
              <input
                type="number"
                step="any"
                value={labData[key] || ''}
                onChange={(e) => updateField(key, e.target.value)}
                placeholder="0"
              />
              <span className="unit">{unit}</span>
            </div>
          </label>
        ))}
      </div>
      <button
        type="button"
        className="btn-link"
        onClick={() => setExpanded(!expanded)}
      >
        {expanded ? '▲ 收起' : '▼ 展开全部指标 (21 项)'}
      </button>
    </section>
  );
}
