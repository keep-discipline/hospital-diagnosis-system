import { useState, useRef } from 'react';
import type { LabReport } from '../types/diagnosis';

interface Props {
  labData: LabReport;
  onChange: (data: LabReport) => void;
}

// 21 项化验指标定义
const ALL_FIELDS: { key: keyof LabReport; label: string; unit: string }[] = [
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

// 分组定义：每组包含的字段 key
const LAB_GROUPS = [
  {
    icon: '🩸',
    title: '炎症指标',
    keys: ['wbc', 'neutrophil_pct', 'lymphocyte_pct', 'crp'] as (keyof LabReport)[],
  },
  {
    icon: '❤️',
    title: '生命体征',
    keys: [
      'temperature', 'systolic_bp', 'diastolic_bp',
      'heart_rate', 'respiratory_rate', 'spo2',
    ] as (keyof LabReport)[],
  },
  {
    icon: '🔴',
    title: '血常规',
    keys: ['rbc', 'hemoglobin', 'hematocrit', 'platelet'] as (keyof LabReport)[],
  },
  {
    icon: '🧪',
    title: '生化指标',
    keys: [
      'glucose', 'creatinine', 'bun', 'alt',
      'ast', 'total_cholesterol', 'triglycerides',
    ] as (keyof LabReport)[],
  },
];

export default function LabReportForm({ labData, onChange }: Props) {
  const [expandedGroups, setExpandedGroups] = useState<Record<number, boolean>>({
    0: true, // 炎症指标默认展开
    1: true, // 生命体征默认展开
    2: false,
    3: false,
  });

  // 上传状态
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [ocrStatus, setOcrStatus] = useState<'idle' | 'processing' | 'success' | 'error'>('idle');
  const [ocrMessage, setOcrMessage] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const toggleGroup = (idx: number) => {
    setExpandedGroups((prev) => ({ ...prev, [idx]: !prev[idx] }));
  };

  const updateField = (key: keyof LabReport, value: string) => {
    onChange({ ...labData, [key]: parseFloat(value) || 0 });
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setUploadFile(file);
      setOcrStatus('idle');
      setOcrMessage('');
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) {
      setUploadFile(file);
      setOcrStatus('idle');
      setOcrMessage('');
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleOCR = async () => {
    if (!uploadFile) return;

    setOcrStatus('processing');
    setOcrMessage('正在识别化验单...');

    try {
      const formData = new FormData();
      formData.append('file', uploadFile);

      const response = await fetch('/api/ocr', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`识别失败 (${response.status})`);
      }

      const result = await response.json();
      // 用识别结果填充表单
      const updated = { ...labData };
      for (const [key, value] of Object.entries(result.lab_data)) {
        if (key in updated) {
          updated[key as keyof LabReport] = value as number;
        }
      }
      onChange(updated);

      setOcrStatus('success');
      setOcrMessage(`识别成功！已填充 ${Object.keys(result.lab_data).length} 项指标`);
    } catch (err) {
      setOcrStatus('error');
      setOcrMessage(err instanceof Error ? err.message : '识别失败，请重试');
    }
  };

  // 从字段列表中查 label/unit
  const getFieldInfo = (key: keyof LabReport) => {
    return ALL_FIELDS.find((f) => f.key === key)!;
  };

  return (
    <section className="card">
      <div className="card-header">
        <span className="icon">🔬</span>
        <h2>化验单数据</h2>
        <span className="status-badge ready">21 项指标</span>
      </div>

      {/* ── 图片上传区域 ── */}
      <div
        className={`upload-area ${uploadFile ? 'has-file' : ''}`}
        onClick={handleUploadClick}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          capture="environment"
          style={{ display: 'none' }}
          onChange={handleFileChange}
        />
        {uploadFile ? (
          <>
            <div className="upload-icon">📄</div>
            <div className="upload-file-name">{uploadFile.name}</div>
            <div className="upload-hint">点击重新选择图片</div>
          </>
        ) : (
          <>
            <div className="upload-icon">📸</div>
            <div className="upload-text">上传化验单照片自动识别</div>
            <div className="upload-hint">支持拍照或从相册选取 · JPG/PNG</div>
          </>
        )}
      </div>

      {/* OCR 按钮 + 状态 */}
      {uploadFile && (
        <div style={{ marginBottom: '0.75rem' }}>
          <button
            type="button"
            className="btn btn-outline"
            onClick={handleOCR}
            disabled={ocrStatus === 'processing'}
            style={{ width: '100%' }}
          >
            {ocrStatus === 'processing' ? '⏳ 识别中...' : '🔍 开始识别'}
          </button>
          {ocrMessage && (
            <div className={`ocr-status ${ocrStatus} mt-1`}>{ocrMessage}</div>
          )}
        </div>
      )}

      {/* ── 分组表单 ── */}
      <div className="lab-groups">
        {LAB_GROUPS.map((group, idx) => {
          const isOpen = expandedGroups[idx];
          const fields = group.keys.map(getFieldInfo);
          const filledCount = group.keys.filter(
            (k) => labData[k] && labData[k] !== 0
          ).length;

          return (
            <div key={group.title} className="lab-group-card">
              <div
                className="lab-group-header"
                onClick={() => toggleGroup(idx)}
              >
                <span className="lab-group-icon">{group.icon}</span>
                <span className="lab-group-title">{group.title}</span>
                <span className="lab-group-count">
                  {filledCount}/{group.keys.length}
                </span>
                <span className={`lab-group-arrow ${isOpen ? 'open' : ''}`}>
                  ▼
                </span>
              </div>

              {isOpen && (
                <div className="lab-group-body">
                  {fields.map(({ key, label, unit }) => (
                    <div key={key} className="lab-field">
                      <span className="lab-field-name">{label}</span>
                      <div className="lab-field-input-wrap">
                        <input
                          type="number"
                          step="any"
                          value={labData[key] || ''}
                          onChange={(e) => updateField(key, e.target.value)}
                          placeholder="-"
                        />
                        <span className="lab-field-unit">{unit}</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>

      <p className="text-muted text-center mt-2">
        提示：上传化验单照片可自动识别填充，也可手动输入。未填项默认为正常值。
      </p>
    </section>
  );
}
