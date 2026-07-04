import { useState, useCallback } from 'react';
import type { DiagnoseRequest } from '../types/diagnosis';

interface Props {
  formData: DiagnoseRequest;
  onChange: (data: DiagnoseRequest) => void;
}

interface FieldErrors {
  name?: string;
  age?: string;
  gender?: string;
  symptom_description?: string;
}

type PatientField = keyof FieldErrors;

/** 校验单字段，返回错误信息或 undefined */
function validateField(field: PatientField, value: string | number): string | undefined {
  switch (field) {
    case 'name':
      if (!value || (typeof value === 'string' && !value.trim())) return '请输入姓名';
      return undefined;
    case 'age':
      if (value === 0 || value === undefined || value === '') return '请输入年龄';
      if (Number(value) < 0 || Number(value) > 150) return '年龄范围 0-150';
      return undefined;
    case 'gender':
      if (!value) return '请选择性别';
      return undefined;
    case 'symptom_description':
      if (!value || (typeof value === 'string' && !value.trim())) return '请输入症状描述';
      if (typeof value === 'string' && value.trim().length < 2) return '症状描述至少 2 个字符';
      return undefined;
    default:
      return undefined;
  }
}

export default function PatientForm({ formData, onChange }: Props) {
  const [touched, setTouched] = useState<Record<string, boolean>>({});
  const [errors, setErrors] = useState<FieldErrors>({});

  const handleBlur = useCallback((field: PatientField) => {
    setTouched((prev) => ({ ...prev, [field]: true }));
    const err = validateField(field, formData[field]);
    setErrors((prev) => {
      const next = { ...prev };
      if (err) next[field] = err;
      else delete next[field];
      return next;
    });
  }, [formData]);

  const update = (field: PatientField, value: string | number) => {
    onChange({ ...formData, [field]: value });
    if (touched[field]) {
      const err = validateField(field, value);
      setErrors((prev) => {
        const next = { ...prev };
        if (err) next[field] = err;
        else delete next[field];
        return next;
      });
    }
  };

  const hasError = (field: PatientField) => touched[field] && errors[field];
  const descLen = formData.symptom_description.length;

  return (
    <section className="card">
      <div className="card-header">
        <span className="icon">📋</span>
        <h2>患者信息</h2>
      </div>

      {/* 基本信息 */}
      <div className="form-row-3">
        <div className={`form-group ${hasError('name') ? 'has-error' : ''}`}>
          <span className="form-label">
            姓名 <span className="required">*</span>
          </span>
          <input
            type="text"
            value={formData.name}
            onChange={(e) => update('name', e.target.value)}
            onBlur={() => handleBlur('name')}
            placeholder="请输入姓名"
          />
          {hasError('name') && <span className="field-error">{errors.name}</span>}
        </div>

        <div className={`form-group ${hasError('age') ? 'has-error' : ''}`}>
          <span className="form-label">
            年龄 <span className="required">*</span>
          </span>
          <input
            type="number"
            value={formData.age || ''}
            onChange={(e) => update('age', Number(e.target.value))}
            onBlur={() => handleBlur('age')}
            placeholder="0"
            min={0}
            max={150}
          />
          {hasError('age') && <span className="field-error">{errors.age}</span>}
        </div>

        <div className={`form-group ${hasError('gender') ? 'has-error' : ''}`}>
          <span className="form-label">
            性别 <span className="required">*</span>
          </span>
          <select
            value={formData.gender}
            onChange={(e) => update('gender', e.target.value)}
            onBlur={() => handleBlur('gender')}
          >
            <option value="">请选择</option>
            <option value="male">男</option>
            <option value="female">女</option>
            <option value="other">其他</option>
          </select>
          {hasError('gender') && <span className="field-error">{errors.gender}</span>}
        </div>
      </div>

      {/* 症状描述 */}
      <div className={`form-group mt-2 ${hasError('symptom_description') ? 'has-error' : ''}`}>
        <span className="form-label">
          症状描述 <span className="required">*</span>
        </span>
        <textarea
          value={formData.symptom_description}
          onChange={(e) => update('symptom_description', e.target.value)}
          onBlur={() => handleBlur('symptom_description')}
          placeholder="请详细描述您的症状，例如：发烧39度，咳嗽有黄痰，胸闷气短，全身乏力..."
          rows={4}
          maxLength={2000}
        />
        <div className="field-meta">
          {hasError('symptom_description') && (
            <span className="field-error">{errors.symptom_description}</span>
          )}
          <span className="char-count">{descLen}/2000</span>
        </div>
      </div>
    </section>
  );
}
