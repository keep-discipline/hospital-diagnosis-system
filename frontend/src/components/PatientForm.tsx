import type { DiagnoseRequest } from '../types/diagnosis';

interface Props {
  formData: DiagnoseRequest;
  onChange: (data: DiagnoseRequest) => void;
}

export default function PatientForm({ formData, onChange }: Props) {
  const update = (field: string, value: string | number) => {
    onChange({ ...formData, [field]: value });
  };

  return (
    <section className="form-section">
      <h2>📋 基本信息</h2>
      <div className="form-row">
        <label>
          姓名
          <input
            type="text"
            value={formData.name}
            onChange={(e) => update('name', e.target.value)}
            placeholder="请输入姓名"
          />
        </label>
        <label>
          年龄
          <input
            type="number"
            value={formData.age || ''}
            onChange={(e) => update('age', Number(e.target.value))}
            placeholder="0"
            min={0}
            max={150}
          />
        </label>
        <label>
          性别
          <select
            value={formData.gender}
            onChange={(e) => update('gender', e.target.value)}
          >
            <option value="">请选择</option>
            <option value="male">男</option>
            <option value="female">女</option>
            <option value="other">其他</option>
          </select>
        </label>
      </div>

      <label className="form-field-full">
        症状描述
        <textarea
          value={formData.symptom_description}
          onChange={(e) => update('symptom_description', e.target.value)}
          placeholder="请详细描述您的症状，例如：头痛发烧三天，咳嗽有黄痰，胸闷气短..."
          rows={4}
        />
      </label>
    </section>
  );
}
