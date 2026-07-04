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
    <section className="card">
      <div className="card-header">
        <span className="icon">📋</span>
        <h2>患者信息</h2>
      </div>

      <div className="form-row-3">
        <div className="form-group">
          <span className="form-label">姓名</span>
          <input
            type="text"
            value={formData.name}
            onChange={(e) => update('name', e.target.value)}
            placeholder="请输入姓名"
          />
        </div>
        <div className="form-group">
          <span className="form-label">年龄</span>
          <input
            type="number"
            value={formData.age || ''}
            onChange={(e) => update('age', Number(e.target.value))}
            placeholder="0"
            min={0}
            max={150}
          />
        </div>
        <div className="form-group">
          <span className="form-label">性别</span>
          <select
            value={formData.gender}
            onChange={(e) => update('gender', e.target.value)}
          >
            <option value="">请选择</option>
            <option value="male">男</option>
            <option value="female">女</option>
            <option value="other">其他</option>
          </select>
        </div>
      </div>

      <div className="form-group mt-2">
        <span className="form-label">症状描述</span>
        <textarea
          value={formData.symptom_description}
          onChange={(e) => update('symptom_description', e.target.value)}
          placeholder="请详细描述您的症状，例如：发烧39度，咳嗽有黄痰，胸闷气短，全身乏力..."
          rows={4}
        />
      </div>
    </section>
  );
}
