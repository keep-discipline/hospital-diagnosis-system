"""诊断预测深度学习模型（MLP）

输入: 21 维化验指标 → 输出: 10 种疾病的概率分布
设计选择: MLP 而非 Transformer，因为化验单是结构化数值数据，没有序列关系。
"""

import torch
import torch.nn as nn

# 疾病标签（10 种）
DISEASE_LABELS = [
    "细菌性肺炎",
    "病毒性感冒",
    "急性支气管炎",
    "高血压",
    "2型糖尿病",
    "冠心病",
    "慢性胃炎",
    "尿路感染",
    "缺铁性贫血",
    "甲状腺功能亢进",
]

# 化验指标名称（21 维，对齐 LabReport schema）
LAB_FEATURE_NAMES = [
    "wbc", "neutrophil_pct", "lymphocyte_pct", "crp",
    "temperature", "systolic_bp", "diastolic_bp", "heart_rate",
    "respiratory_rate", "spo2", "rbc", "hemoglobin",
    "hematocrit", "platelet", "glucose", "creatinine",
    "bun", "alt", "ast", "total_cholesterol", "triglycerides",
]


class DiagnosisPredictor(nn.Module):
    """MLP 疾病诊断预测模型 (v2 — 60类深度版)

    结构:
        Input(21) → BatchNorm → Linear(256) → ReLU → Dropout(0.3)
                  → Linear(256) → ReLU → Dropout(0.3)
                  → Linear(128) → ReLU → Dropout(0.3)
                  → Linear(64) → ReLU
                  → Linear(N) → Softmax
    """

    def __init__(
        self,
        input_dim: int = 21,
        num_diseases: int = 60,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.output = nn.Linear(64, num_diseases)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.batch_norm(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.relu(self.fc4(x))
        x = self.output(x)
        return torch.softmax(x, dim=-1)


def create_model(model_path: str | None = None) -> DiagnosisPredictor:
    """工厂函数：创建模型实例，如果有权重文件则加载

    Args:
        model_path: .pt 权重文件路径，为 None 则从配置读取

    Returns:
        DiagnosisPredictor 实例（eval 模式如果加载了权重）
    """
    from app.config import settings

    model = DiagnosisPredictor(
        input_dim=len(LAB_FEATURE_NAMES),
        num_diseases=len(DISEASE_LABELS),
    )
    path = model_path or settings.model_path

    try:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
    except FileNotFoundError:
        pass  # 未训练时使用随机权重（此时预测无意义，但结构可验证）

    return model
