"""诊断预测推理服务

封装 MLP 模型：加载权重 → 预处理化验单 → 推理 → 后处理结果
使用单例模式，模型只加载一次。
"""

import torch

from app.ml.diagnosis_model import (
    DiagnosisPredictor,
    DISEASE_LABELS,
    LAB_FEATURE_NAMES,
    create_model,
)

# 治疗方案映射
TREATMENTS = {
    "细菌性肺炎": "抗生素治疗（头孢曲松+阿奇霉素），退热，氧疗，充分休息，多饮水",
    "病毒性感冒": "对症支持治疗：退热药（对乙酰氨基酚），抗组胺药，充分休息，补充维生素C",
    "急性支气管炎": "止咳祛痰（氨溴索），支气管扩张剂，雾化吸入，多饮水，避免刺激",
    "高血压": "降压药（氨氯地平/缬沙坦），低盐饮食，规律运动，控制体重，监测血压",
    "2型糖尿病": "降糖药（二甲双胍），饮食控制，规律运动，血糖监测，糖尿病教育",
    "冠心病": "抗血小板药（阿司匹林），他汀类降脂药，硝酸酯类，控制危险因素",
    "慢性胃炎": "抑酸药（奥美拉唑），胃黏膜保护剂，规律饮食，避免辛辣刺激",
    "尿路感染": "抗生素（左氧氟沙星/呋喃妥因），多饮水，注意个人卫生",
    "缺铁性贫血": "补铁剂（硫酸亚铁），维生素C促进吸收，富含铁食物，病因治疗",
    "甲状腺功能亢进": "抗甲状腺药（甲巯咪唑），β受体阻滞剂，定期复查甲功",
}


class DiagnosisService:
    """诊断推理服务 — 单例，模型只加载一次"""

    _instance = None
    _model: DiagnosisPredictor | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def model(self) -> DiagnosisPredictor:
        """延迟加载：第一次调用 predict() 时才加载权重"""
        if self._model is None:
            self._model = create_model()
        return self._model

    def preprocess(self, lab_report: dict) -> torch.Tensor:
        """化验单 dict → 模型输入张量 [1, 21]"""
        features = [
            float(lab_report.get(feat, 0.0)) for feat in LAB_FEATURE_NAMES
        ]
        return torch.tensor([features], dtype=torch.float32)

    def predict(self, lab_report: dict) -> dict:
        """根据化验单数据预测疾病

        Args:
            lab_report: 包含 21 项化验指标的 dict（对齐 LabReport schema）

        Returns:
            {
                "top_prediction": str,    # 最可能的疾病
                "confidence": float,      # 置信度
                "top3": [...],            # Top-3 疾病 + 概率
                "treatment_suggestion": str,  # 治疗建议
            }
        """
        x = self.preprocess(lab_report)
        with torch.no_grad():
            probs = self.model(x)[0]  # [num_diseases]

        # 取 Top-3
        top3_indices = torch.topk(probs, 3).indices.tolist()
        top3_probs = torch.topk(probs, 3).values.tolist()

        top_prediction = DISEASE_LABELS[top3_indices[0]]
        confidence = round(top3_probs[0], 4)

        top3 = [
            {"disease": DISEASE_LABELS[idx], "probability": round(prob, 4)}
            for idx, prob in zip(top3_indices, top3_probs)
        ]

        treatment = TREATMENTS.get(
            top_prediction, "请咨询专业医生获取治疗方案"
        )

        return {
            "top_prediction": top_prediction,
            "confidence": confidence,
            "top3": top3,
            "treatment_suggestion": treatment,
        }


# 全局单例
diagnosis_service = DiagnosisService()
