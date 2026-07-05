"""诊断预测推理服务 v2

支持 XGBoost (优先) + MLP (备选)
从 model_meta.json 加载疾病标签和特征名
"""

import json
import pickle
import logging
from pathlib import Path

import numpy as np
import torch

from app.config import settings

logger = logging.getLogger(__name__)

# ── 治疗建议库（可扩展）─────────────────────────
DEFAULT_TREATMENTS = {
    "高血压": "降压药（氨氯地平/缬沙坦），低盐低脂饮食，规律运动，监测血压",
    "冠心病": "抗血小板药（阿司匹林），他汀类降脂药，控制危险因素",
    "2型糖尿病": "降糖药（二甲双胍），饮食控制，规律运动，血糖监测",
    "高血糖": "控制碳水化合物摄入，规律运动，必要时口服降糖药",
    "低血糖": "立即补充糖分（糖果/果汁），调整降糖方案，规律进食",
    "细菌性肺炎": "抗生素治疗（头孢曲松+阿奇霉素），退热，氧疗，充分休息",
    "支气管炎": "止咳祛痰，支气管扩张剂，雾化吸入，多饮水",
    "哮喘": "支气管扩张剂，吸入糖皮质激素，避免过敏原",
    "上呼吸道感染": "对症支持治疗，多休息，补充维生素C",
    "流感": "抗病毒药（奥司他韦），退热，充分休息，补充水分",
    "肺结核": "抗结核四联疗法（异烟肼+利福平+吡嗪酰胺+乙胺丁醇）",
    "慢阻肺": "支气管扩张剂，氧疗，戒烟，肺康复训练",
    "感冒": "多休息，多饮水，对症退热止咳",
    "胸膜炎": "抗生素+镇痛，必要时胸腔穿刺引流",
    "肺气肿": "支气管扩张剂，氧疗，戒烟，预防感染",
    "缺铁性贫血": "补铁剂（硫酸亚铁），维生素C促进吸收，富含铁食物",
    "巨幼细胞性贫血": "补充叶酸+维生素B12，调整饮食",
    "血小板减少症": "糖皮质激素，免疫抑制剂，必要时输注血小板",
    "白血病": "化疗+靶向治疗，骨髓移植评估，支持治疗",
    "淋巴瘤": "化疗+放疗+靶向治疗（利妥昔单抗）",
    "肝炎": "保肝降酶（甘草酸制剂），抗病毒（如适用），戒酒",
    "肝硬化": "保肝，利尿，低盐饮食，防治并发症",
    "脂肪肝": "饮食控制+运动减重，控制血脂血糖，戒酒",
    "乙肝": "抗病毒治疗（恩替卡韦/替诺福韦），定期监测肝功能和HBV-DNA",
    "丙肝": "直接抗病毒药物（DAA），3-6个月可治愈",
    "肝癌": "手术切除+TACE+靶向+免疫治疗，MDT综合评估",
    "胆囊炎": "抗生素+禁食+补液，必要时腹腔镜胆囊切除",
    "肾炎": "ACEI/ARB降压护肾，低盐低蛋白饮食，控制血压",
    "肾功能不全": "控制原发病，优质低蛋白饮食，纠正水电解质紊乱",
    "肾病综合征": "糖皮质激素+免疫抑制剂，利尿消肿，ACEI/ARB降蛋白",
    "肾结石": "多饮水，药物排石，体外冲击波碎石，必要时手术",
    "尿路感染": "抗生素（左氧氟沙星/呋喃妥因），多饮水，注意卫生",
    "心力衰竭": "利尿剂+ACEI/ARB+β阻滞剂，限盐限水，监测体重",
    "心绞痛": "硝酸酯类+β阻滞剂+他汀，控制危险因素",
    "心肌缺血": "抗血小板+他汀+β阻滞剂，改善生活方式",
    "心律失常": "抗心律失常药（胺碘酮/普罗帕酮），必要时射频消融",
    "高脂血症": "他汀类降脂药，低脂饮食，规律运动",
    "动脉硬化": "抗血小板+他汀，控制血压血糖，健康饮食",
    "痛风": "非甾体抗炎药（急性期），降尿酸（别嘌醇/非布司他），低嘌呤饮食",
    "甲状腺功能亢进": "甲巯咪唑/丙硫氧嘧啶，β阻滞剂，必要时碘131或手术",
    "甲状腺功能减退": "左甲状腺素替代治疗，定期监测TSH",
    "慢性胃炎": "抑酸药（奥美拉唑）+胃黏膜保护剂，规律饮食，避免刺激",
    "胃溃疡": "PPI+铋剂+两种抗生素（四联疗法根除HP），规律饮食",
    "胰腺炎": "禁食+补液+镇痛+抑制胰酶分泌，治疗原发病",
    "肠炎": "抗生素（感染性），补液纠正电解质，对症止泻",
    "结肠炎": "5-ASA/激素/免疫抑制剂，必要时生物制剂",
    "脑梗塞": "溶栓+抗血小板+他汀，康复训练，控制危险因素",
    "脑出血": "控制血压+降颅压+止血，必要时手术清除血肿",
    "红斑狼疮": "糖皮质激素+羟氯喹+免疫抑制剂，避免日晒",
    "类风湿关节炎": "甲氨蝶呤+生物制剂，功能锻炼，控制炎症",
    "强直性脊柱炎": "NSAIDs+TNF抑制剂，功能锻炼保持脊柱活动度",
    "中耳炎": "抗生素口服+滴耳，必要时鼓膜切开引流",
    "扁桃体炎": "抗生素（青霉素类），对症退热止痛",
    "前列腺炎": "抗生素+α阻滞剂，温水坐浴，规律排精",
    "盆腔炎": "广谱抗生素联合治疗，必要时手术引流",
}


class DiagnosisService:
    """诊断推理服务 — 优先 XGBoost, 备选 MLP"""

    _instance = None
    _model = None
    _model_type = None
    _scaler = None
    _le = None
    _labels: list[str] = []
    _features: list[str] = []

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def labels(self) -> list[str]:
        if not self._labels:
            self._load_meta()
        return self._labels

    @property
    def features(self) -> list[str]:
        if not self._features:
            self._load_meta()
        return self._features

    def _load_meta(self):
        meta_path = Path("data/models/model_meta.json")
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self._labels = meta["disease_labels"]
            self._features = meta["lab_features"]

    def _load_model(self):
        """加载 XGBoost 主力模型（在 60 类问题上优于 MLP）"""
        xgb_path = Path("data/models/xgboost_baseline.pkl")

        if xgb_path.exists():
            logger.info("加载 XGBoost 诊断模型...")
            with open(xgb_path, "rb") as f:
                self._model, self._scaler, self._le = pickle.load(f)
            self._model_type = "xgboost"
            return

        # 回退 MLP
        mlp_path = Path(settings.model_path)
        if mlp_path.exists():
            logger.info("XGBoost 不存在, 回退 MLP")
            from app.ml.diagnosis_model import create_model
            self._model = create_model()
            self._model_type = "mlp"
            self._scaler = None
            return

        logger.warning("无任何模型权重, 使用随机 MLP")
        from app.ml.diagnosis_model import create_model
        self._model = create_model()
        self._model_type = "mlp"
        self._scaler = None

    def preprocess(self, lab_report: dict) -> np.ndarray:
        """化验单 dict → numpy [1, N]"""
        feats = self.features if self.features else list(lab_report.keys())
        values = [float(lab_report.get(f, 0.0)) for f in feats]
        return np.array([values], dtype=np.float32)

    def predict(self, lab_report: dict) -> dict:
        """预测疾病 Top-3"""
        if self._model is None:
            self._load_model()

        x = self.preprocess(lab_report)
        labels = self.labels

        if self._model_type == "xgboost":
            if self._scaler:
                x = self._scaler.transform(x)
            probs = self._model.predict_proba(x)[0]
            top3_idx = np.argsort(probs)[-3:][::-1]
            top3_labels = [labels[i] if i < len(labels) else "未知" for i in top3_idx]
            top3_probs = [float(probs[i]) for i in top3_idx]
        else:
            # MLP
            with torch.no_grad():
                probs = self._model(torch.FloatTensor(x))[0]
            top3_idx = torch.topk(probs, min(3, len(probs))).indices.tolist()
            top3_probs = torch.topk(probs, min(3, len(probs))).values.tolist()
            top3_labels = [labels[i] if i < len(labels) else "未知" for i in top3_idx]
            top3_probs = [float(p) for p in top3_probs]

        top = top3_labels[0]
        conf = top3_probs[0]
        treatment = DEFAULT_TREATMENTS.get(top, f"针对{top}的标准治疗方案，请遵医嘱")

        return {
            "top_prediction": top,
            "confidence": round(conf, 4),
            "top3": [{"disease": lbl, "probability": round(p, 4)}
                     for lbl, p in zip(top3_labels, top3_probs)],
            "treatment_suggestion": treatment,
        }


diagnosis_service = DiagnosisService()
