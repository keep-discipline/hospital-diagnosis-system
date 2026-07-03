"""模拟医疗数据生成器

基于医学常识为每种疾病定义典型化验指标模式，加随机噪声模拟真实数据。
"""

import random
import numpy as np
from app.ml.diagnosis_model import DISEASE_LABELS, LAB_FEATURE_NAMES

# ── 各疾病的典型化验指标 (均值, 标准差) ────────────────────
# 仅定义偏离正常的指标，未定义的指标使用 NORMAL_DEFAULTS
DISEASE_PATTERNS = {
    "细菌性肺炎": {
        "wbc": (15.0, 3.0), "neutrophil_pct": (88, 5), "lymphocyte_pct": (10, 3),
        "crp": (60, 20), "temperature": (39.0, 0.5), "heart_rate": (100, 10),
        "respiratory_rate": (24, 3), "spo2": (93, 2),
    },
    "病毒性感冒": {
        "wbc": (7.0, 2.0), "neutrophil_pct": (60, 8), "lymphocyte_pct": (35, 8),
        "crp": (10, 5), "temperature": (38.0, 0.5), "heart_rate": (85, 10),
        "respiratory_rate": (18, 2), "spo2": (97, 2),
    },
    "急性支气管炎": {
        "wbc": (10.0, 2.5), "neutrophil_pct": (72, 8), "lymphocyte_pct": (22, 5),
        "crp": (25, 10), "temperature": (37.8, 0.5), "heart_rate": (90, 10),
        "respiratory_rate": (20, 3), "spo2": (95, 2),
    },
    "高血压": {
        "systolic_bp": (160, 10), "diastolic_bp": (100, 8),
        "heart_rate": (80, 10), "total_cholesterol": (5.8, 0.8),
        "triglycerides": (2.0, 0.5),
    },
    "2型糖尿病": {
        "glucose": (9.0, 2.0), "total_cholesterol": (5.5, 0.8),
        "triglycerides": (2.2, 0.6), "bun": (7.0, 1.5), "creatinine": (90, 15),
    },
    "冠心病": {
        "heart_rate": (90, 12), "systolic_bp": (150, 12), "diastolic_bp": (95, 8),
        "total_cholesterol": (6.2, 1.0), "triglycerides": (2.5, 0.7),
        "glucose": (6.0, 1.0),
    },
    "慢性胃炎": {
        "rbc": (4.0, 0.5), "hemoglobin": (110, 15), "hematocrit": (35, 5),
        "wbc": (8.5, 2.0),
    },
    "尿路感染": {
        "wbc": (13.0, 3.0), "neutrophil_pct": (82, 6), "crp": (35, 15),
        "temperature": (38.2, 0.5), "heart_rate": (92, 10),
    },
    "缺铁性贫血": {
        "rbc": (3.0, 0.5), "hemoglobin": (80, 15), "hematocrit": (28, 4),
        "platelet": (350, 50), "heart_rate": (95, 10), "spo2": (96, 2),
    },
    "甲状腺功能亢进": {
        "heart_rate": (105, 12), "temperature": (37.5, 0.3),
        "systolic_bp": (145, 10), "diastolic_bp": (85, 7),
        "glucose": (6.5, 1.0), "total_cholesterol": (3.5, 0.5),
    },
}

# 正常默认值
NORMAL_DEFAULTS = {
    "wbc": (6.5, 1.5), "neutrophil_pct": (58, 8), "lymphocyte_pct": (33, 6),
    "crp": (5, 3), "temperature": (36.8, 0.3), "systolic_bp": (120, 8),
    "diastolic_bp": (78, 6), "heart_rate": (72, 8), "respiratory_rate": (16, 2),
    "spo2": (98, 1), "rbc": (4.8, 0.5), "hemoglobin": (140, 12),
    "hematocrit": (42, 4), "platelet": (250, 50), "glucose": (5.0, 0.8),
    "creatinine": (75, 12), "bun": (4.5, 1.2), "alt": (22, 8),
    "ast": (20, 7), "total_cholesterol": (4.5, 0.6), "triglycerides": (1.2, 0.4),
}

SYMPTOM_TEMPLATES = {
    "细菌性肺炎": [
        "发烧{temp}度，咳嗽有黄痰，胸闷气短，全身乏力",
        "高热不退，咳脓痰，胸痛，呼吸急促",
        "持续发热伴寒战，咳嗽剧烈，痰中带血丝",
    ],
    "病毒性感冒": [
        "鼻塞流涕，打喷嚏，喉咙痛，轻度发热{temp}度",
        "全身酸痛乏力，头痛，流清涕，轻微咳嗽",
        "畏寒发热，咽痛，鼻塞，食欲不振",
    ],
    "急性支气管炎": [
        "咳嗽频繁，有白色黏痰，胸闷不适，低热",
        "刺激性干咳，夜间加重，咽部不适",
        "感冒后持续咳嗽，咳痰，轻度呼吸困难",
    ],
    "高血压": [
        "头晕头痛，颈项僵硬，心悸，失眠多梦",
        "后脑勺胀痛，眩晕耳鸣，注意力不集中",
        "无明显不适，体检发现血压偏高",
    ],
    "2型糖尿病": [
        "口渴多饮，多尿，体重下降，疲乏无力",
        "视物模糊，四肢麻木，伤口愈合缓慢",
        "经常饥饿，食量增大但体重减轻",
    ],
    "冠心病": [
        "活动后胸闷胸痛，气短，心悸，左肩放射痛",
        "心前区压榨感，持续数分钟，休息后缓解",
        "夜间阵发性呼吸困难，下肢轻度水肿",
    ],
    "慢性胃炎": [
        "上腹部隐痛，饭后饱胀，反酸嗳气，食欲减退",
        "胃部灼烧感，恶心，空腹时疼痛加重",
        "消化不良，腹胀腹泻交替，口苦口干",
    ],
    "尿路感染": [
        "尿频尿急尿痛，下腹部不适，腰酸",
        "排尿灼热感，尿液混浊，低热",
        "腰痛发热，尿频加重，全身不适",
    ],
    "缺铁性贫血": [
        "面色苍白，头晕乏力，心慌气短，注意力不集中",
        "指甲脆薄易裂，口角炎，异食癖",
        "活动后心悸加重，耳鸣，月经量多",
    ],
    "甲状腺功能亢进": [
        "心悸手抖，怕热多汗，食欲亢进但体重下降",
        "情绪易激动，失眠，大便次数增多",
        "颈部增粗，眼球突出，乏力消瘦",
    ],
}

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


def _generate_value(feature: str, disease: str) -> float:
    """为指定疾病和指标生成带高斯噪声的数值"""
    pattern = DISEASE_PATTERNS.get(disease, {})
    mean, std = pattern.get(feature, NORMAL_DEFAULTS[feature])
    val = np.random.normal(mean, std)
    return round(max(0.0, val), 2)


def generate_mock_data(n_per_disease: int = 50) -> list[dict]:
    """生成模拟病例数据

    Args:
        n_per_disease: 每种疾病生成的病例数

    Returns:
        病例列表，每个 dict 包含 name/age/gender/symptom_description/
        diagnosis/treatment/lab_data
    """
    patients = []
    for disease in DISEASE_LABELS:
        for i in range(n_per_disease):
            gender = random.choice(["male", "female"])
            age = random.randint(18, 80)
            templates = SYMPTOM_TEMPLATES[disease]
            template = random.choice(templates)
            temp_val = round(random.uniform(36.5, 39.5), 1)
            symptom_text = template.format(temp=temp_val)

            lab_data = {
                feat: _generate_value(feat, disease)
                for feat in LAB_FEATURE_NAMES
            }
            # 确保体温与症状描述一致
            lab_data["temperature"] = temp_val

            patients.append({
                "name": f"模拟患者_{disease}_{i + 1:03d}",
                "age": age,
                "gender": gender,
                "symptom_description": symptom_text,
                "diagnosis": disease,
                "treatment": TREATMENTS[disease],
                "lab_data": lab_data,
            })
    random.shuffle(patients)
    return patients
