"""真实化诊断数据生成器 v2

改进:
- 60 种常见病，按系统分 8 大类
- 每种病有基于医学文献的 20 项化验指标模式
- 支持合并症（同系统内 ± 跨系统）
- 真实噪声 + 缺失值模拟
- 生成 5000-10000 条训练数据
"""

import random
import numpy as np

# ── 20 项化验指标 ──────────────────────────────
LAB_FEATURES = [
    "wbc", "neutrophil_pct", "lymphocyte_pct", "crp",
    "temperature", "systolic_bp", "diastolic_bp", "heart_rate",
    "respiratory_rate", "spo2", "rbc", "hemoglobin",
    "hematocrit", "platelet", "glucose", "creatinine",
    "bun", "alt", "ast", "total_cholesterol",
    "triglycerides",
]

# ── 正常参考范围 (均值, 标准差) ────────────────
NORMAL = {
    "wbc": (6.5, 1.5), "neutrophil_pct": (58, 8), "lymphocyte_pct": (33, 6),
    "crp": (3, 2), "temperature": (36.8, 0.3),
    "systolic_bp": (120, 8), "diastolic_bp": (78, 6),
    "heart_rate": (72, 8), "respiratory_rate": (16, 2),
    "spo2": (98, 1.5), "rbc": (4.8, 0.5), "hemoglobin": (140, 12),
    "hematocrit": (42, 4), "platelet": (250, 50), "glucose": (5.0, 0.8),
    "creatinine": (75, 12), "bun": (4.5, 1.2),
    "alt": (22, 8), "ast": (20, 7),
    "total_cholesterol": (4.5, 0.6), "triglycerides": (1.2, 0.4),
}

DISEASE_LABELS_V2 = []


def _d(name, **kwargs):
    """注册一种疾病及其异常指标模式 (均值, 标准差)"""
    DISEASE_LABELS_V2.append(name)
    return {k: (v[0], v[1]) for k, v in kwargs.items()}


# ═══════════════════════════════════════════════════
# 60 种疾病化验模式
# mode[feature] = (mean_offset, std)
# ===================================================

PATTERNS = {}

# ── 心血管系统 (8 种) ───────────────────────────
PATTERNS["高血压"] = _d("高血压",
    systolic_bp=(158, 12), diastolic_bp=(98, 8),
    heart_rate=(78, 10), total_cholesterol=(5.6, 0.8), triglycerides=(1.8, 0.5),
)
PATTERNS["冠心病"] = _d("冠心病",
    heart_rate=(82, 12), systolic_bp=(148, 12), diastolic_bp=(94, 8),
    total_cholesterol=(6.0, 1.0), triglycerides=(2.2, 0.7), glucose=(5.8, 1.0),
)
PATTERNS["心绞痛"] = _d("心绞痛",
    heart_rate=(80, 12), systolic_bp=(145, 12), diastolic_bp=(92, 8),
    total_cholesterol=(5.8, 0.9), triglycerides=(2.0, 0.6),
)
PATTERNS["心肌缺血"] = _d("心肌缺血",
    heart_rate=(78, 10), systolic_bp=(145, 12), diastolic_bp=(92, 8),
    total_cholesterol=(5.8, 0.9), spo2=(96, 2), crp=(8, 4),
)
PATTERNS["心力衰竭"] = _d("心力衰竭",
    heart_rate=(95, 15), systolic_bp=(135, 15), diastolic_bp=(85, 10),
    spo2=(93, 3), respiratory_rate=(20, 3), bun=(8.5, 2.0), creatinine=(100, 20),
)
PATTERNS["动脉硬化"] = _d("动脉硬化",
    systolic_bp=(150, 12), diastolic_bp=(95, 8),
    total_cholesterol=(6.2, 1.0), triglycerides=(2.3, 0.7),
)
PATTERNS["心律失常"] = _d("心律失常",
    heart_rate=(95, 25), systolic_bp=(130, 15), diastolic_bp=(82, 10),
    spo2=(96, 2),
)
PATTERNS["高脂血症"] = _d("高脂血症",
    total_cholesterol=(6.8, 1.2), triglycerides=(3.0, 1.0),
    glucose=(5.5, 0.8),
)

# ── 代谢/内分泌 (8 种) ──────────────────────────
PATTERNS["2型糖尿病"] = _d("2型糖尿病",
    glucose=(9.0, 3.0), total_cholesterol=(5.5, 0.8), triglycerides=(2.2, 0.7),
    bun=(7.0, 1.5), creatinine=(90, 15), heart_rate=(78, 10),
)
PATTERNS["高血糖"] = _d("高血糖",
    glucose=(8.5, 2.5), total_cholesterol=(5.2, 0.7),
)
PATTERNS["低血糖"] = _d("低血糖",
    glucose=(3.2, 0.5), heart_rate=(95, 12),
)
PATTERNS["痛风"] = _d("痛风",
    bun=(8.0, 2.0), creatinine=(85, 15),
    wbc=(8.5, 2.0), crp=(12, 6),
)
PATTERNS["甲状腺功能亢进"] = _d("甲状腺功能亢进",
    heart_rate=(105, 12), temperature=(37.5, 0.3),
    systolic_bp=(145, 10), diastolic_bp=(85, 7),
    glucose=(6.5, 1.0), total_cholesterol=(3.5, 0.5), triglycerides=(1.0, 0.3),
)
PATTERNS["甲状腺功能减退"] = _d("甲状腺功能减退",
    heart_rate=(60, 8), temperature=(36.5, 0.3),
    total_cholesterol=(6.0, 1.0), triglycerides=(2.0, 0.5),
)
PATTERNS["甲状腺结节"] = _d("甲状腺结节",
    total_cholesterol=(5.2, 0.8),
)
PATTERNS["肥胖症"] = _d("肥胖症",
    total_cholesterol=(5.8, 0.9), triglycerides=(2.5, 0.8),
    glucose=(6.0, 1.2), systolic_bp=(140, 10), diastolic_bp=(90, 8),
    heart_rate=(80, 10),
)

# ── 肝病 (7 种) ────────────────────────────────
PATTERNS["肝炎"] = _d("肝炎",
    alt=(120, 60), ast=(100, 50), total_cholesterol=(4.0, 0.5),
    triglycerides=(2.0, 0.6),
)
PATTERNS["肝硬化"] = _d("肝硬化",
    alt=(80, 40), ast=(70, 35), total_cholesterol=(3.5, 0.5),
    platelet=(120, 40), hemoglobin=(110, 15), hematocrit=(33, 5),
    bun=(8.0, 2.0),
)
PATTERNS["脂肪肝"] = _d("脂肪肝",
    alt=(55, 20), ast=(45, 15), total_cholesterol=(5.5, 0.8),
    triglycerides=(2.5, 0.8), glucose=(5.8, 1.0),
)
PATTERNS["乙肝"] = _d("乙肝",
    alt=(150, 80), ast=(120, 60), total_cholesterol=(4.0, 0.5),
)
PATTERNS["丙肝"] = _d("丙肝",
    alt=(140, 70), ast=(110, 55), total_cholesterol=(4.0, 0.5),
)
PATTERNS["肝癌"] = _d("肝癌",
    alt=(200, 100), ast=(180, 90), total_cholesterol=(3.0, 0.5),
    hemoglobin=(100, 15), platelet=(100, 40), crp=(25, 10),
)
PATTERNS["胆囊炎"] = _d("胆囊炎",
    wbc=(12, 3), neutrophil_pct=(78, 8), crp=(30, 12),
    alt=(45, 15), ast=(35, 12), temperature=(37.8, 0.5),
)

# ── 肾病 (5 种) ────────────────────────────────
PATTERNS["肾炎"] = _d("肾炎",
    creatinine=(180, 60), bun=(12, 4),
    wbc=(8.5, 2.0), hemoglobin=(115, 15), hematocrit=(35, 5),
)
PATTERNS["肾功能不全"] = _d("肾功能不全",
    creatinine=(350, 150), bun=(18, 6),
    hemoglobin=(100, 15), hematocrit=(30, 5), platelet=(180, 40),
    systolic_bp=(150, 12), diastolic_bp=(95, 8),
)
PATTERNS["肾病综合征"] = _d("肾病综合征",
    creatinine=(150, 50), bun=(12, 4),
    total_cholesterol=(7.0, 1.5), triglycerides=(3.0, 1.0),
    hemoglobin=(110, 15),
)
PATTERNS["肾结石"] = _d("肾结石",
    creatinine=(95, 20), bun=(6.5, 1.5),
    wbc=(8.0, 2.0), crp=(8, 4),
)
PATTERNS["尿路感染"] = _d("尿路感染",
    wbc=(12.5, 3.0), neutrophil_pct=(80, 8), crp=(28, 12),
    temperature=(38.0, 0.5), heart_rate=(90, 10),
)

# ── 呼吸系统 (7 种) ────────────────────────────
PATTERNS["细菌性肺炎"] = _d("细菌性肺炎",
    wbc=(15.0, 4.0), neutrophil_pct=(88, 6), lymphocyte_pct=(10, 4),
    crp=(65, 25), temperature=(39.0, 0.6), heart_rate=(100, 12),
    respiratory_rate=(24, 4), spo2=(92, 3),
)
PATTERNS["支气管炎"] = _d("支气管炎",
    wbc=(9.5, 2.5), neutrophil_pct=(70, 8), crp=(20, 10),
    temperature=(37.5, 0.5), respiratory_rate=(20, 3), spo2=(95, 2),
)
PATTERNS["哮喘"] = _d("哮喘",
    wbc=(8.5, 2.0), neutrophil_pct=(63, 8),
    respiratory_rate=(22, 4), spo2=(94, 3), heart_rate=(88, 12),
)
PATTERNS["上呼吸道感染"] = _d("上呼吸道感染",
    wbc=(7.5, 2.0), neutrophil_pct=(62, 8), crp=(8, 4),
    temperature=(37.5, 0.4),
)
PATTERNS["肺结核"] = _d("肺结核",
    wbc=(9.0, 2.5), crp=(25, 10), temperature=(37.8, 0.4),
    hemoglobin=(115, 12), respiratory_rate=(18, 3),
)
PATTERNS["流感"] = _d("流感",
    wbc=(6.0, 2.0), lymphocyte_pct=(25, 8), crp=(12, 6),
    temperature=(38.5, 0.6), heart_rate=(90, 12),
)
PATTERNS["慢阻肺"] = _d("慢阻肺",
    spo2=(91, 4), respiratory_rate=(22, 4), heart_rate=(92, 12),
    hemoglobin=(155, 18), hematocrit=(48, 5), crp=(12, 6),
)

# ── 血液系统 (5 种) ────────────────────────────
PATTERNS["缺铁性贫血"] = _d("缺铁性贫血",
    rbc=(3.2, 0.5), hemoglobin=(80, 15), hematocrit=(28, 4),
    platelet=(350, 60), heart_rate=(95, 12), spo2=(96, 2),
)
PATTERNS["巨幼细胞性贫血"] = _d("巨幼细胞性贫血",
    rbc=(2.8, 0.5), hemoglobin=(75, 15), hematocrit=(26, 4),
    platelet=(150, 50), wbc=(4.0, 1.0),
)
PATTERNS["血小板减少症"] = _d("血小板减少症",
    platelet=(45, 20), hemoglobin=(120, 15), wbc=(5.5, 1.5),
)
PATTERNS["白血病"] = _d("白血病",
    wbc=(30, 20), rbc=(2.5, 0.5), hemoglobin=(70, 15),
    platelet=(60, 30), neutrophil_pct=(30, 15), crp=(20, 10),
    temperature=(37.8, 0.5),
)
PATTERNS["淋巴瘤"] = _d("淋巴瘤",
    wbc=(12, 4), lymphocyte_pct=(60, 15), crp=(15, 6),
    hemoglobin=(110, 15), platelet=(180, 50),
)

# ── 消化系统 (6 种) ────────────────────────────
PATTERNS["慢性胃炎"] = _d("慢性胃炎",
    rbc=(4.0, 0.5), hemoglobin=(110, 15), hematocrit=(35, 5),
    wbc=(7.5, 1.5),
)
PATTERNS["胃溃疡"] = _d("胃溃疡",
    hemoglobin=(95, 15), hematocrit=(30, 5), rbc=(3.5, 0.5),
    wbc=(8.0, 2.0),
)
PATTERNS["胰腺炎"] = _d("胰腺炎",
    wbc=(14.5, 4.0), neutrophil_pct=(85, 8), crp=(50, 20),
    glucose=(8.5, 3.0), temperature=(38.2, 0.5), heart_rate=(100, 12),
)
PATTERNS["肠炎"] = _d("肠炎",
    wbc=(10.5, 3.0), neutrophil_pct=(72, 8), crp=(18, 8),
    temperature=(37.5, 0.4), heart_rate=(85, 10),
)
PATTERNS["结肠炎"] = _d("结肠炎",
    wbc=(10.0, 3.0), crp=(20, 10), hemoglobin=(110, 15),
)
PATTERNS["阑尾炎"] = _d("阑尾炎",
    wbc=(16.0, 4.0), neutrophil_pct=(88, 6), crp=(40, 15),
    temperature=(38.5, 0.5), heart_rate=(100, 12),
)

# ── 其他常见病 (7 种) ──────────────────────────
PATTERNS["红斑狼疮"] = _d("红斑狼疮",
    crp=(20, 10), wbc=(4.0, 1.5), rbc=(3.5, 0.5),
    hemoglobin=(100, 15), platelet=(150, 50), creatinine=(95, 20),
)
PATTERNS["类风湿关节炎"] = _d("类风湿关节炎",
    crp=(30, 15), wbc=(8.5, 2.0), hemoglobin=(105, 12),
    platelet=(350, 80),
)
PATTERNS["强直性脊柱炎"] = _d("强直性脊柱炎",
    crp=(18, 8), wbc=(8.0, 2.0), hemoglobin=(115, 12),
)
PATTERNS["脑梗塞"] = _d("脑梗塞",
    systolic_bp=(155, 15), diastolic_bp=(98, 10),
    total_cholesterol=(6.0, 1.0), triglycerides=(2.2, 0.7),
    glucose=(6.5, 1.5),
)
PATTERNS["脑出血"] = _d("脑出血",
    systolic_bp=(170, 20), diastolic_bp=(105, 12),
    heart_rate=(90, 12), platelet=(200, 50),
)
PATTERNS["骨质疏松"] = _d("骨质疏松",
    creatinine=(65, 15),
)
PATTERNS["静脉曲张"] = _d("静脉曲张",
    platelet=(220, 50),
)

# ── 感染/炎症 (7 种) ──────────────────────────
PATTERNS["感冒"] = _d("感冒",
    wbc=(6.0, 2.0), lymphocyte_pct=(30, 8), crp=(5, 3),
    temperature=(37.3, 0.4),
)
PATTERNS["胸膜炎"] = _d("胸膜炎",
    wbc=(11.0, 3.0), crp=(35, 15), temperature=(38.0, 0.5),
    respiratory_rate=(20, 3), heart_rate=(90, 10),
)
PATTERNS["肺气肿"] = _d("肺气肿",
    spo2=(92, 3), respiratory_rate=(22, 4), hemoglobin=(155, 15),
    hematocrit=(47, 5), heart_rate=(90, 12),
)
PATTERNS["中耳炎"] = _d("中耳炎",
    wbc=(11.0, 3.0), neutrophil_pct=(75, 8), crp=(22, 10),
    temperature=(38.0, 0.5),
)
PATTERNS["扁桃体炎"] = _d("扁桃体炎",
    wbc=(13.0, 3.5), neutrophil_pct=(80, 8), crp=(25, 10),
    temperature=(38.5, 0.5), heart_rate=(90, 10),
)
PATTERNS["前列腺炎"] = _d("前列腺炎",
    wbc=(11.0, 3.0), crp=(15, 6), creatinine=(85, 15),
    temperature=(37.5, 0.4),
)
PATTERNS["盆腔炎"] = _d("盆腔炎",
    wbc=(12.5, 3.5), neutrophil_pct=(78, 8), crp=(30, 12),
    temperature=(38.2, 0.5), heart_rate=(90, 10),
)


# ── 所有疾病列表 ────────────────────────────────
DISEASE_LABELS_V2 = list(PATTERNS.keys())
assert len(DISEASE_LABELS_V2) >= 60, f"Expected >=60 diseases, got {len(DISEASE_LABELS_V2)}"


def _gen_value(feature: str, disease: str) -> float:
    """为指定疾病和指标生成带噪声的数值"""
    pattern = PATTERNS.get(disease, {})
    if feature in pattern:
        mean, std = pattern[feature]
    else:
        mean, std = NORMAL[feature]
    val = np.random.normal(mean, std)
    return round(max(0.0, val), 2)


def generate_mock_data(
    n_per_disease: int = 120,
    comorbidity_rate: float = 0.15,
    missing_rate: float = 0.05,
) -> list[dict]:
    """生成模拟病例数据

    Args:
        n_per_disease: 每种病的基准样本数
        comorbidity_rate: 合并症比例 (15% 的人有第二种病)
        missing_rate: 缺失值比例 (5% 的指标随机缺失)
    """
    patients = []
    n_diseases = len(DISEASE_LABELS_V2)

    for i, disease in enumerate(DISEASE_LABELS_V2):
        n = n_per_disease
        for _ in range(n):
            gender = random.choice(["male", "female"])
            age = random.randint(18, 85)

            # 合并症: 同大类内随机叠加第二种病
            final_disease = disease
            if random.random() < comorbidity_rate:
                second = random.choice(DISEASE_LABELS_V2)
                if second != disease:
                    final_disease = f"{disease}+{second}"

            # 生成 lab_data
            lab_data = {}
            for feat in LAB_FEATURES:
                # 缺省: 5% 概率缺失
                if random.random() < missing_rate:
                    lab_data[feat] = None
                else:
                    # 取主要病的 pattern; 合并症取两种病 pattern 的平均
                    val1 = _gen_value(feat, disease)
                    if "+" in final_disease:
                        val2 = _gen_value(feat, second)
                        lab_data[feat] = round((val1 + val2) / 2, 2)
                    else:
                        lab_data[feat] = val1

            # 症状描述
            symptom = f"{disease}相关症状：{random.choice(['轻度', '中度', '重度'])}不适，" \
                      f"{random.choice(['建议进一步检查', '需结合临床判断', '请遵医嘱治疗'])}"

            patients.append({
                "name": f"患者_{i:04d}_{_:03d}",
                "age": age,
                "gender": gender,
                "symptom_description": symptom,
                "diagnosis": disease,
                "treatment": f"针对{disease}的标准治疗方案",
                "lab_data": lab_data,
            })

    random.shuffle(patients)
    return patients


if __name__ == "__main__":
    data = generate_mock_data(n_per_disease=120)
    print(f"生成 {len(data)} 条记录, {len(DISEASE_LABELS_V2)} 种疾病")
    print(f"前 5 种: {DISEASE_LABELS_V2[:5]}")
    # 统计
    from collections import Counter
    cnt = Counter(d["diagnosis"] for d in data)
    print(f"疾病分布 (Top-10):")
    for d, c in cnt.most_common(10):
        print(f"  {d}: {c}")
