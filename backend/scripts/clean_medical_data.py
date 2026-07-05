"""医疗对话数据清洗脚本 v3

从 Chinese-medical-dialogue-data CSV 提取「症状描述 → 诊断」对。
使用 200+ 常见病名列表精确匹配，标题优先。
"""

import csv, json, re, sys, random
from collections import Counter
from pathlib import Path

# ── 200+ 常见疾病名 ────────────────────────────
NAMED_DISEASES = sorted([
    # 心血管
    '高血压', '冠心病', '心绞痛', '心肌梗死', '心肌缺血', '心力衰竭', '心律失常',
    '心房颤动', '心脏瓣膜病', '心包炎', '高脂血症', '动脉硬化', '高尿酸血症', '心悸',
    # 脑血管
    '脑梗塞', '脑出血', '脑血管病', '中风', '脑血栓',
    # 内分泌
    '糖尿病', '甲亢', '甲状腺功能亢进', '甲减', '甲状腺功能减退',
    '甲状腺结节', '甲状腺炎', '痛风', '高血糖', '低血糖',
    # 呼吸
    '感冒', '流感', '肺炎', '支气管炎', '哮喘', '支气管哮喘', '慢阻肺', 'COPD',
    '肺气肿', '肺结核', '胸膜炎', '上呼吸道感染', '鼻炎', '过敏性鼻炎',
    # 消化
    '胃炎', '慢性胃炎', '浅表性胃炎', '萎缩性胃炎', '胃溃疡', '胃癌',
    '肝炎', '乙肝', '丙肝', '脂肪肝', '肝硬化', '肝癌',
    '胰腺炎', '胆囊炎', '胆结石', '肠炎', '结肠炎', '阑尾炎', '痔疮', '便秘',
    # 泌尿
    '肾炎', '肾结石', '肾功能不全', '肾病综合征', '膀胱炎',
    '尿道炎', '尿路感染', '前列腺炎', '前列腺增生',
    # 血液
    '贫血', '缺铁性贫血', '白血病', '淋巴瘤', '血小板减少',
    # 神经
    '癫痫', '帕金森病', '面瘫', '三叉神经痛', '偏头痛', '头痛',
    '脑膜炎', '脑炎', '失眠', '神经衰弱', '头晕', '眩晕',
    # 免疫/皮肤
    '荨麻疹', '湿疹', '皮炎', '风湿', '类风湿关节炎',
    '红斑狼疮', '强直性脊柱炎', '过敏', '银屑病', '白癜风',
    # 骨科
    '颈椎病', '腰椎间盘突出', '骨质增生', '骨折', '关节炎', '骨质疏松',
    '肩周炎', '腰肌劳损', '坐骨神经痛',
    # 妇科
    '盆腔炎', '宫颈炎', '阴道炎', '卵巢囊肿', '子宫肌瘤', '乳腺增生',
    # 五官
    '中耳炎', '扁桃体炎', '青光眼', '白内障', '结膜炎', '耳鸣',
    # 其他常见
    '静脉曲张', '血栓', '抑郁症', '焦虑症', '强迫症', '脂肪瘤',
    '肾虚', '脾虚', '湿气', '上火', '发热', '咳嗽', '腹泻',
], key=len, reverse=True)

# ── 函数 ────────────────────────────────────


def extract_diseases(text: str) -> list[str]:
    """从文本提取疾病名，优先 NAMED_DISEASES 精确匹配"""
    found = []
    seen = set()
    for m in re.finditer('|'.join(NAMED_DISEASES), text):
        word = m.group(0)
        if word not in seen:
            seen.add(word)
            found.append(word)
    return found[:3]


def extract_diagnosis(answer: str, title: str, department: str = "") -> str:
    """提取诊断: 标题优先 → 回答 → 科室兜底"""
    # 标题通常更干净（如"高血压怎么治疗"）
    if title:
        d = extract_diseases(title)
        if d:
            return d[0]
    # 回答里找
    d = extract_diseases(answer)
    if d:
        return d[0]
    # 兜底
    return f"{department}-其他" if department else "其他疾病"


def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[【】《》""''「」『』]', '', text)
    return text


def detect_and_read_csv(filepath: str) -> list[dict]:
    for enc in ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'gb18030']:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                sample = rows[0].get('ask', '')
                if any('一' <= c <= '鿿' for c in sample[:50]):
                    print(f"✓ 编码: {enc}, {len(rows)} 条")
                    return rows
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"无法识别编码: {filepath}")


# ── 随机信息 ─────────────────────────────────
FAMILY = ['张', '李', '王', '刘', '陈', '杨', '赵', '黄', '周', '吴',
          '徐', '孙', '胡', '朱', '高', '林', '何', '郭', '马', '罗']
GIVEN_M = ['伟', '强', '磊', '洋', '军', '勇', '杰', '涛', '明', '超',
           '建国', '志强', '文博', '浩然', '宇轩']
GIVEN_F = ['芳', '娜', '敏', '静', '丽', '艳', '娟', '秀英', '雨涵', '思远']


def random_patient():
    g = random.choice(['male', 'female'])
    fn = FAMILY
    gn = GIVEN_M if g == 'male' else GIVEN_F
    return random.choice(fn) + random.choice(gn), random.randint(18, 80), g


# ── Main ──────────────────────────────────────
def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else '/app/data/raw/neike.csv'
    output_path = sys.argv[2] if len(sys.argv) > 2 else '/app/data/cleaned_patients.json'
    max_records = int(sys.argv[3]) if len(sys.argv) > 3 else 20000

    print(f"📖 {csv_path}")
    rows = detect_and_read_csv(csv_path)

    cleaned = []
    seen = set()
    d_counter = Counter()
    skip = {'empty': 0, 'short': 0, 'dup': 0}

    for row in rows:
        ask = clean_text(row.get('ask', ''))
        answer = clean_text(row.get('answer', ''))
        title = clean_text(row.get('title', ''))
        dept = row.get('department', '')

        if not ask or not answer:
            skip['empty'] += 1; continue
        if len(ask) < 8 or len(answer) < 15:
            skip['short'] += 1; continue
        key = ask[:60]
        if key in seen:
            skip['dup'] += 1; continue
        seen.add(key)

        disease = extract_diagnosis(answer, title, dept)
        d_counter[disease] += 1
        name, age, gender = random_patient()

        cleaned.append({
            "name": name, "age": age, "gender": gender,
            "symptom_description": ask,
            "diagnosis": disease,
            "treatment": answer[:600],
        })

    print(f"📊 原始{len(rows)} → 有效{len(cleaned)} "
          f"(空{skip['empty']} 短{skip['short']} 重{skip['dup']})")

    # 分层采样
    if len(cleaned) > max_records:
        by_disease = {}
        for c in cleaned:
            by_disease.setdefault(c['diagnosis'], []).append(c)

        diseases = sorted(by_disease, key=lambda d: len(by_disease[d]), reverse=True)
        sampled = []
        for d in diseases:
            pool = by_disease[d]
            n = min(len(pool), max(3, max_records // len(diseases)))
            if n > 0:
                sampled.extend(random.sample(pool, n))
        random.shuffle(sampled)
        cleaned = sampled[:max_records]

    counts = Counter(c['diagnosis'] for c in cleaned)
    print(f"🏥 疾病种类: {len(counts)}   📦 {len(cleaned)} 条\n📋 Top-30:")
    for d, cnt in counts.most_common(30):
        print(f"  {d}: {cnt}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    print(f"\n💾 {output_path}")


if __name__ == '__main__':
    main()
