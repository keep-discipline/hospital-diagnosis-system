"""诊断报告 PDF 生成服务"""

import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os, glob

# 自动查找中文字体
_font_path = None
for root, _, files in os.walk("/usr/share/fonts"):
    for f in files:
        if f.endswith((".ttf", ".ttc")) and any(k in f.lower() for k in ("uming", "arphic", "wqy", "noto", "wenkai")):
            _font_path = os.path.join(root, f)
            break
    if _font_path:
        break

if _font_path:
    pdfmetrics.registerFont(TTFont("ChineseFont", _font_path))
    FONT = "ChineseFont"
else:
    FONT = "Helvetica"

PRIMARY = HexColor("#0891B2")
DARK = HexColor("#1E293B")
GRAY = HexColor("#64748B")
LIGHT_BG = HexColor("#F8FAFC")


def generate_report(
    patient_name: str, patient_age: int, patient_gender: str,
    symptom: str, lab_data: dict, diagnosis: dict, similar_cases: list,
) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    y = h - 50

    def draw_text(text, x, y_pos, size=11, color=DARK, bold=False):
        c.setFont(FONT, size)
        c.setFillColor(color)
        c.drawString(x, y_pos, text)
        return y_pos - size * 1.8

    # 标题
    c.setFillColor(PRIMARY)
    c.rect(0, h - 45, w, 45, fill=1, stroke=0)
    c.setFillColor(HexColor("#FFFFFF"))
    c.setFont(FONT, 18)
    c.drawString(30, h - 32, "智能医疗诊断报告")

    y = h - 65
    draw_text(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 30, y, 9, GRAY)
    y -= 18

    # 患者信息
    c.setStrokeColor(PRIMARY)
    c.setLineWidth(1.5)
    c.line(30, y, w - 30, y)
    y -= 20
    draw_text("患者信息", 30, y, 14, PRIMARY)
    gender_map = {"male": "男", "female": "女", "other": "其他"}
    y -= 18
    y = draw_text(f"姓名: {patient_name}    年龄: {patient_age}    性别: {gender_map.get(patient_gender, patient_gender)}", 30, y, 12)

    # 症状描述
    y -= 10
    draw_text("症状描述", 30, y, 12, DARK)
    y -= 16
    # 折行处理
    if len(symptom) > 100:
        symptom = symptom[:97] + "..."
    y = draw_text(symptom, 40, y, 10, GRAY)
    y -= 8

    # 化验数据
    c.line(30, y, w - 30, y)
    y -= 18
    draw_text("化验指标", 30, y, 14, PRIMARY)
    y -= 18

    # 表头
    for col_x, col_t in [(30, "项目"), (280, "结果"), (380, "单位"), (460, "参考范围")]:
        draw_text(col_t, col_x, y, 10, GRAY)
    y -= 14
    c.line(30, y, w - 30, y)
    y -= 6

    normal_ranges = {
        "wbc": ("白细胞计数", "×10⁹/L", "4.0-10.0"),
        "neutrophil_pct": ("中性粒细胞%", "%", "50-70"),
        "lymphocyte_pct": ("淋巴细胞%", "%", "20-40"),
        "crp": ("C反应蛋白", "mg/L", "<8"),
        "temperature": ("体温", "°C", "36.0-37.2"),
        "systolic_bp": ("收缩压", "mmHg", "90-140"),
        "diastolic_bp": ("舒张压", "mmHg", "60-90"),
        "heart_rate": ("心率", "次/分", "60-100"),
        "respiratory_rate": ("呼吸频率", "次/分", "12-20"),
        "spo2": ("血氧饱和度", "%", "95-100"),
        "rbc": ("红细胞计数", "×10¹²/L", "4.0-5.5"),
        "hemoglobin": ("血红蛋白", "g/L", "120-160"),
        "hematocrit": ("血细胞比容", "%", "40-50"),
        "platelet": ("血小板计数", "×10⁹/L", "100-300"),
        "glucose": ("空腹血糖", "mmol/L", "3.9-6.1"),
        "creatinine": ("肌酐", "μmol/L", "44-133"),
        "bun": ("尿素氮", "mmol/L", "2.9-8.2"),
        "alt": ("谷丙转氨酶", "U/L", "<40"),
        "ast": ("谷草转氨酶", "U/L", "<40"),
        "total_cholesterol": ("总胆固醇", "mmol/L", "<5.2"),
        "triglycerides": ("甘油三酯", "mmol/L", "<1.7"),
    }

    abnormal_keys = set()
    for key, (label, unit, ref) in normal_ranges.items():
        val = lab_data.get(key, "-")
        is_ab = False
        if isinstance(val, (int, float)) and val > 0:
            if key == "wbc" and val > 10:
                is_ab = True
            elif key == "neutrophil_pct" and val > 70:
                is_ab = True
            elif key == "crp" and val > 8:
                is_ab = True
            elif key == "temperature" and (val < 36 or val > 37.2):
                is_ab = True
            elif key == "spo2" and val < 95:
                is_ab = True
            elif key == "heart_rate" and (val < 60 or val > 100):
                is_ab = True
        color = HexColor("#DC2626") if is_ab else DARK
        y = draw_text(f"{label}", 30, y, 9, color)
        c.drawString(280, y + 4, f"{val}")
        c.drawString(380, y + 4, unit)
        c.drawString(460, y + 4, ref)
        if is_ab:
            abnormal_keys.add(key)

    # 诊断结果
    y -= 15
    c.line(30, y, w - 30, y)
    y -= 18
    draw_text("AI 诊断结果", 30, y, 14, PRIMARY)
    y -= 20

    top = diagnosis.get("top_prediction", "未知")
    conf = diagnosis.get("confidence", 0) * 100
    draw_text(f"主要诊断: {top}  (置信度 {conf:.1f}%)", 40, y, 13, HexColor("#DC2626"))
    y -= 16
    for item in diagnosis.get("top3", [])[:3]:
        pct = item["probability"] * 100
        draw_text(f"  {item['disease']}: {pct:.1f}%", 40, y, 10, GRAY)
        y -= 13

    # 治疗建议
    y -= 8
    c.line(30, y, w - 30, y)
    y -= 18
    draw_text("治疗建议", 30, y, 14, PRIMARY)
    y -= 18
    treatment = diagnosis.get("treatment_suggestion", "请咨询医生")
    if len(treatment) > 120:
        treatment = treatment[:117] + "..."
    y = draw_text(treatment, 40, y, 10, DARK)

    # 相似病例
    if similar_cases:
        y -= 10
        c.line(30, y, w - 30, y)
        y -= 18
        draw_text(f"相似历史病例 (RAG 检索)", 30, y, 12, PRIMARY)
        y -= 16
        for sc in similar_cases[:3]:
            sim = sc["similarity"] * 100
            text = f"相似度 {sim:.0f}% | {sc['diagnosis']}"
            if len(text) > 70:
                text = text[:67] + "..."
            y = draw_text(text, 40, y, 9, GRAY)

    # 页脚
    c.setFont(FONT, 8)
    c.setFillColor(GRAY)
    c.drawString(30, 20, "本报告由 AI 生成，仅供学习参考，不构成医疗建议。")

    c.save()
    return buf.getvalue()
