"""OCR 化验单识别服务

流程: 化验单图片 → 百度 OCR / EasyOCR 文字提取 → DeepSeek 结构化解析 → 21 项化验指标 JSON

OCR 引擎优先级: 百度 OCR（云端，精确版）→ EasyOCR（本地 PyTorch，降级回退）
"""

import json
import io
import base64
import logging
from typing import Optional

import numpy as np
from PIL import Image
from openai import OpenAI
from app.config import settings

logger = logging.getLogger(__name__)

# 21 项化验指标的 prompt 描述
LAB_FIELDS_PROMPT = """你是一个专业的医学化验单数据提取助手。请从以下 OCR 识别出的文本中，提取化验指标数值。

需要提取的 21 项指标（如果某项在文本中找不到，值设为 null）：
- wbc: 白细胞计数 (×10⁹/L)，正常范围 4.0-10.0
- neutrophil_pct: 中性粒细胞百分比 (%)，正常范围 50-70
- lymphocyte_pct: 淋巴细胞百分比 (%)，正常范围 20-40
- crp: C反应蛋白 (mg/L)，正常 < 8
- temperature: 体温 (°C)，正常 36.0-37.2
- systolic_bp: 收缩压 (mmHg)，正常 90-140
- diastolic_bp: 舒张压 (mmHg)，正常 60-90
- heart_rate: 心率 (次/分)，正常 60-100
- respiratory_rate: 呼吸频率 (次/分)，正常 12-20
- spo2: 血氧饱和度 (%)，正常 95-100
- rbc: 红细胞计数 (×10¹²/L)，正常 4.0-5.5
- hemoglobin: 血红蛋白 (g/L)，正常 120-160
- hematocrit: 血细胞比容 (%)，正常 40-50
- platelet: 血小板计数 (×10⁹/L)，正常 100-300
- glucose: 空腹血糖 (mmol/L)，正常 3.9-6.1
- creatinine: 肌酐 (μmol/L)，正常 44-133
- bun: 尿素氮 (mmol/L)，正常 2.9-8.2
- alt: 谷丙转氨酶 (U/L)，正常 < 40
- ast: 谷草转氨酶 (U/L)，正常 < 40
- total_cholesterol: 总胆固醇 (mmol/L)，正常 < 5.2
- triglycerides: 甘油三酯 (mmol/L)，正常 < 1.7

注意事项：
1. 识别数值和单位，注意单位换算（如 g/L 和 mg/dL 的区别）
2. 百分比类指标（如中性粒细胞百分比）的值是百分数，如 "70%" → 70.0
3. 如果某个指标有多个数值（如多次测量），取最新或最合理的值
4. 如果文本中完全找不到某指标，值设为 null
5. 只返回纯 JSON，不要包含任何其他文字

请严格按以下 JSON 格式返回：
{"lab_data": {"wbc": 数值或null, "neutrophil_pct": 数值或null, ...}}"""


class OCRService:
    """化验单 OCR 识别服务 — 单例模式

    OCR 引擎：
    - 主引擎：百度 OCR 精确版（云端 API，中文识别准确率高）
    - 降级回退：EasyOCR（本地 PyTorch，无需网络）
    - 结构化：DeepSeek API

    百度 OCR 不可用时自动降级到 EasyOCR，确保服务可用性。
    """

    _instance: Optional["OCRService"] = None
    _ocr = None  # EasyOCR Reader（延迟加载）
    _baidu_client = None  # 百度 OCR 客户端（延迟加载）
    _deepseek_client: Optional[OpenAI] = None

    def __new__(cls) -> "OCRService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # ── 百度 OCR（主引擎）──────────────────────────────────

    @property
    def baidu_available(self) -> bool:
        """检查百度 OCR 凭证是否已配置"""
        return bool(
            settings.baidu_ocr_app_id
            and settings.baidu_ocr_api_key
            and settings.baidu_ocr_secret_key
        )

    @property
    def baidu(self):
        """延迟加载百度 OCR 客户端"""
        if self._baidu_client is None and self.baidu_available:
            try:
                from aip import AipOcr

                self._baidu_client = AipOcr(
                    settings.baidu_ocr_app_id,
                    settings.baidu_ocr_api_key,
                    settings.baidu_ocr_secret_key,
                )
                logger.info("百度 OCR 客户端初始化完成")
            except Exception as e:
                logger.error(f"百度 OCR 客户端初始化失败: {e}")
                self._baidu_client = False  # 标记为不可用，避免重复尝试
        return self._baidu_client if self._baidu_client is not False else None

    async def _extract_text_baidu(self, image_bytes: bytes) -> str:
        """百度 OCR 精确版：从图片中提取文字

        Args:
            image_bytes: 图片二进制数据（JPEG/PNG）

        Returns:
            识别出的纯文本，每行一个文本块
        """
        # 调用百度 OCR 精确版（支持中英文混合）
        options = {
            "language_type": "CHN_ENG",
            "detect_direction": "true",
            "paragraph": "true",
        }
        result = self.baidu.basicAccurate(image_bytes, options)

        # 检查错误
        if "error_code" in result:
            error_msg = result.get("error_msg", "未知错误")
            raise RuntimeError(f"百度 OCR 返回错误 (code={result['error_code']}): {error_msg}")

        # 提取文字
        words_result = result.get("words_result", [])
        lines = [item["words"] for item in words_result if item.get("words")]

        logger.info(f"百度 OCR 识别到 {len(lines)} 行文字")
        return "\n".join(lines)

    # ── EasyOCR（降级回退）─────────────────────────────────

    @property
    def ocr(self):
        """延迟加载 EasyOCR Reader（首次使用下载模型 ~100MB，需 10-30 秒）"""
        if self._ocr is None:
            logger.info("正在初始化 EasyOCR（首次加载会下载中文识别模型，请稍候）...")
            try:
                import easyocr

                # 中文简体和英文，不使用 GPU
                self._ocr = easyocr.Reader(
                    ["ch_sim", "en"],
                    gpu=False,
                    verbose=False,
                )
                logger.info("EasyOCR 初始化完成（降级引擎就绪）")
            except Exception as e:
                logger.error(f"EasyOCR 初始化失败: {e}")
                raise RuntimeError(f"EasyOCR 初始化失败: {e}")
        return self._ocr

    async def _extract_text_easyocr(self, image_bytes: bytes) -> str:
        """EasyOCR 从图片中提取文字（降级引擎）

        Args:
            image_bytes: 图片二进制数据（JPEG/PNG）

        Returns:
            识别出的纯文本，每行一个文本块
        """
        # 图片二进制 → numpy array
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_np = np.array(image)

        # EasyOCR 识别
        results = self.ocr.readtext(image_np)

        # 提取文本，按置信度过滤
        lines = []
        for bbox, text, confidence in results:
            if confidence > 0.3:
                lines.append(text)

        raw_text = "\n".join(lines)
        logger.info(f"EasyOCR 识别到 {len(lines)} 行文字")
        return raw_text

    # ── 统一入口（带降级）─────────────────────────────────

    async def extract_text(self, image_bytes: bytes) -> str:
        """从图片中提取文字，优先百度 OCR，失败时降级到 EasyOCR

        Args:
            image_bytes: 图片二进制数据（JPEG/PNG）

        Returns:
            识别出的纯文本
        """
        # 优先尝试百度 OCR
        if self.baidu_available:
            try:
                return await self._extract_text_baidu(image_bytes)
            except Exception as e:
                logger.warning(f"百度 OCR 失败，降级到 EasyOCR: {e}")

        # 降级到 EasyOCR
        logger.info("使用 EasyOCR 降级引擎...")
        return await self._extract_text_easyocr(image_bytes)

    # ── DeepSeek 结构化 ────────────────────────────────────

    @property
    def deepseek(self) -> OpenAI:
        """延迟加载 DeepSeek 客户端"""
        if self._deepseek_client is None:
            api_key = settings.deepseek_api_key
            if not api_key:
                raise RuntimeError("DEEPSEEK_API_KEY 未配置")
            self._deepseek_client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com",
            )
        return self._deepseek_client

    async def structure_lab_data(self, ocr_text: str) -> dict:
        """用 DeepSeek 将 OCR 文本结构化为 21 项化验指标

        Args:
            ocr_text: OCR 提取的原始文本

        Returns:
            {"lab_data": {"wbc": 16.8, ...}}
        """
        if not ocr_text.strip():
            return {"lab_data": {}}

        response = self.deepseek.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": LAB_FIELDS_PROMPT},
                {
                    "role": "user",
                    "content": f"请从以下化验单 OCR 文本中提取指标：\n\n{ocr_text}",
                },
            ],
            temperature=0.0,
            max_tokens=2000,
        )

        content = response.choices[0].message.content.strip()

        # 清理可能的 markdown 代码块包裹
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content[:-3]

        try:
            parsed = json.loads(content)
            lab_data = parsed.get("lab_data", {})

            # 只保留数值类型
            cleaned = {}
            for key, val in lab_data.items():
                if isinstance(val, (int, float)):
                    cleaned[key] = float(val)
            return {"lab_data": cleaned}
        except json.JSONDecodeError as e:
            logger.error(f"DeepSeek JSON 解析失败: {content[:200]}")
            raise RuntimeError(f"AI 结构化解析失败: {e}")

    async def process(self, image_bytes: bytes) -> dict:
        """完整 OCR 流程：图片 → 百度 OCR/EasyOCR 文字提取 → DeepSeek 结构化

        Args:
            image_bytes: 化验单图片

        Returns:
            {"lab_data": {"wbc": 16.8, ...}, "raw_text": "OCR 原始文本"}
        """
        # Step 1: OCR 文字提取（百度优先，EasyOCR 降级）
        raw_text = await self.extract_text(image_bytes)
        if not raw_text.strip():
            raise RuntimeError(
                "未能从图片中识别到文字，请确认图片清晰且包含化验数据"
            )

        # Step 2: DeepSeek 结构化
        result = await self.structure_lab_data(raw_text)
        result["raw_text"] = raw_text

        return result


# 全局单例
ocr_service = OCRService()
