"""OCR 化验单识别接口"""

from fastapi import APIRouter, File, UploadFile, HTTPException

from app.services.ocr_service import ocr_service

router = APIRouter(prefix="/ocr", tags=["ocr"])


@router.post("")
async def ocr_lab_report(file: UploadFile = File(...)):
    """上传化验单图片，返回 OCR 识别并结构化后的数据

    Args:
        file: 化验单图片（JPG/PNG），支持拍照或扫描件

    Returns:
        {
            "lab_data": {"wbc": 16.8, "crp": 78.0, ...},
            "raw_text": "OCR 识别原始文本"
        }
    """
    # 校验文件类型
    if file.content_type not in (
        "image/jpeg",
        "image/png",
        "image/jpg",
        "image/webp",
    ):
        raise HTTPException(
            status_code=400,
            detail="仅支持 JPG/PNG/WebP 格式的图片",
        )

    # 读取图片二进制
    try:
        image_bytes = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="无法读取上传文件")

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="上传文件为空")

    if len(image_bytes) > 10 * 1024 * 1024:  # 10MB 限制
        raise HTTPException(status_code=400, detail="图片大小不能超过 10MB")

    # OCR 识别 + 结构化
    try:
        result = await ocr_service.process(image_bytes)
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR 识别失败: {e}")

    return result
