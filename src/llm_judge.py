"""LLM judge layer: ask an OpenAI-compatible multimodal model for final selection."""

from __future__ import annotations

import base64
import io
from PIL import Image
import json
import mimetypes
from pathlib import Path
from typing import Any

from src import config


SYSTEM_PROMPT = """你是一位极其苛刻的摄影艺术总监。
你要从候选照片中选出最有高级感、情绪张力和决定性瞬间的一张。
请重点评估：情绪浓度、眼神光、光影质感、构图秩序、主体姿态、瞬间稀缺性、废片风险。
不要平均主义，不要选择只是清晰但无情绪的照片。
你必须只输出 JSON，不要输出 Markdown。格式：
{"best_photo_index": 0, "reasoning": "简短但具体的选择理由"}"""


class LLMJudge:
    def __init__(
        self,
        api_key: str = config.LLM_API_KEY,
        base_url: str = config.LLM_BASE_URL,
        model: str = config.LLM_MODEL,
        timeout: float = config.LLM_TIMEOUT_SECONDS,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout = timeout

    @staticmethod
    def _image_to_data_url(image_path: Path) -> str:
        """读取图片，压缩瘦身后再转成 base64 Data URL，防止触发 413 报错"""
        with Image.open(image_path) as img:
            # 1. 转换为通用的 RGB 模式（丢弃可能存在的透明通道）
            img = img.convert("RGB")

            # 2. 等比例缩小：限制最大边长为 1024 像素（大模型看图完全够用了）
            img.thumbnail((1024, 1024))

            # 3. 将压缩后的图片存入内存缓冲区
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)  # 使用 85% 质量的 JPEG 压缩

            # 4. 转换为 base64 字符串并拼接成标准协议头
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{encoded}"

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start >= 0 and end > start:
                return json.loads(content[start : end + 1])
            raise

    def judge_candidates(
        self,
        candidate_paths: list[str | Path],
        reference_image_path: str | Path | None = None,
    ) -> dict[str, Any]:
        candidate_files = [Path(path) for path in candidate_paths]
        candidate_files = [path for path in candidate_files if path.exists()]

        if not candidate_files:
            return {
                "best_photo_index": None,
                "reasoning": "没有可用候选图片，无法调用终审。",
                "error": "missing_candidates",
            }

        if not self.api_key or not self.base_url or not self.model:
            return {
                "best_photo_index": 0,
                "reasoning": "LLM API 未配置，临时沿用本地评分第一名。",
                "fallback": True,
            }

        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "请在候选图中选择唯一最佳照片。"
                    "候选图索引从 0 开始，对应我发送的候选图顺序。"
                ),
            }
        ]

        reference_path = Path(reference_image_path) if reference_image_path else None
        if reference_path and reference_path.exists():
            content.append(
                {"type": "text", "text": "以下是大师参考图，用于校准审美标准。"}
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._image_to_data_url(reference_path)},
                }
            )

        for index, image_path in enumerate(candidate_files):
            content.append(
                {"type": "text", "text": f"候选图 index={index}: {image_path.name}"}
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._image_to_data_url(image_path)},
                }
            )

        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
            response = client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
            )
            raw_content = response.choices[0].message.content or "{}"
            parsed = self._parse_json(raw_content)
            parsed["candidate_files"] = [path.name for path in candidate_files]
            print(f"🎩 LLM 终审完成: best_photo_index={parsed.get('best_photo_index')}")
            return parsed
        except Exception as exc:
            print(f"🔴 LLM 终审失败，沿用本地评分第一名: {exc}")
            return {
                "best_photo_index": 0,
                "reasoning": "LLM 调用失败，临时沿用本地评分第一名。",
                "error": str(exc),
                "fallback": True,
            }


if __name__ == "__main__":
    config.ensure_project_dirs()
    sample_images = sorted(config.RAW_PHOTOS_DIR.glob("*"))[:3]
    judge = LLMJudge()
    print(judge.judge_candidates(sample_images, config.MASTER_REFERENCE_IMAGE))
