"""Perception layer: extract CLIP visual features and cache them as .npy files."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from src import config


class ImagePerceiver:
    """Extract 768-dim CLIP visual embeddings with the vision tower only."""

    def __init__(self, model_id: str = config.CLIP_MODEL_ID) -> None:
        self.model_id = model_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model: CLIPModel | None = None
        self.processor: CLIPProcessor | None = None

        print(f"🧬 初始化感知层: model={model_id}")
        if self.device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            print(f"⚡ 检测到 CUDA: {gpu_name}，需要提取新特征时会启用 FP16")
        else:
            print("🟡 未检测到 CUDA，将使用 CPU + FP32 推理，速度会明显变慢")
        print("✅ 感知层就绪，模型将按需懒加载")

    def _load_model(self) -> None:
        if self.model is not None and self.processor is not None:
            return

        print(f"🧠 正在加载 CLIP 视觉模型: {self.model_id}")
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        self.model = CLIPModel.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            use_safetensors=True,
        ).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        self.model.eval()

        projection_dim = int(getattr(self.model.config, "projection_dim", 0) or 0)
        if projection_dim != config.FEATURE_DIM:
            print(
                f"🟡 当前模型 projection_dim={projection_dim}，"
                f"配置期望={config.FEATURE_DIM}，后续会按实际输出校验。"
            )
        print(f"✅ 模型加载完成: device={self.device}, dtype={self.dtype}")

    @staticmethod
    def _iter_images(input_dir: Path) -> list[Path]:
        if not input_dir.exists():
            print(f"🔴 原图目录不存在: {input_dir}")
            return []
        return sorted(
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in config.IMAGE_EXTENSIONS
        )

    @staticmethod
    def _is_valid_cache(cache_path: Path, expected_dim: int = config.FEATURE_DIM) -> bool:
        if not cache_path.exists() or cache_path.stat().st_size == 0:
            return False
        try:
            vector = np.load(cache_path)
        except Exception:
            return False
        return vector.size == expected_dim and np.isfinite(vector).all()

    def _extract_one(self, image_path: Path) -> np.ndarray:
        self._load_model()
        assert self.model is not None
        assert self.processor is not None

        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")

        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)

        with torch.inference_mode():
            # 关键路径：只调用 CLIP 视觉塔，再通过 visual_projection 得到 768 维图像特征。
            vision_outputs = self.model.vision_model(pixel_values=pixel_values)
            pooled_output = vision_outputs.pooler_output
            features = self.model.visual_projection(pooled_output)
            features = features / features.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)

        vector = features.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if vector.size != config.FEATURE_DIM:
            raise ValueError(f"特征维度异常: got={vector.size}, expected={config.FEATURE_DIM}")
        return vector

    @staticmethod
    def _atomic_save(cache_path: Path, vector: np.ndarray) -> None:
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with tmp_path.open("wb") as handle:
            np.save(handle, vector)
        os.replace(tmp_path, cache_path)

    def extract_and_save_features(
        self,
        input_dir: str | Path,
        cache_dir: str | Path,
        force: bool = False,
    ) -> dict[str, int]:
        """Scan images, extract features, and persist one [photo_stem].npy per image."""
        input_path = Path(input_dir)
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        image_files = self._iter_images(input_path)
        if not image_files:
            print(f"🟡 在目录 {input_path} 中没有找到可处理图片")
            return {"total": 0, "processed": 0, "skipped": 0, "failed": 0}

        processed = 0
        skipped = 0
        failed = 0
        print(f"📸 发现 {len(image_files)} 张图片，开始提取 768 维视觉特征")

        for image_path in tqdm(image_files, desc="🧬 感知进度", unit="张"):
            output_path = cache_path / f"{image_path.stem}.npy"

            if not force and self._is_valid_cache(output_path):
                skipped += 1
                continue

            try:
                vector = self._extract_one(image_path)
                self._atomic_save(output_path, vector)
                processed += 1
            except Exception as exc:
                failed += 1
                print(f"\n🔴 处理失败: {image_path.name} -> {exc}")
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        print(
            f"✅ 感知层完成: 新处理 {processed} 张，跳过缓存 {skipped} 张，失败 {failed} 张"
        )
        return {
            "total": len(image_files),
            "processed": processed,
            "skipped": skipped,
            "failed": failed,
        }


if __name__ == "__main__":
    config.ensure_project_dirs()
    perceiver = ImagePerceiver()
    perceiver.extract_and_save_features(config.RAW_PHOTOS_DIR, config.CACHE_DIR)
