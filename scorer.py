"""Scoring layer: rank candidates inside each burst group."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from joblib import load as joblib_load
from sklearn.metrics.pairwise import cosine_similarity

from src import config


class AestheticScorer:
    """Rank images by a lightweight aesthetic backend or a robust local fallback."""

    def __init__(
        self,
        cache_dir: str | Path = config.CACHE_DIR,
        top_n: int = config.SCORER_TOP_N,
        centroid_path: str | Path = config.MASTER_CENTROID_PATH,
        model_path: str | Path = config.AESTHETIC_MODEL_PATH,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.top_n = max(1, int(top_n))
        self.centroid_path = Path(centroid_path)
        self.model_path = Path(model_path)
        self.backend_name = "group_centrality_fallback"
        self.centroid: np.ndarray | None = None
        self.model: Any | None = None

        self._load_backend()
        print(f"🎚️ 初始化评分层: backend={self.backend_name}, top_n={self.top_n}")

    def _load_backend(self) -> None:
        if self.model_path.exists() and self.model_path.stat().st_size > 0:
            try:
                self.model = joblib_load(self.model_path)
                self.backend_name = f"model:{self.model_path.name}"
                return
            except Exception as exc:
                try:
                    with self.model_path.open("rb") as handle:
                        self.model = pickle.load(handle)
                    self.backend_name = f"model:{self.model_path.name}"
                    return
                except Exception:
                    print(f"🟡 审美模型加载失败，将尝试质心向量: {exc}")

        if self.centroid_path.exists() and self.centroid_path.stat().st_size > 0:
            try:
                centroid = np.load(self.centroid_path).astype(np.float32).reshape(-1)
                if centroid.size != config.FEATURE_DIM:
                    raise ValueError(
                        f"centroid dim={centroid.size}, expected={config.FEATURE_DIM}"
                    )
                norm = np.linalg.norm(centroid)
                self.centroid = centroid / max(float(norm), 1e-12)
                self.backend_name = f"centroid:{self.centroid_path.name}"
                return
            except Exception as exc:
                print(f"🟡 大师质心向量加载失败，将使用组内稳定性兜底: {exc}")

        print("🟡 未找到可用审美模型/大师质心，将使用组内中心性作为兜底评分")

    def _load_vector(self, photo_id: str) -> np.ndarray | None:
        feature_path = self.cache_dir / f"{photo_id}.npy"
        if not feature_path.exists():
            print(f"🟡 缺少特征文件: {feature_path.name}")
            return None
        try:
            vector = np.load(feature_path).astype(np.float32).reshape(-1)
            if vector.size != config.FEATURE_DIM:
                print(f"🟡 特征维度异常: {feature_path.name} -> {vector.size}")
                return None
            return vector
        except Exception as exc:
            print(f"🔴 读取特征失败: {feature_path.name} -> {exc}")
            return None

    def _score_with_model(self, matrix: np.ndarray) -> np.ndarray:
        assert self.model is not None
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(matrix)
            return np.asarray(proba)[:, -1].astype(np.float32)
        if hasattr(self.model, "decision_function"):
            raw = np.asarray(self.model.decision_function(matrix), dtype=np.float32)
            return 1.0 / (1.0 + np.exp(-raw))
        if hasattr(self.model, "predict"):
            return np.asarray(self.model.predict(matrix), dtype=np.float32).reshape(-1)
        raise TypeError("审美模型缺少 predict/predict_proba/decision_function 接口")

    def _score_group(self, photo_ids: list[str]) -> list[dict[str, Any]]:
        loaded: list[tuple[str, np.ndarray]] = []
        for photo_id in photo_ids:
            vector = self._load_vector(photo_id)
            if vector is not None:
                loaded.append((photo_id, vector))

        if not loaded:
            return []

        ids = [item[0] for item in loaded]
        matrix = np.vstack([item[1] for item in loaded]).astype(np.float32)

        try:
            if self.model is not None:
                scores = self._score_with_model(matrix)
            elif self.centroid is not None:
                scores = cosine_similarity(
                    matrix, self.centroid.reshape(1, -1)
                ).reshape(-1)
            else:
                similarity = cosine_similarity(matrix)
                scores = similarity.mean(axis=1)
        except Exception as exc:
            print(f"🔴 评分失败，改用组内中心性兜底: {exc}")
            similarity = cosine_similarity(matrix)
            scores = similarity.mean(axis=1)

        ranked = sorted(
            (
                {
                    "photo_id": photo_id,
                    "score": round(float(score), 6),
                    "backend": self.backend_name,
                }
                for photo_id, score in zip(ids, scores)
            ),
            key=lambda item: item["score"],
            reverse=True,
        )
        return ranked

    def rank_groups(
        self, groups: dict[int, list[str]]
    ) -> dict[int, list[dict[str, Any]]]:
        ranked_groups: dict[int, list[dict[str, Any]]] = {}

        for group_id, photo_ids in groups.items():
            ranked = self._score_group(photo_ids)
            top_items = ranked[: self.top_n]
            ranked_groups[group_id] = top_items
            preview = ", ".join(
                f"{item['photo_id']}({item['score']:.4f})" for item in top_items
            )
            print(f"🏅 组 {group_id} Top {len(top_items)}: {preview}")

        return ranked_groups


if __name__ == "__main__":
    config.ensure_project_dirs()
    demo_groups = {
        0: [path.stem for path in sorted(config.CACHE_DIR.glob("*.npy"))[:5]]
    }
    scorer = AestheticScorer()
    print(scorer.rank_groups(demo_groups))
