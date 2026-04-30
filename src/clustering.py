"""Clustering layer: group similar burst photos with DBSCAN on cosine distance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances

from src import config


class ImageClusterer:
    def __init__(
        self,
        eps: float = config.DBSCAN_EPS,
        min_samples: int = config.DBSCAN_MIN_SAMPLES,
        expected_dim: int = config.FEATURE_DIM,
    ) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.expected_dim = expected_dim
        print(
            f"📦 初始化聚类层: DBSCAN eps={eps}, min_samples={min_samples}, "
            f"metric=precomputed"
        )

    def load_features(self, cache_dir: str | Path) -> tuple[np.ndarray | None, list[str]]:
        cache_path = Path(cache_dir)
        if not cache_path.exists():
            print(f"🔴 特征缓存目录不存在: {cache_path}")
            return None, []

        feature_list: list[np.ndarray] = []
        filename_list: list[str] = []
        skipped = 0

        for file_path in sorted(cache_path.glob("*.npy")):
            try:
                vector = np.load(file_path).astype(np.float32).reshape(-1)
                if vector.size != self.expected_dim:
                    skipped += 1
                    print(
                        f"🟡 跳过维度异常的特征: {file_path.name} "
                        f"({vector.size} != {self.expected_dim})"
                    )
                    continue
                if not np.isfinite(vector).all():
                    skipped += 1
                    print(f"🟡 跳过包含 NaN/Inf 的特征: {file_path.name}")
                    continue
                feature_list.append(vector)
                filename_list.append(file_path.stem)
            except Exception as exc:
                skipped += 1
                print(f"🔴 读取特征失败: {file_path.name} -> {exc}")

        if not feature_list:
            print("🟡 缓存目录中没有可用特征，请先运行 perception.py")
            return None, []

        if skipped:
            print(f"🟡 已跳过 {skipped} 个不可用特征文件")

        features = np.vstack(feature_list).astype(np.float32)
        return features, filename_list

    def run(self, cache_dir: str | Path) -> dict[int, list[str]]:
        features, filenames = self.load_features(cache_dir)
        if features is None or len(filenames) < self.min_samples:
            print("🟡 可聚类照片数量不足")
            return {}

        print(f"🧮 正在为 {len(filenames)} 张照片计算余弦距离矩阵")
        dist_matrix = cosine_distances(features).astype(np.float32)
        dist_matrix = np.clip(dist_matrix, 0.0, 2.0)

        dbscan = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric="precomputed",
        )
        labels = dbscan.fit_predict(dist_matrix)

        groups: dict[int, list[str]] = {}
        noise_count = 0
        for idx, label in enumerate(labels):
            if label == -1:
                noise_count += 1
                continue
            groups.setdefault(int(label), []).append(filenames[idx])

        print(
            f"✅ 聚类完成: 发现 {len(groups)} 个连拍组，"
            f"剔除独立照片 {noise_count} 张"
        )
        for group_id, photos in groups.items():
            print(f"  🧺 组 {group_id}: {len(photos)} 张 -> {photos}")
        return groups


if __name__ == "__main__":
    config.ensure_project_dirs()
    clusterer = ImageClusterer()
    found_groups = clusterer.run(config.CACHE_DIR)
    print(found_groups)
