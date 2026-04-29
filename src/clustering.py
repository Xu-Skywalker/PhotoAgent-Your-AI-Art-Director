import os
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances


class ImageClusterer:
    def __init__(self, eps=0.05, min_samples=2):
        """
        初始化聚类模块
        :param eps: 邻域半径（核心参数）。
                    值越小，要求照片越像才能分到一组。
                    经验值：0.05 ~ 0.1 适合找连拍；0.2 ~ 0.3 适合找同一场景。
        :param min_samples: 最少几张照片才能构成一个“组”。
        """
        self.eps = eps
        self.min_samples = min_samples
        print(f"初始化聚类模块 (阈值 eps={eps}, 最小成组数={min_samples})...")

    def load_features(self, cache_dir):
        """
        从硬盘读取提取好的 .npy 特征
        """
        feature_list = []
        filename_list = []

        if not os.path.exists(cache_dir):
            print(f"找不到缓存目录: {cache_dir}")
            return None, None

        # 扫描缓存文件夹
        for file in os.listdir(cache_dir):
            if file.endswith(".npy"):
                try:
                    # 加载向量并确保维度正确 (512,)
                    feat = np.load(os.path.join(cache_dir, file)).flatten()
                    feature_list.append(feat)
                    # 记录对应的原图名（去除 .npy 后缀）
                    filename_list.append(file.replace(".npy", ""))
                except Exception as e:
                    print(f"读取 {file} 失败: {e}")

        if not feature_list:
            print("缓存目录为空，请先运行 perception.py 提取特征。")
            return None, None

        return np.array(feature_list), filename_list

    def run(self, cache_dir):
        """
        执行聚类流程
        """
        features, filenames = self.load_features(cache_dir)
        if features is None:
            return {}

        print(f"正在对 {len(filenames)} 张照片进行空间距离计算...")

        # 1. 计算余弦距离矩阵 (1 - 余弦相似度)
        # 距离越接近 0，表示照片越相似
        dist_matrix = cosine_distances(features)

        # 2. 运行 DBSCAN 算法
        # metric='precomputed' 表示我们已经自己算好距离矩阵了
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="precomputed")
        labels = db.fit_predict(dist_matrix)

        # 3. 整理结果
        groups = {}
        for idx, label in enumerate(labels):
            if label == -1:
                # -1 代表噪声点，即没有相似照片的独立图，我们暂时跳过
                continue

            if label not in groups:
                groups[label] = []
            groups[label].append(filenames[idx])

        print(f"聚类完成！共发现 {len(groups)} 个相似连拍组。")
        return groups


if __name__ == "__main__":
    # 单独测试逻辑
    CACHE_DIR = "data/cache"
    # 如果你发现连拍的照片没被分到一起，就调大 eps（如 0.08）
    # 如果你发现完全不同的照片被误分到一起，就调小 eps（如 0.03）
    clusterer = ImageClusterer(eps=0.2)
    groups = clusterer.run(CACHE_DIR)

    for gid, photos in groups.items():
        print(f"组 {gid}: {photos}")
