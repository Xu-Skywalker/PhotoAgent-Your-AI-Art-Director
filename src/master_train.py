"""
Training Pipeline: 提取大师作品特征并生成质心模型
"""

import numpy as np
from src import config
from src.perception import ImagePerceiver


def train_master_aesthetic():
    print("🚀 启动大师审美训练管线...")

    # 1. 确保目录存在
    config.ensure_project_dirs()

    # 2. 复用感知层，从大师文件夹提取特征，存入大师缓存文件夹
    perceiver = ImagePerceiver()
    perceiver.extract_and_save_features(
        input_dir=config.MASTER_PHOTOS_DIR, cache_dir=config.MASTER_CACHE_DIR
    )

    # 3. 读取所有大师的 .npy 特征准备融合
    feature_list = []
    print(f"📥 正在从 {config.MASTER_CACHE_DIR} 读取大师特征用于计算质心...")
    for file_path in config.MASTER_CACHE_DIR.glob("*.npy"):
        try:
            vector = np.load(file_path).astype(np.float32).reshape(-1)
            if vector.size == config.FEATURE_DIM:
                feature_list.append(vector)
        except Exception:
            continue

    if not feature_list:
        print(
            "🔴 训练失败：未找到任何有效的大师特征，请检查图片是否放入了 master_photos 文件夹。"
        )
        return

    print(f"🧮 成功读取 {len(feature_list)} 张大师图片的特征，开始数学融合...")

    # 4. 核心算法：求平均质心并做 L2 归一化
    matrix = np.vstack(feature_list)
    centroid = np.mean(matrix, axis=0)
    norm = np.linalg.norm(centroid)
    centroid = centroid / max(float(norm), 1e-12)

    # 5. 保存结果到模型专属目录
    output_path = config.MASTER_CENTROID_PATH
    np.save(output_path, centroid)
    print(f"✅ 训练大功告成！大师质心已永久封存于: {output_path}")


if __name__ == "__main__":
    train_master_aesthetic()
