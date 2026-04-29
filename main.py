# main.py
# 这是整个 Agent 项目的总入口

import os

# 从你的 src 文件夹中的 perception.py 文件里，导入 ImagePerceiver 这个类
from src.perception import ImagePerceiver


def run_pipeline():
    print("摄影 Agent 核心管线启动...")

    # 1. 配置路径 (未来这些会移到 config.py 里)
    RAW_PHOTOS_DIR = "data/raw_photos"
    CACHE_DIR = "data/cache"

    # 2. 阶段一：感知层特征提取
    print("\n--- [阶段一]：图像感知与特征提取 ---")
    perceiver = ImagePerceiver()
    perceiver.extract_and_save_features(RAW_PHOTOS_DIR, CACHE_DIR)

    # 3. 阶段二：聚类与分组 (预留位置)
    # print("\n--- [阶段二]：相似图像聚类 ---")
    # TODO: 之后我们会在这里调用 clustering.py 的代码


if __name__ == "__main__":
    # 当你运行 python main.py 时，执行整个流水线
    run_pipeline()
