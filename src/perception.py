import os
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm


class ImagePerceiver:
    def __init__(self, model_id="openai/clip-vit-large-patch14"):
        """
        初始化感知层：负责将图片转化为 768 维的数学特征向量
        """
        print(f"正在初始化感知模块，加载模型: {model_id} ...")

        # 自动检测你的 RTX 4060 显卡
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 针对 40系 显卡的终极优化：开启 FP16 半精度
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        # 加载 CLIP 模型和图像预处理器
        self.model = CLIPModel.from_pretrained(
            model_id, torch_dtype=dtype, use_safetensors=True
        ).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)

        # 切换到评估模式 (极其重要：关闭训练引擎，节省极大的显存和算力)
        self.model.eval()
        print(f"模型加载完毕！当前运行设备: {self.device.upper()} (精度: {dtype})")

    def extract_and_save_features(self, input_dir, cache_dir):
        """
        核心工作流：扫描图片 -> 提取特征 -> 保存为 .npy 硬盘文件
        """
        # 确保输出的缓存目录存在，如果不存在则自动创建
        os.makedirs(cache_dir, exist_ok=True)

        # 定义支持的图片格式
        valid_extensions = (".png", ".jpg", ".jpeg", ".webp")

        # 扫描文件夹里的所有图片文件
        image_files = [
            f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)
        ]

        if not image_files:
            print(f"在目录 '{input_dir}' 中没有找到任何图片。")
            return

        print(f"发现 {len(image_files)} 张图片，开始提取特征向量...")

        # 记录成功提取的数量
        success_count = 0

        # tqdm 会在控制台为你画出一个非常漂亮的进度条
        for filename in tqdm(image_files, desc="处理进度", unit="张"):
            image_path = os.path.join(input_dir, filename)

            # 构造 .npy 文件的保存路径 (例如: DSC_001.jpg -> DSC_001.npy)
            base_name = os.path.splitext(filename)[0]
            cache_path = os.path.join(cache_dir, f"{base_name}.npy")

            # 【防抖设计】：如果特征已经存在，直接跳过，支持随时中断和恢复
            if os.path.exists(cache_path):
                success_count += 1
                continue

            try:
                # 1. 读取图片并转化为 RGB 格式
                image = Image.open(image_path).convert("RGB")

                # 2. 预处理图片并推入显卡
                inputs = self.processor(images=image, return_tensors="pt").to(
                    self.device
                )

                # 3. 提取特征 (使用 torch.no_grad() 彻底切断反向传播，保护显存)
                with torch.no_grad():
                    # 第一步：绕过多模态总闸，直接只调用“视觉子模型”
                    vision_outputs = self.model.vision_model(
                        pixel_values=inputs["pixel_values"]
                    )

                    # 第二步：从大礼包对象中，强行拿出池化后的纯数学张量
                    pooled_output = vision_outputs.pooler_output

                    # 第三步：将其穿过投影层，精准降维到 CLIP 标准的 512 维特征
                    features = self.model.visual_projection(pooled_output)

                # 4. L2 归一化 (将向量长度压缩为1，这是后续计算余弦相似度的前提)
                features = features / features.norm(p=2, dim=-1, keepdim=True)

                # 5. 从显卡抓回内存，转换为普通浮点数 Numpy 数组并保存落盘
                feature_array = features.cpu().numpy().astype(np.float32)
                np.save(cache_path, feature_array)

                success_count += 1

            except Exception as e:
                print(f"\n处理图片 {filename} 时发生错误: {e}")

        print(
            f"\n提取完成！成功处理: {success_count}/{len(image_files)} 张图片。特征已存入 {cache_dir}"
        )


# ================= 用于本地单独测试的入口 =================
if __name__ == "__main__":
    # 这里的路径是相对于你运行终端的目录 (项目根目录)
    # 确保你已经在 data/raw_photos 里放了几张测试照片
    RAW_PHOTOS_DIR = "data/raw_photos"
    CACHE_DIR = "data/cache"

    # 实例化感知器
    perceiver = ImagePerceiver()

    # 执行提取任务
    perceiver.extract_and_save_features(RAW_PHOTOS_DIR, CACHE_DIR)
