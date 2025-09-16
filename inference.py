import tensorflow as tf
import numpy as np
import cv2
import os

# --- 1. 配置与常量定义 ---
IMG_WIDTH = 320
IMG_HEIGHT = 240
NUM_JOINTS = 15

# 加载训练好的模型
MODEL_PATH = 'pose_estimation_mobilenet_v3_small.h5'
print(f"正在加载模型: {MODEL_PATH}")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("模型加载成功。")
    model.summary()
except Exception as e:
    print(f"模型加载失败: {e}")
    exit()

def preprocess_image(image_path):
    """
    加载并预处理单张深度图以进行推理。
    """
    if not os.path.exists(image_path):
        print(f"错误: 图像文件不存在于 {image_path}")
        return None, None

    # 使用 OpenCV 加载图像，并保持其原始颜色（用于可视化）
    original_image = cv2.imread(image_path)
    # 将其转换为灰度图进行处理
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    # 调整图像大小以匹配模型输入
    resized_image = cv2.resize(gray_image, (IMG_WIDTH, IMG_HEIGHT))
    
    # 归一化到 [0, 1]
    normalized_image = resized_image.astype(np.float32) / 255.0
    
    # 扩展维度以匹配模型的输入形状 (1, height, width, 1)
    input_tensor = np.expand_dims(normalized_image, axis=-1)
    input_tensor = np.expand_dims(input_tensor, axis=0)
    
    return input_tensor, original_image

def predict_and_visualize(image_path):
    """
    对单张图像进行姿态估计并可视化结果。
    """
    # --- 2. 图像预处理 ---
    input_tensor, original_image = preprocess_image(image_path)
    if input_tensor is None:
        return

    # --- 3. 模型推理 ---
    print("正在进行推理...")
    predicted_normalized_coords = model.predict(input_tensor)
    print("推理完成。")

    # --- 4. 结果后处理 ---
    # 将扁平化的输出重塑为 (NUM_JOINTS, 2)
    reshaped_coords = predicted_normalized_coords.reshape((NUM_JOINTS, 2))
    
    # 获取原始图像的尺寸
    orig_h, orig_w, _ = original_image.shape
    
    # 将归一化的坐标反转为原始图像尺寸的像素坐标
    # 注意：我们在训练时将坐标归一化到了 (320, 240)，所以这里要用 (320, 240) 来反归一化
    # 然后再根据原始图像和320x240的比例进行缩放
    scale_x = orig_w / IMG_WIDTH
    scale_y = orig_h / IMG_HEIGHT
    
    predicted_pixels = reshaped_coords * np.array([IMG_WIDTH, IMG_HEIGHT])
    predicted_pixels[:, 0] *= scale_x
    predicted_pixels[:, 1] *= scale_y
    predicted_pixels = predicted_pixels.astype(int)

    # --- 5. 结果可视化 ---
    # 在原始图像上绘制关节点
    vis_image = original_image.copy()
    for i, (x, y) in enumerate(predicted_pixels):
        # 绘制一个红色的圆圈代表关节点
        cv2.circle(vis_image, (x, y), 5, (0, 0, 255), -1)
        # 在关节点旁边标注编号
        cv2.putText(vis_image, str(i), (x + 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # 保存结果图像
    output_filename = "inference_result.jpg"
    cv2.imwrite(output_filename, vis_image)
    print(f"结果已保存为: {output_filename}")

    # 显示结果图像
    cv2.imshow('Pose Estimation Result', vis_image)
    print("按任意键退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- 主程序入口 ---
if __name__ == '__main__':
    # ******************************************************************
    # ** 请将这里的路径替换为你要测试的一张深度图的实际路径 **
    # ******************************************************************
    TEST_IMAGE_PATH = 'C:/Users/29172/Downloads/side_test_images/1_00000.jpg'  # 示例路径

    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"\n错误: 测试图片 '{TEST_IMAGE_PATH}' 不存在。")
        print("请打开 `inference.py` 文件，并将 `TEST_IMAGE_PATH` 变量修改为一张实际存在的图片路径。")
    else:
        predict_and_visualize(TEST_IMAGE_PATH)
