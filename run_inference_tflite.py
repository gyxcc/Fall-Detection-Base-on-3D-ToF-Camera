import tensorflow as tf
import numpy as np
import h5py
from PIL import Image
import matplotlib.pyplot as plt
import os
import glob

# 配置参数
MODEL_PATH = 'C:/Users/29172/Downloads/FYP/maxisense-main/MobileNet/pose_estimation_mobilenet_v3_small.tflite'
TEST_IMG_DIR = 'C:/Users/29172/Downloads/side_test_images'
TEST_LABEL_PATH = 'C:/Users/29172/Downloads/ITOP_side_test_labels.h5'
IMG_WIDTH = 320
IMG_HEIGHT = 240
NUM_JOINTS = 15
JOINT_DIMENSIONS = 2

# 关节点ID到名称的映射
joint_id_to_name = {
    0: 'Head', 1: 'Neck', 2: 'R_Shoulder', 3: 'L_Shoulder', 4: 'R_Elbow',
    5: 'L_Elbow', 6: 'R_Hand', 7: 'L_Hand', 8: 'Torso', 9: 'R_Hip',
    10: 'L_Hip', 11: 'R_Knee', 12: 'L_Knee', 13: 'R_Foot', 14: 'L_Foot'
}

# 1. 加载TFLite模型
print('加载TFLite模型...')
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print('TFLite模型加载完成')

# 2. 加载测试数据
def load_test_data():
    """加载并预处理测试数据"""
    print("正在加载测试数据...")
    # 加载图像
    image_files = sorted(glob.glob(os.path.join(TEST_IMG_DIR, '*.jpg')))
    
    x_test = []
    for img_path in image_files:
        img = Image.open(img_path).convert('L')  # 转换为灰度图
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=-1)  # 增加通道维度
        x_test.append(img_array)
    x_test = np.array(x_test)

    # 加载标签
    with h5py.File(TEST_LABEL_PATH, 'r') as f:
        # 确保使用正确的键 'image_coordinates'
        y_test_all = np.array(f['image_coordinates'])

    # 确保标签数量和图像数量一致
    num_images = len(x_test)
    if len(y_test_all) > num_images:
        print(f"警告: 标签数量 ({len(y_test_all)}) 大于图像数量 ({num_images}). 将截断标签以匹配图像。")
        y_test = y_test_all[:num_images]
    elif len(y_test_all) < num_images:
        print(f"警告: 图像数量 ({num_images}) 大于标签数量 ({len(y_test_all)}). 将截断图像以匹配标签。")
        x_test = x_test[:len(y_test_all)]
        y_test = y_test_all
    else:
        y_test = y_test_all

    print(f"测试数据加载完成。图像: {x_test.shape}, 标签: {y_test.shape}")
    return x_test, y_test

x_test, y_test = load_test_data()

# 3. 定义推理函数
def run_inference_on_range(start_idx, end_idx):
    """在指定索引范围内运行TFLite推理"""
    if start_idx < 0 or end_idx > len(x_test) or start_idx >= end_idx:
        print(f"错误：索引范围 [{start_idx}, {end_idx}) 无效。")
        return None, None, None

    print(f"开始在索引 {start_idx} 到 {end_idx} 的图像上运行TFLite推理...")
    
    predictions = []
    for i in range(start_idx, end_idx):
        img_array = x_test[i]
        input_data = np.expand_dims(img_array, axis=0).astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        predictions.append(output_data.flatten())

    y_pred = np.array(predictions).reshape((-1, NUM_JOINTS, JOINT_DIMENSIONS))
    y_true_subset = y_test[start_idx:end_idx]
    x_subset = x_test[start_idx:end_idx]
    
    return y_pred, y_true_subset, x_subset

# 4. 定义评估函数
def evaluate_predictions(y_pred, y_true):
    """评估预测结果"""
    mae = np.mean(np.abs(y_pred - y_true))
    mse = np.mean((y_pred - y_true) ** 2)
    joint_mae = np.mean(np.abs(y_pred - y_true), axis=(0, 2))
    
    print(f'整体 MAE: {mae:.4f}, MSE: {mse:.4f}')
    print('各关节点 MAE:')
    for i, joint_name in joint_id_to_name.items():
        print(f'  {joint_name}: {joint_mae[i]:.4f}')

# 5. 定义可视化函数
def visualize_results(images, y_pred, y_true, start_idx):
    """可视化预测结果和真实标签"""
    num_samples = len(images)
    for i in range(num_samples):
        plt.figure(figsize=(8, 6))
        plt.imshow(images[i])
        
        pred_coords = y_pred[i]
        true_coords = y_true[i]
        
        plt.scatter(pred_coords[:, 0], pred_coords[:, 1], c='r', marker='x', label='Prediction')
        plt.scatter(true_coords[:, 0], true_coords[:, 1], c='b', marker='o', label='Ground Truth')
        
        for j in range(NUM_JOINTS):
            plt.text(pred_coords[j, 0], pred_coords[j, 1], joint_id_to_name[j], color='red', fontsize=8)
            plt.text(true_coords[j, 0], true_coords[j, 1], joint_id_to_name[j], color='blue', fontsize=8)

        plt.title(f'Sample {start_idx + i}')
        plt.legend()
        plt.show()

# 6. 交互式推理控制
def interactive_inference():
    """交互式推理控制"""
    print(f'\n=== 交互式TFLite推理控制 ===')
    print(f'数据总量: {x_test.shape[0]}')
    print('使用示例: 输入 "0,10" (不带引号) 来测试第0到9张图片。')
    print('输入 "exit" 退出。')
    
    while True:
        user_input = input("请输入测试范围 (start,end): ")
        if user_input.lower() == 'exit':
            break
        try:
            start_str, end_str = user_input.split(',')
            start_idx = int(start_str.strip())
            end_idx = int(end_str.strip())
            
            y_pred, y_true, x_subset = run_inference_on_range(start_idx, end_idx)
            
            if y_pred is not None:
                evaluate_predictions(y_pred, y_true)
                visualize_results(x_subset, y_pred, y_true, start_idx)

        except ValueError:
            print("输入格式错误，请输入如 '0,10' 的格式。")
        except Exception as e:
            print(f"发生错误: {e}")

if __name__ == "__main__":
    # 运行一个初始演示
    print('\n=== 运行一个初始演示 (0-5) ===')
    y_pred_demo, y_true_demo, x_demo = run_inference_on_range(0, 5)
    if y_pred_demo is not None:
        evaluate_predictions(y_pred_demo, y_true_demo)
        visualize_results(x_demo, y_pred_demo, y_true_demo, 0)

    # 进入交互式推理模式
    interactive_inference()
