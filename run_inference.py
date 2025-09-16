joint_id_to_name = {
    0: 'Head', 1: 'Neck', 2: 'R Shoulder', 3: 'L Shoulder', 4: 'R Elbow', 5: 'L Elbow',
    6: 'R Hand', 7: 'L Hand', 8: 'Torso', 9: 'R Hip', 10: 'L Hip', 11: 'R Knee',
    12: 'L Knee', 13: 'R Foot', 14: 'L Foot'
}
import tensorflow as tf
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
from PIL import Image

# 配置参数
MODEL_PATH = 'pose_estimation_mobilenet_v3_small.h5'  # 模型文件路径（相对路径）
TEST_IMG_DIR = 'C:/Users/29172/Downloads/side_test_images'  # 测试图片文件夹
TEST_LABEL_PATH = 'C:/Users/29172/Downloads/ITOP_side_test_labels.h5'  # 测试标签h5文件
IMG_WIDTH = 320
IMG_HEIGHT = 240
NUM_JOINTS = 15
JOINT_DIMENSIONS = 2
BATCH_SIZE = 32  # Keras推理时可以使用批量

# 1. 加载Keras模型
print('加载Keras模型...')
model = tf.keras.models.load_model(MODEL_PATH)
print('Keras模型加载完成')
model.summary()

# 2. 加载测试数据
print('加载测试数据...')
def load_test_data(img_dir, label_path):
    with h5py.File(label_path, 'r') as f:
        coords_all = np.array(f['image_coordinates'])  # (10501, 15, 2)
        ids_all = np.array(f['id'])  # (10501,)
        ids_all = [id_.decode('utf-8') if isinstance(id_, bytes) else str(id_) for id_ in ids_all]
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')])
    imgs = []
    coords = []
    for fname in img_files:
        frame_id = os.path.splitext(fname)[0]
        if frame_id in ids_all:
            idx = ids_all.index(frame_id)
            img = Image.open(os.path.join(img_dir, fname)).convert('L').resize((IMG_WIDTH, IMG_HEIGHT))
            img = np.array(img, dtype=np.float32) / 255.0
            imgs.append(img)
            coords.append(coords_all[idx])
    imgs = np.stack(imgs, axis=0)[..., np.newaxis]
    coords = np.stack(coords, axis=0)
    return imgs, coords

x_test, y_test = load_test_data(TEST_IMG_DIR, TEST_LABEL_PATH)
print(f'测试集样本数: {x_test.shape[0]}')

# 用户控制推理范围的函数
def run_inference_on_range(start_idx, end_idx):
    """在指定索引范围内运行推理并评估"""
    if start_idx < 0 or end_idx > len(x_test) or start_idx >= end_idx:
        print(f"错误：索引范围 [{start_idx}, {end_idx}) 无效。有效范围是 [0, {len(x_test)})。")
        return

    print(f"开始在索引 {start_idx} 到 {end_idx} 的图像上运行推理...")

    # 使用Keras模型进行批量推理
    print("正在准备数据并进行批量预测...")
    x_subset = x_test[start_idx:end_idx]
    
    # Keras的predict方法是高度优化的，会自动使用GPU（如果配置正确）
    # 它返回一个numpy数组，形状为 (num_samples, num_joints * joint_dims)
    y_pred_flat = model.predict(x_subset, batch_size=BATCH_SIZE)
    
    # 将预测结果重塑为 (num_samples, num_joints, joint_dims)
    y_pred_normalized = y_pred_flat.reshape((-1, NUM_JOINTS, JOINT_DIMENSIONS))

    # !!! 关键修复：反归一化预测坐标 !!!
    y_pred = y_pred_normalized.copy()
    y_pred[..., 0] *= IMG_WIDTH  # 将x坐标乘回图像宽度
    y_pred[..., 1] *= IMG_HEIGHT # 将y坐标乘回图像高度
    
    print(f"批量预测完成，共处理 {len(x_subset)} 张图像。")

    y_true_subset = y_test[start_idx:end_idx]
    
    # 评估预测结果
    evaluate_predictions(y_pred, y_true_subset)
    
    # 可视化结果
    visualize_results(x_subset, y_true_subset, y_pred, start_idx)

# 评估模型性能的函数
def evaluate_predictions(y_pred, y_true):
    """评估预测结果"""
    mae = np.mean(np.abs(y_pred - y_true))
    mse = np.mean((y_pred - y_true) ** 2)
    
    # 计算每个关节点的误差
    joint_mae = np.mean(np.abs(y_pred - y_true), axis=(0, 2))
    
    print(f'整体 MAE: {mae:.4f}, MSE: {mse:.4f}')
    print('各关节点 MAE:')
    for i, joint_name in joint_id_to_name.items():
        print(f'  {joint_name}: {joint_mae[i]:.4f}')
    
    return mae, mse, joint_mae

# 交互式推理控制
def interactive_inference():
    """交互式推理控制"""
    print(f'\n=== 交互式推理控制 ===')
    print(f'数据总量: {x_test.shape[0]}')
    print('使用示例:')
    print('  推理前100个样本: run_inference_on_range(0, 100)')
    print('  推理第500-600个样本: run_inference_on_range(500, 600)')
    print('  推理最后50个样本: run_inference_on_range(-50, None)')
    
    while True:
        try:
            print(f'\n当前数据范围: [0, {x_test.shape[0]})')
            start_str = input('请输入起始索引 (或输入 q 退出): ').strip()
            
            if start_str.lower() == 'q':
                print('退出交互模式')
                break
            
            start_idx = int(start_str)
            end_str = input('请输入结束索引: ').strip()
            end_idx = int(end_str)
            
            # 处理负数索引
            if start_idx < 0:
                start_idx = x_test.shape[0] + start_idx
            if end_idx < 0:
                end_idx = x_test.shape[0] + end_idx
            
            # 运行推理
            run_inference_on_range(start_idx, end_idx)
            
        except ValueError as e:
            print(f'输入错误: {e}')
        except KeyboardInterrupt:
            print('\n用户中断')
            break

# 可视化结果函数
def visualize_results(x_data, y_true, y_pred, start_idx=0, max_samples=5):
    """可视化预测结果"""
    num_samples = min(max_samples, len(x_data))
    print(f'可视化前 {num_samples} 个样本的预测结果...')
    for i in range(num_samples):
        plot_keypoints(x_data[i], y_true[i], y_pred[i], start_idx + i)

def plot_keypoints(img, gt, pred, idx):
    plt.figure(figsize=(6, 4))
    plt.imshow(img.squeeze(), cmap='gray')
    plt.scatter(gt[:, 0], gt[:, 1], c='g', label='GT', s=30)
    plt.scatter(pred[:, 0], pred[:, 1], c='r', marker='x', label='Pred', s=30)
    for i in range(NUM_JOINTS):
        plt.plot([gt[i, 0], pred[i, 0]], [gt[i, 1], pred[i, 1]], 'y--', linewidth=0.5)
        plt.text(gt[i, 0], gt[i, 1], joint_id_to_name[i], color='white', fontsize=7, ha='right', va='bottom')
    plt.title(f'Sample {idx}')
    plt.legend()
    plt.show()

# 启动交互式推理模式
if __name__ == "__main__":
    # 示例：推理前10个样本
    print('\n=== 示例：推理前10个样本 ===')
    run_inference_on_range(0, 10)
    
    # 启动交互模式
    interactive_inference()
