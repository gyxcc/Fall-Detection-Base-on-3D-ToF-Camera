import tensorflow as tf
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt

# 确保 TensorFlow 版本正确，支持 MobileNetV3
print(f"TensorFlow version: {tf.__version__}")

# --- 1. 数据准备 (Data Preparation) ---
# 定义图像和关节点的尺寸
IMG_WIDTH = 320
IMG_HEIGHT = 240
NUM_JOINTS = 15
JOINT_DIMENSIONS = 3  # 三维坐标 (x, y, z)
BATCH_SIZE = 32

def load_and_preprocess_itop_data(depth_file_path, labels_file_path):
    """
    加载并预处理 ITOP 数据集。
    
    参数:
    - depth_file_path: 深度图 .h5 文件的路径。
    - labels_file_path: 标签 .h5 文件的路径。
    
    返回:
    - 预处理后的深度图数据 (x)，形状为 (n, 240, 320, 1)
    - 预处理后的关节点坐标数据 (y)，形状为 (n, 15, 3)
    """
    print(f"正在加载深度图文件: {depth_file_path}")
    try:
        f_depth = h5py.File(depth_file_path, 'r')
        depth_data = np.asarray(f_depth.get('data'))
        f_depth.close()
    except Exception as e:
        print(f"加载深度图文件失败: {e}")
        return None, None
        
    print(f"正在加载标签文件: {labels_file_path}")
    try:
        f_labels = h5py.File(labels_file_path, 'r')
        # 根据你提供的文档，我们需要使用 'real_world_coordinates'
        labels_data = np.asarray(f_labels.get('real_world_coordinates'))
        # 也可以使用 'is_valid' 过滤无效数据，但在本示例中我们假设所有数据都有效
        # is_valid = np.asarray(f_labels.get('is_valid'))
        f_labels.close()
    except Exception as e:
        print(f"加载标签文件失败: {e}")
        return None, None

    print("开始预处理数据...")
    
    # 深度图预处理
    # 深度值是 float16，单位是米。我们将其归一化到 [0, 1] 范围。
    # 假设最大深度值在 3.0 米左右，这对于 ITOP 数据集是合理的。
    max_depth = 3.0 
    depth_data = depth_data.astype('float32') / max_depth
    # 将深度图从 (n, 240, 320) 转换为 (n, 240, 320, 1) 以匹配模型输入
    depth_data = np.expand_dims(depth_data, axis=-1)

    # 关节点标签预处理
    # 坐标值也是 float16，单位是米。我们同样进行归一化。
    # 为了与模型的输出层匹配，我们将 (n, 15, 3) 形状展平为 (n, 45)
    labels_data = labels_data.astype('float32')
    # 同样使用 max_depth 进行归一化，确保坐标与图像深度值在同一尺度上
    labels_data /= max_depth
    
    # 将标签数据展平
    # labels_data = labels_data.reshape((labels_data.shape[0], -1))

    print(f"处理后的深度图数据形状: {depth_data.shape}")
    print(f"处理后的关节点标签形状: {labels_data.shape}")
    
    return depth_data, labels_data

# 假设你已经下载并解压了 ITOP 数据集
# 请根据你的实际文件路径修改以下变量
TRAIN_DEPTH_FILE = 'ITOP_side_train_depth_map.h5'
TRAIN_LABELS_FILE = 'ITOP_side_train_labels.h5'
TEST_DEPTH_FILE = 'ITOP_side_test_depth_map.h5'
TEST_LABELS_FILE = 'ITOP_side_test_labels.h5'

# 加载训练集和测试集
x_train, y_train = load_and_preprocess_itop_data(TRAIN_DEPTH_FILE, TRAIN_LABELS_FILE)
x_val, y_val = load_and_preprocess_itop_data(TEST_DEPTH_FILE, TEST_LABELS_FILE)

# 如果文件加载失败，退出程序
if x_train is None or x_val is None:
    print("数据加载失败，请检查文件路径和完整性。")
    exit()

# --- 2. 模型构建 (Model Building) ---
def build_model(input_shape, num_joints, joint_dims):
    """
    构建基于 MobileNetV3 Small 的姿势识别模型。
    """
    print("构建模型...")

    # MobileNetV3 Small 需要 RGB 图像作为输入，所以我们将深度图复制三次
    input_tensor = tf.keras.Input(shape=input_shape)
    
    # 复制单通道深度图为三通道
    x = tf.keras.layers.Lambda(lambda x: tf.repeat(x, 3, axis=-1))(input_tensor)

    # 加载预训练的 MobileNetV3 Small 模型
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        alpha=1.0,
        minimalistic=False,
        include_top=False,
        weights='imagenet'
    )

    # 冻结基础模型，只训练新添加的层
    base_model.trainable = False

    # 构建我们的模型
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    
    # 添加全连接层用于姿势回归
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # 输出层，用于回归关节点坐标
    # 输出的维度是 NUM_JOINTS * JOINT_DIMENSIONS，即 15 * 3 = 45
    outputs = tf.keras.layers.Dense(num_joints * joint_dims, activation='linear')(x)
    # 在模型的输出部分，我们直接回归展平后的坐标
    
    model = tf.keras.Model(inputs=input_tensor, outputs=outputs)
    
    print("模型构建完成。")
    model.summary()
    return model

# 展平标签数据以匹配模型输出
y_train_flat = y_train.reshape((y_train.shape[0], -1))
y_val_flat = y_val.reshape((y_val.shape[0], -1))

model = build_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 1), 
                    num_joints=NUM_JOINTS, 
                    joint_dims=JOINT_DIMENSIONS)

# --- 3. 模型编译与训练 (Model Compilation and Training) ---
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='mean_squared_error',
              metrics=['mean_absolute_error'])

print("开始训练模型...")
history = model.fit(x_train, y_train_flat,
                    epochs=5, 
                    batch_size=BATCH_SIZE,
                    validation_data=(x_val, y_val_flat))

print("模型训练完成。")

# --- 4. 转换为 TensorFlow Lite 模型 (TFLite Conversion) ---
print("将模型转换为 TensorFlow Lite 格式...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

tflite_model_path = 'pose_estimation_mobilenet_v3_small.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"TensorFlow Lite 模型已保存到: {tflite_model_path}")
print(f"模型大小: {os.path.getsize(tflite_model_path) / 1024:.2f} KB")
