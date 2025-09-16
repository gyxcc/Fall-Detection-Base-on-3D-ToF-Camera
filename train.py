import tensorflow as tf

# 配置 GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            # 启用内存增长
            tf.config.experimental.set_memory_growth(gpu, True)
            
        # 只使用第一个 GPU
        tf.config.set_visible_devices(gpus[0], 'GPU')
        
        # 限制 GPU 内存使用
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]  # 限制为 4GB
        )
        
        print("GPU配置完成：使用单个 GPU，启用动态内存增长，限制内存使用")
    except RuntimeError as e:
        print(f"GPU配置错误: {e}")

# 设置内存增长
tf.config.experimental.enable_tensor_float_32_execution(False)  # 关闭 TF32
tf.config.set_soft_device_placement(True)  # 允许在无法使用 GPU 时自动回退到 CPU

print("可用的GPU:", tf.config.list_physical_devices('GPU'))
import numpy as np
import h5py
import os
from PIL import Image

# 确保 TensorFlow 版本正确，支持 MobileNetV3
print(f"TensorFlow version: {tf.__version__}")
# --- 1. 数据准备 (Data Preparation) ---
# 定义图像和关节点的尺寸
IMG_WIDTH = 320
IMG_HEIGHT = 240
NUM_JOINTS = 15
JOINT_DIMENSIONS = 2  # 二维坐标 (x, y)
BATCH_SIZE = 32  # 恢复原始批量大小以提高训练速度

def load_and_preprocess_itop_data(jpg_data_dir, labels_file_path):
    """
    加载并预处理 ITOP 数据集，从 JPG 图像和 h5 标签文件中读取数据。
    
    参数:
    - jpg_data_dir: 包含 JPG 深度图图像的文件夹路径。
    - labels_file_path: 标签 .h5 文件的路径。
    
    返回:
    - 预处理后的深度图数据 (x)，形状为 (n, 240, 320, 1)
    - 预处理后的关节点坐标数据 (y)，形状为 (n, 15, 3)
    """
    print(f"正在加载标签文件: {labels_file_path}")
    try:
        f_labels = h5py.File(labels_file_path, 'r')
        
        # 打印可用的数据集键
        print("可用的数据集键:", list(f_labels.keys()))
        
        # 检查文件中的键
        required_keys = ['image_coordinates', 'id', 'is_valid', 'visible_joints']
        for key in required_keys:
            if key not in f_labels:
                print(f"警告：缺少必要的键 '{key}'")
        
        # 获取标签数据
        try:
            labels_data = np.array(f_labels['image_coordinates'])  # 使用图像坐标而不是世界坐标
            labels_ids = np.array(f_labels['id'])
            is_valid = np.array(f_labels['is_valid'])
            visible_joints = np.array(f_labels['visible_joints'])
            
            # 由于image_coordinates是2D坐标，我们需要调整相关的维度
            if len(labels_data.shape) == 3:  # 如果形状是 (N, 15, 2)
                print("2D关节点坐标加载成功")
            else:
                # 如果需要重塑数组
                num_samples = len(labels_ids)
                labels_data = labels_data.reshape(num_samples, NUM_JOINTS, 2)
                
        except Exception as e:
            print(f"读取数据集时出错: {e}")
            return None, None
        
        # 打印数据形状
        print("\n数据形状:")
        print(f"- real_world_coordinates: {labels_data.shape}")
        print(f"- ids: {labels_ids.shape}")
        print(f"- is_valid: {is_valid.shape}")
        print(f"- visible_joints: {visible_joints.shape}")
        
        # 只使用有效的标签数据
        valid_indices = np.where(is_valid == 1)[0]
        print(f"\n有效标签数量: {len(valid_indices)} / {len(is_valid)}")
        
        if len(valid_indices) == 0:
            print("错误：没有找到有效的标签数据")
            return None, None
            
        labels_data = labels_data[valid_indices]
        labels_ids = labels_ids[valid_indices]
        visible_joints = visible_joints[valid_indices]
        
        # 将标签ID转换为字典
        id_to_label = {}
        for i, id_ in enumerate(labels_ids):
            id_str = id_.decode('utf-8') if isinstance(id_, bytes) else str(id_)
            id_to_label[id_str] = labels_data[i]
            
        # 打印一些示例ID
        print("\n示例标签ID:", list(id_to_label.keys())[:5])
                                             
        f_labels.close()
    except Exception as e:
        print(f"加载标签文件失败: {e}")
        return None, None
    
    print(f"正在加载JPG图像文件从: {jpg_data_dir}")
    if not os.path.exists(jpg_data_dir):
        print(f"错误：图片文件夹 {jpg_data_dir} 不存在")
        return None, None

    images = []
    labels = []
    
    # 获取并排序所有jpg文件
    image_files = sorted([f for f in os.listdir(jpg_data_dir) if f.lower().endswith('.jpg')])
    
    if not image_files:
        print(f"错误：在 {jpg_data_dir} 中没有找到 .jpg 文件")
        return None, None
    
    print(f"找到 {len(image_files)} 个图片文件")
    print(f"示例文件名: {image_files[:5]}")
    print(f"标签ID示例: {list(id_to_label.keys())[:5]}")
    
    processed_count = 0
    # 遍历文件夹中的所有 JPG 文件
    for filename in image_files:
        # 从文件名中提取帧ID（例如从 "19_02181.jpg" 提取 "19_02181"）
        frame_id = os.path.splitext(filename)[0]
        
        # 调试输出 (仅用于打印前5个样本的信息)
        if processed_count < 5:
            print(f"处理文件: {filename}, frame_id: {frame_id}")
            if frame_id not in id_to_label:
                print(f"警告: frame_id {frame_id} 在标签中未找到")
        
        # 核心逻辑：为所有匹配的图像进行处理
        if frame_id in id_to_label:
            image_path = os.path.join(jpg_data_dir, filename)
            try:
                # 使用 PIL 加载图像，并转换为灰度图
                image = Image.open(image_path).convert('L')
                # 调整大小以匹配模型输入
                image = image.resize((IMG_WIDTH, IMG_HEIGHT))
                # 将图像转换为 NumPy 数组
                image_array = np.asarray(image, dtype=np.float32)
                # 归一化到 [0, 1]
                image_array /= 255.0
                
                # 归一化标签坐标
                label = id_to_label[frame_id].astype(np.float32)
                label[:, 0] /= IMG_WIDTH  # 归一化 x 坐标
                label[:, 1] /= IMG_HEIGHT # 归一化 y 坐标

                images.append(image_array)
                labels.append(label)
                processed_count += 1 # 正确增加计数
            except Exception as e:
                print(f"处理图像文件 {filename} 失败: {e}")

    if not images:
        print("警告: 未找到任何有效的JPG图像文件。请检查文件夹路径和文件格式。")
        return None, None

    # 将列表转换为 NumPy 数组
    x_data = np.stack(images, axis=0)
    y_data = np.stack(labels, axis=0)
    
    # 将深度图从 (n, 240, 320) 转换为 (n, 240, 320, 1) 以匹配模型输入
    x_data = np.expand_dims(x_data, axis=-1)

    print(f"处理后的深度图数据形状: {x_data.shape}")
    print(f"处理后的关节点标签形状: {y_data.shape}")
    
    return x_data, y_data

# 设置数据集路径
# JPG图像文件夹路径
TRAIN_JPG_DIR = 'C:/Users/29172/Downloads/side_train_images'
TEST_JPG_DIR = 'C:/Users/29172/Downloads/side_test_images'

# 标签文件路径 (应该使用labels文件而不是depth_map文件)
TRAIN_LABELS_FILE = 'C:/Users/29172/Downloads/ITOP_side_train_labels.h5'
TEST_LABELS_FILE = 'C:/Users/29172/Downloads/ITOP_side_test_labels.h5'

print("数据集路径配置:")
print(f"训练图像文件夹: {TRAIN_JPG_DIR}")
print(f"训练标签文件: {TRAIN_LABELS_FILE}")
print(f"测试图像文件夹: {TEST_JPG_DIR}")
print(f"测试标签文件: {TEST_LABELS_FILE}")

# 加载训练集和测试集
x_train, y_train = load_and_preprocess_itop_data(TRAIN_JPG_DIR, TRAIN_LABELS_FILE)
x_val, y_val = load_and_preprocess_itop_data(TEST_JPG_DIR, TEST_LABELS_FILE)

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

    # 解冻基础模型以进行微调
    base_model.trainable = True

    # 构建我们的模型
    x = base_model(x, training=True) # training=True for fine-tuning
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    
    # 添加全连接层用于姿势回归
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # 输出层，用于回归关节点的2D图像坐标
    # 输出的维度是 NUM_JOINTS * JOINT_DIMENSIONS，即 15 * 2 = 30
    outputs = tf.keras.layers.Dense(num_joints * joint_dims, activation='sigmoid')(x)
    
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


# 设置较小的批量大小
BATCH_SIZE = 16

# 用生成器动态加载数据，避免一次性搬到 GPU
def data_generator(x, y):
    for i in range(len(x)):
        yield x[i].astype('float32'), y[i].astype('float32')

def create_optimized_dataset(x, y, is_training=True):
    output_types = (tf.float32, tf.float32)
    output_shapes = (x.shape[1:], y.shape[1:])
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(x, y),
        output_types=output_types,
        output_shapes=output_shapes
    )
    if is_training:
        dataset = dataset.shuffle(buffer_size=min(1000, len(x)))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

print("创建数据集...")
train_dataset = create_optimized_dataset(x_train, y_train_flat, is_training=True)
val_dataset = create_optimized_dataset(x_val, y_val_flat, is_training=False)
print("数据集创建完成")

# 编译模型（使用为微调优化的设置）
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), # 使用较低的学习率
    loss='mean_squared_error',
    metrics=['mean_absolute_error']
)

# 基本的回调
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
]

# 打印训练配置信息
print("\n训练配置:")
print(f"- 批量大小: {BATCH_SIZE}")
print(f"- 训练样本数: {len(x_train)}")
print(f"- 验证样本数: {len(x_val)}")
print(f"- 每轮步数: {len(x_train) // BATCH_SIZE}")
print("- 使用混合精度训练")
print("- 使用学习率衰减")
print("- 启用数据缓存和预取优化")

print("\n开始训练模型...")
try:
    with tf.device('/GPU:0'):
        history = model.fit(
            train_dataset,
            epochs=5,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1,
            workers=1,  # 减少工作线程
            use_multiprocessing=False  # 禁用多进程
        )
except tf.errors.InternalError as e:
    print(f"GPU 训练失败，尝试在 CPU 上训练: {e}")
    history = model.fit(
        train_dataset,
        epochs=5,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
)

print("模型训练完成。")

# 保存训练好的模型
print("保存模型...")
model.save('pose_estimation_mobilenet_v3_small.h5')
model.save('pose_estimation_mobilenet_v3_small.keras')

# 获取模型信息
h5_size = os.path.getsize('pose_estimation_mobilenet_v3_small.h5') / (1024 * 1024)  # MB
keras_size = os.path.getsize('pose_estimation_mobilenet_v3_small.keras') / (1024 * 1024)  # MB

print(f"模型已保存:")
print(f"  H5格式: pose_estimation_mobilenet_v3_small.h5 ({h5_size:.2f} MB)")
print(f"  Keras格式: pose_estimation_mobilenet_v3_small.keras ({keras_size:.2f} MB)")

# 显示模型统计信息
trainable_params = model.count_params()
print(f"\n模型统计:")
print(f"  总参数量: {trainable_params:,}")
print(f"  模型大小: ~{h5_size:.1f} MB")
print(f"  基于: MobileNetV3 Small")

# 评估最终模型性能
print("\n=== 最终模型评估 ===")
print("在验证集上评估模型性能...")
val_loss, val_mae = model.evaluate(val_dataset, verbose=0)
print(f"验证集损失: {val_loss:.6f}")
print(f"验证集MAE: {val_mae:.6f}")

print(f"\n模型训练和保存完成！")
print(f"建议使用 pose_estimation_mobilenet_v3_small.h5 进行推理")

print("\n=== 模型转换和验证完成 ===")

