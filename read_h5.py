import h5py
import numpy as np

# 将 'your_file.h5' 替换为你实际的文件名
file_path = 'C:/Users/29172/Downloads/ITOP_side_test_depth_map.h5'

try:
    with h5py.File(file_path, 'r') as f:
        print(f"成功打开文件: {file_path}")
        print("--------------------")

        # 列出文件中的所有数据集（键名）
        print("文件包含的数据集 (Keys):")
        for key in f.keys():
            print(f"- {key}")

        # 如果文件中有名为 'data' 的数据集，就读取并打印其信息
        if 'data' in f:
            data = f.get('data')
            print("\n--------------------")
            print(f"数据集 'data' 的形状: {data.shape}")
            print(f"数据集 'data' 的数据类型: {data.dtype}")
            print("前5个样本的第一个切片:")
            # 使用切片来避免打印过多数据
            print(np.asarray(data[:5, 0, 0]))
        else:
            print("\n--------------------")
            print("警告: 文件中没有找到名为 'data' 的数据集。")

        # 同样，你可以用类似的方式读取其他数据集，比如 'real_world_coordinates'
        # 例如：
        # if 'real_world_coordinates' in f:
        #     labels = f.get('real_world_coordinates')
        #     print("\n--------------------")
        #     print(f"数据集 'real_world_coordinates' 的形状: {labels.shape}")
        #     print("前5个样本的第一个关节点坐标:")
        #     print(np.asarray(labels[:5, 0]))

except FileNotFoundError:
    print(f"错误: 文件 {file_path} 未找到，请检查路径。")
except Exception as e:
    print(f"发生错误: {e}")