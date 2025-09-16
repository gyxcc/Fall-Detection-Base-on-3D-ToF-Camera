import os
import time
import subprocess
from datetime import datetime

def run_data_collection():
    print("开始数据采集...")
    subprocess.run(["python", "C:/Users/29172/Downloads/maxisense-main/Tuesday/save_hex_data.py"])

def run_image_conversion():
    print("\n开始图像转换...")
    subprocess.run(["python", "C:/Users/29172/Downloads/maxisense-main/Tuesday/trans2pic.py"])

def main():
    # 创建必要的目录
    os.makedirs("./raw_data", exist_ok=True)
    os.makedirs("./depth_images", exist_ok=True)
    
    # 记录开始时间
    start_time = datetime.now()
    print(f"任务开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 运行数据采集
    run_data_collection()

    # 等待1秒确保文件保存完成
    time.sleep(1)
    print("数据采集完成，开始图像转换...")

    # 运行图像转换
    run_image_conversion()
    print("图像转换完成。")
    # 记录结束时间
    end_time = datetime.now()
    print(f"\n任务结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {end_time - start_time}")

if __name__ == "__main__":
    main()