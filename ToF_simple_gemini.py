# -*- coding: utf-8 -*-
# 导入所需的库
import serial.tools.list_ports
import serial
import os
from struct import unpack
import time
import json
import numpy as np

# 初始化串口和端口列表
ports = serial.tools.list_ports.comports()
serialInst = serial.Serial()

# 定义帧头和帧尾
Frame_head = b"\x00\xFF"
Frame_tail = b"\xDD"

# 传感器帧的尺寸，已更改为100x100
frame_height = 100
frame_width = 100

# ----------------- 实用工具函数 -----------------
def get_filepath(relative_filepath):
    """获取文件的绝对路径。"""
    dir = os.path.dirname(__file__)
    filename = os.path.join(dir, relative_filepath)
    return filename

# ----------------- 串口通信函数 -----------------
def uart_connect():
    """直接连接到COM14串口。"""
    portVar = "COM14"
    print(f"已指定端口: {portVar}")

    serialInst.baudrate = 115200
    serialInst.port = portVar
    serial.bytesize = serial.EIGHTBITS
    serial.parity = serial.PARITY_NONE
    serial.stopbits = serial.STOPBITS_ONE
    try:
        serialInst.open()
        return True
    except serial.SerialException as e:
        print(f"打开串口 {portVar} 失败: {e}")
        return False

def uart_change_connect():
    """关闭当前连接并重新以更高的波特率打开。"""
    if serialInst.is_open:
        serialInst.close()

    portVar = "COM14"
    print(f"重新连接端口: {portVar}")
    
    serialInst.baudrate = 1000000
    serialInst.port = portVar
    serial.bytesize = serial.EIGHTBITS
    serial.parity = serial.PARITY_NONE
    serial.stopbits = serial.STOPBITS_ONE
    try:
        serialInst.open()
        return True
    except serial.SerialException as e:
        print(f"重新打开串口 {portVar} 失败: {e}")
        return False

def uart_send(data):
    """通过串口发送数据。"""
    serialInst.write(data)

def TOF_init():
    """发送AT指令初始化传感器。"""
    # 注意: 传感器尺寸更改后，你可能需要根据实际情况修改这些AT指令
    uart_send(b"AT+BINN=1\r")
    uart_send(b"AT+UNIT=0\r")
    uart_send(b"AT+DISP=2\r")
    uart_send(b"AT+FPS=10\r")
    uart_send(b"AT+BAUD=8\r")
    uart_send(b"AT+SAVE\r")
    uart_change_connect()

# ----------------- 主程序 -----------------
def main():
    """主函数，负责连接传感器并持续读取数据。"""
    print("正在连接ToF传感器...")
    if not uart_connect():
        return
    TOF_init()

    print("传感器已初始化。正在读取数据...")
    
    print("3秒后开始收集数据...")
    time.sleep(3)
    
    # 创建一个带时间戳的文件名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file_path = f"C:/Users/29172/Downloads/FYP/maxisense-main/Tuesday/raw_data/raw_data_{timestamp}.json"
    
    raw_data = b''
    frame_count = 0
    all_frames_data = [] # 用于存储所有读取到的帧数据
    expected_data_length = frame_height * frame_width

    start_time = time.time() # 记录开始时间
    collection_duration = 15 # 收集持续时间（秒）

    try:
        while True:
            # 检查是否超过15秒
            if time.time() - start_time > collection_duration:
                print(f"\n已达到 {collection_duration} 秒收集时间，自动停止。")
                break

            if serialInst.inWaiting() > 0:
                raw_data += serialInst.read(serialInst.inWaiting())
            
            # 寻找帧头
            head_idx = raw_data.find(Frame_head)
            if head_idx < 0:
                # 如果缓冲区没有帧头，为了防止内存无限增长，可以清空或保留部分数据
                if len(raw_data) > 4096: # 设置一个阈值
                    raw_data = b''
                continue
            
            # 将缓冲区截断到第一个帧头
            raw_data = raw_data[head_idx:]

            # 检查数据长度是否足够解析出数据长度字段 (至少4个字节: 2字节帧头 + 2字节长度)
            if len(raw_data) < 4:
                continue

            # 解析数据帧的总长度
            dataLen = unpack("H", raw_data[2:4])[0]
            # 帧头(2) + 长度(2) + 数据(dataLen) + 校验和(1) + 帧尾(1)
            frameLen = 2 + 2 + dataLen + 2

            # 检查接收到的数据是否构成一个完整的帧
            if len(raw_data) < frameLen:
                continue

            # 提取完整的帧
            frame = raw_data[:frameLen]
            # 从缓冲区移除已处理的帧
            raw_data = raw_data[frameLen:]

            # --- 帧校验 ---
            frameTail_byte = frame[-1:]
            checksum = frame[-2]
            
            # 检查帧尾和校验和
            calculated_checksum = sum(frame[:-2]) % 256
            if frameTail_byte != Frame_tail or checksum != calculated_checksum:
                print(f"校验和或帧尾错误，丢弃该帧。帧尾: {frameTail_byte.hex()}, 校验和: {hex(checksum)}, 计算校验和: {hex(calculated_checksum)}")
                continue
            
            # --- 数据解析 ---
            frame_data_bytes = frame[20:-2]
            current_data_length = len(frame_data_bytes)

            if current_data_length > 0:
                frame_data_values = [unpack("B", bytes([val]))[0] for val in frame_data_bytes]
                all_frames_data.append(frame_data_values)
                
                frame_count += 1
                print(f"--- 成功读取第 {frame_count} 帧数据，长度: {current_data_length} ---")

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("程序已终止。正在保存数据...")
    finally:
        if serialInst.is_open:
            serialInst.close()
        
        # 将所有帧数据保存到文件
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(all_frames_data, f, indent=4)
            print(f"所有 {len(all_frames_data)} 帧数据已成功保存到: {output_file_path}")
        except Exception as e:
            print(f"保存数据失败: {e}")
            
        print("串口已关闭。")

if __name__ == "__main__":
    main()
