import serial
import time
import os
from datetime import datetime

FRAME_HEAD = b"\x00\xFF"
FRAME_TAIL = b"\xDD"
FRAME_HEIGHT = 100
FRAME_WIDTH = 100
RESERVED_LEN = 16
DATA_LEN = FRAME_HEIGHT * FRAME_WIDTH
FRAME_SIZE_MIN = 4 + RESERVED_LEN + DATA_LEN + 2  # 帧头+保留+数据+校验和帧尾
TOTAL_FRAMES = 30  # 采集帧数
PORT = "COM15"     # 串口号
BAUD = 1000000     # 波特率

def main():
    ser = serial.Serial(PORT, BAUD, timeout=0.1)
    print(f"串口已打开: {PORT} 波特率: {BAUD}")
    raw_data = b''
    frames_collected = 0
    all_data = []
    found_head = False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = "./raw_data"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = f"{save_dir}/raw_data_{timestamp}.txt"
    print(f"开始采集数据，将保存到文件: {filename}")

    while frames_collected < TOTAL_FRAMES:
        if ser.in_waiting:
            packet = ser.read(ser.in_waiting)
            raw_data += packet

            while True:
                idx = raw_data.find(FRAME_HEAD)
                if idx == -1:
                    raw_data = b''
                    break
                # 查找帧尾
                tail_idx = raw_data.find(FRAME_TAIL, idx + 4 + RESERVED_LEN + DATA_LEN)
                if tail_idx == -1:
                    # 没有帧尾，等待更多数据
                    if idx > 0:
                        raw_data = raw_data[idx:]
                    break
                # 截取一帧
                frame = raw_data[idx:tail_idx+1]
                # 校验长度
                if len(frame) < FRAME_SIZE_MIN:
                    # 数据不够一帧，等待更多数据
                    break
                # 校验和
                data_sum = frame[-2]
                calc_sum = sum(frame[:-2]) % 256
                if data_sum != calc_sum:
                    # 校验和不对，丢弃本帧头，继续查找下一个
                    raw_data = raw_data[idx+2:]
                    continue
                # 保存帧
                frame_hex = [f"{b:02X}" for b in frame]
                all_data.extend(frame_hex)
                frames_collected += 1
                print(f"已采集 {frames_collected}/{TOTAL_FRAMES} 帧")
                # 移除已处理帧
                raw_data = raw_data[tail_idx+1:]
                if frames_collected >= TOTAL_FRAMES:
                    break
        else:
            time.sleep(0.01)

    with open(filename, 'w') as f:
        f.write(" ".join(all_data))
    print(f"数据采集完成，已保存到: {filename}")
    ser.close()

if __name__ == "__main__":
    main()