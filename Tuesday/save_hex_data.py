import serial.tools.list_ports
import time
import os
from datetime import datetime
from struct import unpack

class ToFDataCollector:
    def __init__(self):
        self.frame_height = 50  # 修改为100x100
        self.frame_width = 50
        self.Frame_head = b"\x00\xFF"  # 帧头
        self.Frame_tail = b"\xDD"  # 帧尾
        self.serialInst = serial.Serial()
        self.total_frames = 15  # 需要采集的总帧数（3秒 * 10帧/秒）
        
    def uart_connect(self):
        """首次连接串口，低速初始化，仿ToF_detection.py"""
        self.serialInst.baudrate = 115200
        self.serialInst.port = "COM14"  # 可根据实际修改
        self.serialInst.bytesize = serial.EIGHTBITS
        self.serialInst.parity = serial.PARITY_NONE
        self.serialInst.stopbits = serial.STOPBITS_ONE
        self.serialInst.open()
        # 清除连接后可能存在的缓存数据
        self.serialInst.reset_input_buffer()
        self.serialInst.reset_output_buffer()

    def uart_change_connect(self):
        """切换高速采集串口参数，仿ToF_detection.py"""
        self.serialInst.reset_input_buffer()  # 切换前清除旧数据
        self.serialInst.close()
        self.serialInst.baudrate = 1000000
        self.serialInst.port = "COM14"  # 可根据实际修改
        self.serialInst.bytesize = serial.EIGHTBITS
        self.serialInst.parity = serial.PARITY_NONE
        self.serialInst.stopbits = serial.STOPBITS_ONE
        self.serialInst.open()
        self.serialInst.reset_input_buffer()  # 新连接后清除缓冲区
        self.serialInst.reset_output_buffer()
        
    def init_sensor(self):
        """初始化ToF传感器，发送AT指令后切换高速串口"""
        # 发送指令前清除缓存
        self.serialInst.reset_input_buffer()
        self.serialInst.reset_output_buffer()
        
        commands = [
            b"AT+BINN=1\r",
            b"AT+UNIT=2\r",
            b"AT+DISP=3\r",
            b"AT+FPS=10\r",
            b"AT+BAUD=6\r",
            b"AT+SAVE\r"
        ]
        for cmd in commands:
            self.serialInst.write(cmd)
            time.sleep(1)
            # 每条指令后清除接收缓存，确保不影响下一条指令
            self.serialInst.reset_input_buffer()
        
        # 切换高速采集
        self.uart_change_connect()
            
    def save_frames(self, save_dir="./raw_data"):
        """找到第一次帧头后，直接采集550000字节原始数据并保存为txt，不再分帧。"""
        # 开始采集前清除缓存
        self.serialInst.reset_input_buffer()
        self.serialInst.reset_output_buffer()
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_dir}/raw_data_{timestamp}.txt"
        print(f"开始采集数据，将保存到文件: {filename}")
        raw_data = b''
        found_head = False
        total_bytes_needed = 550000
        while True:
            if self.serialInst.in_waiting:
                packet = self.serialInst.read(self.serialInst.in_waiting)
                raw_data += packet
                if not found_head:
                    idx = raw_data.find(self.Frame_head)
                    if idx != -1:
                        found_head = True
                        raw_data = raw_data[idx:]
                    else:
                        # 没找到帧头，丢弃数据
                        raw_data = b''
                if found_head and len(raw_data) >= total_bytes_needed:
                    # 已采集到足够数据
                    break
            time.sleep(0.01)
        # 只保存550000字节
        save_bytes = raw_data[:total_bytes_needed]
        with open(filename, 'w') as f:
            f.write(" ".join([f"{b:02X}" for b in save_bytes]))
        print(f"数据采集完成，已保存到: {filename}")
        return filename

def main():
    collector = ToFDataCollector()
    collector.uart_connect()
    collector.init_sensor()
    collector.save_frames()

if __name__ == "__main__":
    main()