import subprocess
import torch
import pickle
import sys,os
import time
import socket
from typing import Optional

class det_model:
    def __init__(self, executable_path: str = './dist/det_model/det_model', port: int = 12345, max_retries: int = 5):
        self.executable_path = executable_path
        self.process = None
        self.stride = None
        self._is_running = False
        self.port = port
        self.sock = None
        self.max_retries = max_retries

    def _connect_with_retry(self):
        """尝试连接服务器，带重试机制"""
        retries = 0
        while retries < self.max_retries:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect(('localhost', self.port))
                return True
            except ConnectionRefusedError:
                print(f"Connection attempt {retries + 1} failed, retrying...", file=sys.stderr)
                retries += 1
                time.sleep(2)  # 等待2秒后重试
                continue
        return False

    def start(self):
        try:
            executable_path = os.path.abspath(self.executable_path)
            print(f"Starting process: {executable_path}")
            
          
            self.process = subprocess.Popen(
                ['dist/det_model/det_model', str(self.port)]            
            )
            # 检查进程是否立即退出
            time.sleep(1)
            exit_code = self.process.poll()
            if exit_code is not None:
                stdout, stderr = self.process.communicate()
                print(f"Process exited immediately with code: {exit_code}")
                print(f"Stdout: {stdout}")
                print(f"Stderr: {stderr}")
                raise RuntimeError(f"Process exited with code {exit_code}")

            print("Process started successfully")

            # 等待服务器启动并尝试连接
            if not self._connect_with_retry():
                raise RuntimeError("Failed to connect to server after multiple attempts")
            
            print(f"Process started with PID: {self.process.pid}")
            self._is_running = True
            
        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"Failed to start process: {str(e)}")

    def _send_data(self, data):
        """发送数据到服务器"""
        serialized_data = pickle.dumps(data)
        length = len(serialized_data)
        self.sock.sendall(length.to_bytes(4, byteorder='big'))
        self.sock.sendall(serialized_data)

    def _recv_data(self):
        """从服务器接收数据"""
        length_bytes = self.sock.recv(4)
        if not length_bytes:
            raise RuntimeError("Connection closed by server")
        length = int.from_bytes(length_bytes, byteorder='big')
        
        data = b''
        while len(data) < length:
            chunk = self.sock.recv(length - len(data))
            if not chunk:
                raise RuntimeError("Connection closed by server")
            data += chunk
            
        result = pickle.loads(data)
        if 'error' in result:
            raise RuntimeError(f"Server error: {result['error']}")
        return result

    def __call__(self, x, mode: int = 2):
        if not self._is_running:
            self.start()

        try:
            if mode == 1:
                print(f"mode 1 begin")
                print(f"x: {x}, mode: {mode}")
                
                # 发送数据
                self._send_data({'x': x, 'mode': mode})
                
                # 接收响应
                output = self._recv_data()
                self.stride = output['stride']
                return self.stride
                
            elif mode == 2:
                if not isinstance(x, torch.Tensor):
                    raise TypeError(f"For mode 2, input should be torch.Tensor, got {type(x)}")
                
                # 发送数据
                self._send_data({'x': x, 'mode': mode})
                
                # 接收响应
                output = self._recv_data()
                return output['tensor']
            else:
                raise ValueError(f"Invalid mode: {mode}")

        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"Error during operation: {str(e)}")

    def cleanup(self):
        """清理资源"""
        if self.sock is not None:
            try:
                self.sock.close()
            except:
                pass
            self.sock = None
            
        if self.process is not None:
            try:
                self.process.terminate()
                self.process.wait(timeout=1.0)
            except:
                self.process.kill()
            finally:
                self.process = None
                self._is_running = False

    def __del__(self):
        self.cleanup()

if __name__ == '__main__':
    # 测试代码
    try:
        model = det_model('dist/det_model/det_model')
        model.start()
        print("Process started successfully")
        
        # 测试初始化
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        stride = model(device, mode=1)
        print(f"Model initialized with stride: {stride}")
        
        # 测试推理
        img = torch.randn(1, 3, 640, 640)
        output = model(img, mode=2)
        print(f"Inference successful, output shape: {output.shape}")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        sys.exit(1)