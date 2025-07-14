import subprocess
import torch
import pickle
import atexit
import sys

class reg_model:
    def __init__(self, executable_path: str = './dist/model_exe'):
        self.executable_path = executable_path
        self.process = None
        
    def start(self):
        """启动进程"""
        if self.process is None or self.process.poll() is not None:
            print('Starting process...', file=sys.stderr, flush=True)
            self.process = subprocess.Popen(
                [self.executable_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            print(f'Process started with PID {self.process.pid}', file=sys.stderr, flush=True)
    
    def __call__(self, x: torch.Tensor, device: str = 'cuda') -> torch.Tensor:
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("Process is not running")
            
        try:
            input_data = {
                'tensor': x.to(device),
                'device': device
            }
            
            print('Serializing input data...', file=sys.stderr, flush=True)
            data = pickle.dumps(input_data)
            
            # 先发送数据长度
            print('Sending data length...', file=sys.stderr, flush=True)
            self.process.stdin.write(len(data).to_bytes(4, byteorder='big'))
            
            # 发送数据
            print('Sending data...', file=sys.stderr, flush=True)
            self.process.stdin.write(data)
            self.process.stdin.flush()
            
            # 读取响应长度
            print('Reading response length...', file=sys.stderr, flush=True)
            length_bytes = self.process.stdout.read(4)
            if not length_bytes:
                raise EOFError("Process terminated unexpectedly")
            length = int.from_bytes(length_bytes, byteorder='big')
            
            # 读取响应数据
            print(f'Reading response data ({length} bytes)...', file=sys.stderr, flush=True)
            output_data = self.process.stdout.read(length)
            output = pickle.loads(output_data)
            print('Response received', file=sys.stderr, flush=True)
            
            return output['tensor']
            
        except Exception as e:
            print(f"Error during communication: {e}", file=sys.stderr, flush=True)
            if self.process.poll() is not None:
                print(f"Process terminated with code {self.process.poll()}", 
                      file=sys.stderr, flush=True)
                error = self.process.stderr.read()
                if error:
                    print(f"Process error output: {error.decode()}", 
                          file=sys.stderr, flush=True)
            raise
    
    def cleanup(self):
        """清理资源"""
        if self.process is not None:
            print('Cleaning up process...', file=sys.stderr, flush=True)
            if self.process.poll() is None:  # 如果进程还在运行
                try:
                    self.process.stdin.close()
                    self.process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    print('Process not responding, terminating...', 
                          file=sys.stderr, flush=True)
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        print('Process still not responding, killing...', 
                              file=sys.stderr, flush=True)
                        self.process.kill()
                        self.process.wait()
            
            self.process.stdout.close()
            self.process.stderr.close()
            self.process = None
            print('Process cleaned up', file=sys.stderr, flush=True)

    def __del__(self):
        """析构函数"""
        self.cleanup()