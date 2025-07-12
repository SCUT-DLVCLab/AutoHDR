import subprocess
import torch
import pickle
import atexit
import sys
import time
import os
from typing import Optional

class det_model:
    def __init__(self, executable_path: str = '/data1/zyy/AutoHDR/dist/det_model'):
        self.executable_path = executable_path
        self.process = None
        self.stride = None
        self._is_running = False
        
    def _check_process_status(self) -> bool:
        """检查进程状态"""
        if self.process is None:
            return False
            
        # 检查进程是否终止
        if self.process.poll() is not None:
            return False
            
        # 检查管道是否正常
        if (self.process.stdin is None or 
            self.process.stdout is None or 
            self.process.stderr is None):
            return False
            
        # 检查管道是否关闭
        try:
            if self.process.stdin.closed or self.process.stdout.closed:
                return False
        except Exception:
            return False
            
        return True
        
    def _ensure_process_running(self):
        """确保进程正在运行"""
        if not self._check_process_status():
            raise RuntimeError("Process is not running")
    
    def start(self, device: str = 'cuda'):
        """启动进程并初始化模型，返回stride值"""
        # 如果进程已在运行，先清理
        if self._check_process_status():
            self.cleanup()
            
        print('Starting process...', file=sys.stderr, flush=True)
        
        if not os.path.exists(self.executable_path):
            raise RuntimeError(f"Executable not found: {self.executable_path}")
        
        if not os.access(self.executable_path, os.X_OK):
            os.chmod(self.executable_path, 0o755)
            
        try:
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['LANG'] = 'C.UTF-8'
            env['LC_ALL'] = 'C.UTF-8'
            
            self.process = subprocess.Popen(
                [self.executable_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                bufsize=0,
                close_fds=True
            )
            
            # 检查进程是否成功启动
            time.sleep(0.5)
            if not self._check_process_status():
                _, stderr = self.process.communicate()
                error_msg = stderr.decode('utf-8', errors='replace')
                raise RuntimeError(f"Process failed to start: {error_msg}")
            
            try:
                init_data = {
                    'device': device
                }
                data = pickle.dumps(init_data)
                self._write_data(data)
                
                output_data = self._read_data()
                if output_data is None:
                    raise RuntimeError("No data received from subprocess")
                    
                output = pickle.loads(output_data)
                self.stride = output['stride']
                
                # 标记进程为运行状态
                self._is_running = True
                
                print(f'Process started with PID {self.process.pid}', file=sys.stderr, flush=True)
                print(f'Model initialized with stride {self.stride}', file=sys.stderr, flush=True)
                
                return self.stride
                
            except Exception as e:
                self.cleanup()
                raise e
                
        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"Failed to initialize model: {str(e)}")

    def _write_data(self, data: bytes):
        """写入数据"""
        self._ensure_process_running()
        try:
            length = len(data)
            self.process.stdin.write(length.to_bytes(4, byteorder='big'))
            self.process.stdin.write(data)
            self.process.stdin.flush()
        except IOError as e:
            self.cleanup()
            raise RuntimeError(f"Failed to write data: {str(e)}")

    def _read_data(self) -> Optional[bytes]:
        """读取数据"""
        self._ensure_process_running()
        try:
            length_bytes = self.process.stdout.read(4)
            if not length_bytes:
                self.cleanup()
                return None
            length = int.from_bytes(length_bytes, byteorder='big')
            data = self.process.stdout.read(length)
            if not data:
                self.cleanup()
                return None
            return data
        except IOError as e:
            self.cleanup()
            raise RuntimeError(f"Failed to read data: {str(e)}")
    
    def __call__(self, x: torch.Tensor, device: str = 'cuda',) -> torch.Tensor:
        """执行推理"""
        self._ensure_process_running()
        try:
            input_data = {
                'tensor': x
            }
            data = pickle.dumps(input_data)
            
            self._write_data(data)
            
            output_data = self._read_data()
            if output_data is None:
                raise RuntimeError("No output data received")
                
            output = pickle.loads(output_data)
            return output['tensor']
            
        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"Error during inference: {str(e)}")
    
    def cleanup(self):
        """清理进程"""
        if self.process is not None:
            try:
                if self.process.poll() is None:
                    self.process.stdin.close()
                    self.process.stdout.close()
                    self.process.stderr.close()
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait()
            except:
                pass
            finally:
                self.process = None
                self._is_running = False
    
    def __del__(self):
        """析构函数"""
        self.cleanup()