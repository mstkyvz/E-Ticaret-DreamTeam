import subprocess
import os
import signal

class GgufManager:
    def __init__(self):
        self.command = "./llama.cpp/llama-server -m ./gguf/gemma-2-9b-it-Q4_K_M.gguf -t 5 --port 8001 -ngl 43 -c 8192"
        self.process = None

    def run(self):
        
        self.process = subprocess.Popen(self.command, shell=True)
        print(f"Process started with PID: {self.process.pid+1}")
        return self.process.pid+1

    def kill(self):
        """PID'yi öldür."""
        if self.process is not None:
            print(f"Killing process with PID: {self.process.pid+1}")
            os.kill(self.process.pid+1, signal.SIGTERM)
            self.process = None
        else:
            print("No process is running.")