import subprocess
import time
import requests
import os
import tqdm

class ServerManager:
	'''
	Manages the lifecycle of a llama server process.
	'''
	def __init__(self, model_path, parallel, ctx_size, proj_path=None,
				 n_gpu_layers=99):
		self.model_path = model_path
		self.proj_path = proj_path
		self.parallel = parallel
		self.ctx_size = ctx_size
		self.n_gpu_layers = n_gpu_layers
		self.server_process = None
	
	def get_llama_command(self):
		'''Construct the command to start the llama server. (For windows)'''
		cmd = [
			'llama-server',
			'-m', self.model_path,
			'--n-gpu-layers', str(self.n_gpu_layers),
			'--parallel', str(self.parallel),
			'--cont-batching',
			'--ctx-size', str(self.ctx_size),
			'--host', '0.0.0.0',
			'--port', '8080',
		]

		if self.proj_path != None:
			cmd += ['--mmproj', self.proj_path]

		return cmd

	def start_server(self):
		'''
		Start the llama server using the specified batch file.
		'''
		print('Starting llama server...')
		cmd = self.get_llama_command()
		print(f'Llama server command: {" ".join(cmd)}')

		self.server_process = subprocess.Popen(
			cmd,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			text=True,
		)

		time.sleep(3)

		# Check if the process started successfully
		if self.server_process.poll() is not None:
			stdout, stderr = self.server_process.communicate()
			print(f'Error starting llama server:\n{stderr}')

		print(f'Llama server started with PID {self.server_process.pid}')
	
	def stop_server(self):
		'''
		Stop the llama server if it is running.
		'''
		if self.server_process:
			print('Stopping llama server...')
			try:
				os.system('taskkill /IM llama-server.exe /F')
				time.sleep(2)
				
				self.server_process.terminate()
				self.server_process.wait(timeout=5)
			except subprocess.TimeoutExpired:
				self.server_process.kill()
				self.server_process.wait()
			print('Llama server stopped.')

	def wait_ready(self, max_retries=60, timeout=1):
		'''
		Wait until the llama server is ready or timeout.
		'''
		print(f'Checking if llama server is ready (timeout in {max_retries * timeout}s)...')
		pbar = tqdm.tqdm(desc='Waiting for server', total=max_retries, unit='attempt')
		for attempt in range(1, max_retries + 1):
			try:
				response = requests.get('http://localhost:8080/models', timeout=timeout)
				if response.status_code == 200:
					pbar.close()
					print(f'Llama server is ready! (attempt: {attempt}/{max_retries})')
					return True
			except:
				pass
		
			time.sleep(timeout)
			pbar.update(1)
			
		pbar.close()
		print(f'Error: Llama server not ready in {max_retries * timeout} seconds. Exiting...')
		return False
	