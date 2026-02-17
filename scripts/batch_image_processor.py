import os
import base64
import time
import tqdm
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import json

class BatchImageProcessor:
	'''
	Processes images in batch and generates descriptions using a VLM server.	
	'''

	def __init__(self, max_workers=4):
		self.client = OpenAI(base_url='http://localhost:8080/v1', api_key='sk-no-key-required')
		self.max_workers = max_workers

		# Schema for JSON mode response
		self.image_schema = {
			'type': 'object',
			'properties': {
				'application': {
					'type': 'string',
					'maxLength': 64,
					'description': 'Name of the appliction (software, website, game, etc.) visible in the screenshot (e.g., VS Code, Telegram, Chrome, YouTube, Amazon, Slack, StardewValley). Be specific if possible.'
				},
				'action': {
					'type': 'string',
					'maxLength': 128,
					'description': 'Specific user action being performed in the screenshot (e.g., "Editing Python script", "Viewing email", "Writing document")'
				},
				'topic': {
					'type': 'string',
					'maxLength': 128,
					'description': 'Main topic or subject matter visible in the screenshot (e.g., "Computer vision task related to biomass", "Economics material regarding monopoly and market efficiency", "Project management dashboard for Q2 goals")'
				},
				'detail': {
					'type': 'string',
					'maxLength': 256,
					'description': 'Detailed information or context about the screenshot content (e.g., names of files, key information shown, specific section of work, visible characters in a game, etc.)'
				}
			},
			'required': ['application', 'action', 'topic', 'detail'],
			'additionalProperties': False
		}

	def read_image(self, image_path):
		'''
		Reads an image from the given path and encodes it in base64.
		'''
		with open(image_path, 'rb') as image_file:
			return base64.b64encode(image_file.read()).decode('utf-8')

	def process_image(self, image_path):
		'''
		Process a single image and return the extracted information.
		'''
		# Extrac timestamp, window name, and image
		file_name = os.path.basename(image_path)
		parts = file_name.rsplit('_', 1)  # Split from the right, max 1 split
		timestamp = parts[0]
		window_name = parts[1].rsplit('.', 1)[0]  # Remove only the file extension
		base64_image = self.read_image(image_path)
		
		# Setup prompts
		system_prompt = (
			'You are a screen activity analyzer. '
			'Extract application name, user action, topic of work, and details. '
			'Avoid giving general and vague descriptions. '
			'Within limited words, make sure to include the most relevant information about the specific topics. '
			'Keep responses concise and on single lines.'
		)
		user_prompt = (
			f'Context: Window title is "{window_name}". '
			'Analyze this screenshot and identify the application, action, topic, and detail as single-line strings without newlines.'
		)
		
		# Call VLM server
		try:
			response = self.client.chat.completions.create(
				model='qwen3-vl',
				messages=[
					{
						'role': 'system',
						'content': system_prompt
					},
					{
						'role': 'user',
						'content': [
							{
								'type': 'text', 
								'text': user_prompt
							},
							{
								'type': 'image_url',
								'image_url': {'url': f'data:image/jpeg;base64,{base64_image}'}
							}
						]
					}
				],
				max_tokens=192,
				response_format={
					'type': 'json_schema',
					'json_schema': {
						'name': 'screen_activity',
						'strict': True,
						'schema': self.image_schema
					}
				},
			)
			# Parse and return result
			result = json.loads(response.choices[0].message.content)
			result['timestamp'] = timestamp
			result['window_name'] = window_name
			
			return result

		except Exception as e:
			return {'timestamp': timestamp, 'window_name': window_name, 'application': '', 'action': '', 'topic': '', 'detail': f'Error - {str(e)}'}

	def run_batch(self, photo_paths, output_file):
		'''
		Process all images in the given directory and save results to output file.
		'''
		# Get all images in the directory
		image_extensions = ('.jpg', '.jpeg', '.png')

		print(f'Found {len(photo_paths)} images. Starting batch processing...\n')

		
		batch_start_time = time.time()
		results = []
		
		with ThreadPoolExecutor(max_workers=4) as executor:
			for result in tqdm.tqdm(
					executor.map(self.process_image, photo_paths),
					total=len(photo_paths),
					desc='Processing images',
					unit='image'
				):
				results.append(result)
		
		total_time = time.time() - batch_start_time

		# Save to file
		with open(output_file, 'w', encoding='utf-8') as f:
			for line in results:
				f.write(json.dumps(line, ensure_ascii=False, indent=None) + '\n')
		
		print(f'Done! Results saved to {output_file}')
		print(f'Total time: {total_time:.2f} seconds, avg: {total_time / len(results):.4f}s/image')

if __name__ == '__main__':
	# Test configuration
	processor = BatchImageProcessor(max_workers=4)
	
	date = 'manual'
	FOLDER_DIR = f'../screenshots/{date}'
	OUTPUT_LOG = f'../outputs/image_descriptions_{date}.jsonl'
	processor.run_batch(FOLDER_DIR, OUTPUT_LOG)
