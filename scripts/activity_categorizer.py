from openai import OpenAI
import json
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor

class ActivityCategorizer:
    def __init__(self, category_file_path, log_file_path, num_workers=4):
        self.client = OpenAI(base_url='http://localhost:8080/v1', api_key='sk-no-key-required')
        self.num_workers = num_workers
        # Load categories and log entries
        self.categories = self.get_categories(category_file_path)
        self.log_entries = self.get_log_entries(log_file_path)

        # Get 'Other' category ID
        category_names = [cat['name'] for cat in self.categories]
        category_names.reverse()    # Start checking from the end since 'Other' is more likely to be at the end if it exists
        idx = len(self.categories)
        for cat_name in category_names:
            if 'other' in cat_name.lower():
                self.other_category_id = idx
                break
            idx -= 1
        else:
            # If 'Other' category is not found, assign the last category's ID (this should not happen if categories are properly formatted)
            print('Warning: "Other" category not found in categories list, defaulting to last category ID for "Other". Please ensure the category file is properly formatted.')
            self.other_category_id = len(self.categories)

        print(f'Initialized ActivityCategorizer with {len(self.categories)} categories (Other category ID: {self.other_category_id}) and {len(self.log_entries)} log entries.')
        self.system_prompt = '''You are a classification engine. User will provide a list of categories with IDs and a single activity log entry.
**Your Task:**
1. Compare the log entry against the Category Descriptions.
2. Select the single best-matching Category ID.
3. If the entry is totally irrelevant to all categories, use the ID of "Other" category.

**Output Requirement:**
Respond ONLY with the JSON object containing the ID. Do not include reasoning or conversational text.
'''

    def get_categories(self, category_file_path):
        '''Load categories from the given JSON file and ensure 'Other' category exists. Returns the list of categories.'''
        # Read the category file (JSON)
        with open(category_file_path, 'r', encoding='utf-8') as f:
            categories = json.load(f)
        
        return categories

    def get_log_entries(self, log_file_path):
        '''Load log entries from the given JSONL file. Returns a list of log entries.'''
        entries = []
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line.strip())
                entries.append(entry)
        return entries


    def categorize_batch(self, entries):
        '''Categorize a batch of log entries. Returns a list of category IDs corresponding to each entry.'''
        columns_to_keep = ['application', 'action', 'topic', 'detail']      # Keep only relevant fields for categorization
        filtered_entries = [
            {col: entry.get(col) for col in columns_to_keep}
            for entry in entries
        ]

        user_prompt = f'''
<categories>
{json.dumps(self.categories, ensure_ascii=False, indent=2)}
</categories>

<entries>
{json.dumps(filtered_entries, ensure_ascii=False, indent=2)}
</entries>

<instructions>
For each entry in <entries>, select the single best-matching category ID from <categories> based on the descriptions. If an entry is irrelevant to all categories, assign it the ID of the "Other" category. Respond ONLY with a JSON object containing an array of category IDs in the order of the entries.
'''
        
        try:
            response = self.client.chat.completions.create(
                model='qwen3-14b',
                messages=[
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                response_format={
                    'type': 'json_schema',
                    'json_schema': {
                        'name': 'batch_category_selection',
                        'strict': True,
                        'schema': {
                            'type': 'object',
                            'properties': {
                                'category_ids': {
                                    'type': 'array',
                                    'items': {'type': 'integer'},
                                    'description': f'Exactly {len(entries)} category IDs in order',
                                    'minItems': len(entries),
                                    'maxItems': len(entries)
                                }
                            },
                            'required': ['category_ids'],
                            'additionalProperties': False
                        }
                    }
                },
                max_tokens=200,
            )

            result = json.loads(response.choices[0].message.content)
            # Ensure the indices are within valid range
            valid_category_ids = [cat['id'] for cat in self.categories]
            for i in range(len(result.get('category_ids', []))):
                if result['category_ids'][i] not in valid_category_ids:
                    result['category_ids'][i] = self.other_category_id
            return result.get('category_ids', [self.other_category_id] * len(entries))
        except Exception as e:
            return [self.other_category_id] * len(entries)

    def run(self, output_path, batch_size=4):
        print(f'Running activity categorization with batch size {batch_size}, found {len(self.log_entries)} log entries...\n')
        start_time = time.time()
        categorized_results = self.log_entries.copy()

        # Split entries into batches for concurrent processing
        batches = [
            categorized_results[i:i + batch_size] 
            for i in range(0, len(categorized_results), batch_size)
        ]

        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self.categorize_batch, batch) for batch in batches]

            batch_results = []
            for future in tqdm(futures, desc='Processing Batches', unit='batch'):
                result = future.result()
                batch_results.append(result)

        all_category_ids = [cat_id for batch in batch_results for cat_id in batch]      
        for idx, category_id in enumerate(all_category_ids):
            categorized_results[idx]['category_id'] = category_id

        total_time = time.time() - start_time

        # Save results, overrriting existing file if it exists
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in categorized_results:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        print(f'Categorization completed. Results saved to {output_path}')
        print(f'Total: {total_time:.2f} seconds. Avg: {total_time / len(self.log_entries):.4f}s/entry')

if __name__ == '__main__':
    # Test configuration
    date = '2026-02-03'
    ENTRY_PATH = f'../outputs/image_descriptions_{date}.jsonl'
    CATEGORY_PATH = f'../outputs/activity_categories_{date}.json'

    categorizer = ActivityCategorizer(CATEGORY_PATH, ENTRY_PATH, num_workers=2)
    categorizer.run(f'../outputs/categorized_activities_{date}.jsonl')
