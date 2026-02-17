from openai import OpenAI
import pandas as pd
import json
from gensim.parsing.preprocessing import preprocess_string, remove_stopwords, strip_punctuation, strip_tags, remove_stopwords, strip_short
import jieba
from collections import Counter

class TaxonomyGenerator:
    '''
    Generates activity taxonomy from screen log descriptions using a LLM server.
    '''

    def __init__(self, min_categories=3, max_categories=6):
        self.MIN_CATEGORIES = min_categories
        self.MAX_CATEGORIES = max_categories
        pass
    
    def extract_tf_keywords(self, text_str, top_n=5):
        '''
        Extracts the top N keywords from a given text string based on term frequency.
        Parameters:
        - text_str: A string containing the text to analyze, everything in one string (might include non-English words).
        - top_n: The number of top keywords to return.
        Returns:
        - A list of the top N keywords based on term frequency.
        '''
        # Perform text preprocessing using gensim's preprocess_string with custom filters (for English)
        custom_filters = [
            lambda x: x.lower(),
            strip_tags,
            strip_punctuation,
            remove_stopwords,
            strip_short
        ]
        tokens = preprocess_string(text_str, filters=custom_filters)

        # Perform Chinese word segmentation using jieba
        chinese_stopwords = set(['的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
                                '都', '一个', '上', '也', '很', '到', '说', '要', '去', '你',
                                '会', '着', '没有', '看', '好', '自己', '这',
                                '我们', '他们', '她们', '它们', '吗', '呢', '吧', '啊'])
        chiense_punctuations = set('，。！？；：“”‘’（）【】《》…—-、·')
        final_tokens = []
        for token in tokens:
            # Use jieba to segment Chinese text into words
            segmented = jieba.lcut(token, cut_all=False, HMM=True)
            # Filter out Chinese stopwords and punctuation
            filtered = [word for word in segmented if (
                word not in chinese_stopwords
                and word not in chiense_punctuations
                and (len(word) > 1 or word.isdigit() or any('\u4e00' <= char <= '\u9fff' for char in word))  # Remove English one-letter words but keep numbers and Chinese characters
                )]
            final_tokens.extend(filtered)

        word_counts = Counter(final_tokens)
        top_keywords = [word for word, count in word_counts.most_common(top_n)]
        
        return top_keywords

    def load_and_aggregate(self, path):
        '''Load JSONL file and aggregate activities by application and topic.'''
        df = pd.read_json(path, lines=True)
        # Sample rows sparsely to reduce context length and reduce noise from infrequent activities. This helps the LLM focus on main activities.
        sample_interval = 2     # 1 means no sampling, 2 means take every other row, etc. Adjust as needed based on the size of the logs and LLM context limits.
        df = df.iloc[::sample_interval]

        # Aggregate by application and action, and combine topics and details into lists.
        # This creates a more concise representation of activities for the LLM to analyze.
        agg_df = df.groupby(['application', 'action']).agg({
            'topic': lambda x: list(x),
            'detail': lambda x: list(x),
            'application': 'size'
        }).rename(columns={'application': 'count'})
        agg_df = agg_df.sort_values('count', ascending=False)
        drop_threshold = 0      # 0 means keep all activities, 1 means drop activities that only occur once, etc
        agg_df = agg_df[agg_df['count'] > drop_threshold]    # Drop activities that do not meet the threshold to reduce noise and focus on main activities

        activity_summary = []
        # Extract keywords for each activity group and format the summary string. This provides a concise representation of each activity group for the LLM.
        for idx, row in agg_df.iterrows():
            app, action = idx
            count = row['count']
            topics = row['topic']
            details = row['detail']

            combined_texts = ' '.join(topics + details)     # A string that combines all topics and details for this activity group
            num_words = 3 + count
            keywords = self.extract_tf_keywords(combined_texts, top_n=num_words)

            # Format output
            keyword_str = f'[{", ".join(keywords)}]' if keywords else ''
            summary = f'({count}x) {app}: {action} {keyword_str}'
            activity_summary.append(summary)

        log_str = '\n'.join(activity_summary)
        print(f'Total unique activities after aggregation: {len(activity_summary)}, num characters: {len(log_str)}')
        print(f'Max frequency of an activity: {agg_df["count"].max()}x')
        return log_str

    def get_category_schema(self, min_cats, max_cats):
        '''Return the JSON schema for category validation.'''
        return {
            'type': 'object',
            'properties': {
                'categories': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'name': {
                                'type': 'string',
                                'description': 'Non-overlapping, specific name for the activity category, combining related topics into ONE',
                                'maxLength': 64
                            },
                            'description': {
                                'type': 'string',
                                'description': 'One short paragraph that describes the keywords and activities that belong to this category. Should be concise but comprehensive, containing **SPECIFIC TERMS** that represent the range of activities in this category. Inside the paragraph, include 8-12 keywords that cover the full range of activities in this category.',
                                'maxLength': 1024
                            },
                            'focus': {
                                'type': 'string',
                                'description': 'Report the focus of the category. This should distill the main theme of the category into a brief statement.',
                                'maxLength': 128
                            }
                        },
                        'required': ['name', 'description', 'focus'],
                        'additionalProperties': False
                    },
                    'minItems': min_cats,
                    'maxItems': max_cats
                }
            },
            'required': ['categories'],
            'additionalProperties': False
        }

    def get_prompts(self, log_text, min_cats, max_cats):
        '''Return system and user prompts.'''
        
        system_prompt = f'''You are a Personal Activity Classifier. User will provide a list of computer activities. Your goal is to identify between {min_cats} and {max_cats} main categories of activities.

Guiding Rules:
1. Generate **exactly between {min_cats} and {max_cats} categories**.
2. **AGGRESSIVELY GROUP by High-Level Activity**: Treat similar projects, related topics, or the same field as ONE category. When in doubt, merge rather than split.
- CORRECT: "Economics Study" (Includes price discrimination, monopoly, microeconomics, quantitative finance).
- INCORRECT: "Economics", "Microeconomics", "Quantitative Finance" (these are all ONE field).
- CORRECT: "AI Lab & Model Development" (Includes model training, inference, HuggingFace, GPU config, Python coding).
- INCORRECT: "AI Model Inference", "AI Lab Development" (these are
3. **DEFAULT TO MERGING CATEGORIES**: If two categories could possibly overlap or seem related, they MUST be merged into one broader category.
- CORRECT: "AI Lab & Model Development" (Includes model training, inference, HuggingFace, GPU config, Python coding).
- INCORRECT: "AI Model Inference", "AI Lab Development" (these are sub-tasks of ONE project).
4. **NO OVERLAPS**: Each category must be mutually exclusive. No overlapping categories allowed.
5. **Identify ALL Activities**: Do not feel obligated to contain only productivity-related categories. Explicitly identify games, social media, or entertainment if they exist and takes up a considerable portion of time.
6. Each category 'name' must be specific and descriptive (e.g., "Economics & Microeconomics Study" instead of "Economics" or "Microeconomics").
7. The 'description' must be a brief paragraph that describes the key words and activities that belong to this category. Inside the paragraph, there should be 8-12 keywords that cover the full range of activities in this category.
8. The 'focus' must be a short, ONE-SENTENCE report that is short and exact. Conclude the focus of the category by referencing specific application name, work title, platform name, or topic. E.g., "Studying economics on Khan Academy and YouTube, focusing on market failure and government intervention" or "Leisure activities including watching gameplay videos on YouTube, watching Harry Potter, and Apex gaming".
9. If the activity is vague (like "New Tab"), put it in a "Digital Housekeeping" category.
10. Include an "Other" category ONLY if absolutely necessary for logs that are difficult to categorize. If this is used, ensure its name is strictly set to "Other".
'''

        user_prompt = f'''I am providing a unique list of my screen activities. Each entry has a 'topic', 'detail', and a list of keywords associated with it. Analyze the entries to identify the multiple subjects that engaged with on my computer.

IMPORTANT:
- Remember to aggressively group similar activities into broader categories. When in doubt, merge rather than split.
- The categories you generate must be mutually exclusive with no overlaps.

LOGS:
{log_text}
    '''
        return system_prompt, user_prompt

    def generate_categories(self, client, system_prompt, user_prompt, schema):
        '''Call LLM API to generate activity categories.'''
        response = client.chat.completions.create(
            model='qwen3-14b',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            response_format={
                'type': 'json_schema',
                'json_schema': {
                    'name': 'activity_taxonomy',
                    'strict': True,
                    'schema': schema
                }
            },
            timeout=None,
            max_tokens=4096,
        )
        taxonomy = json.loads(response.choices[0].message.content)

        # If 'Other' category does not exist, create it
        category_names = [cat['name'] for cat in taxonomy['categories']]
        has_other = False
        category_names.reverse()    # Start checking from the end since 'Other' is more likely to be at the end if it exists
        for cat_name in category_names:
            if 'other' in cat_name.lower():
                has_other = True
                break
        if not has_other:
            taxonomy['categories'].append({
                'name': 'Other',
                'description': 'Miscellaneous activities that do not fit into other categories',
                'focus': 'Miscellaneous activities that do not fit into other categories'
            })

        # Generate ids for each category
        for idx, cat in enumerate(taxonomy['categories']):
            cat['id'] = idx + 1  # Start IDs from 1

        return taxonomy['categories']

    def save_categories(self, categories, output_file):
        '''Save categories to JSON file.'''
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(categories, f, ensure_ascii=False, indent=2)

    def run(self, input_file, output_file):
        '''Main execution function.'''
        self.client = OpenAI(base_url='http://localhost:8080/v1', api_key='sk-no-key-required')

        # Load and aggregate
        print(f'Loading data from {input_file}...')
        log_text = self.load_and_aggregate(input_file)

        # Get schema and prompts
        schema = self.get_category_schema(self.MIN_CATEGORIES, self.MAX_CATEGORIES)
        system_prompt, user_prompt = self.get_prompts(log_text, self.MIN_CATEGORIES, self.MAX_CATEGORIES)

        # Generate categories
        print('Generating activity taxonomy...')
        categories = self.generate_categories(self.client, system_prompt, user_prompt, schema)

        # Save and display
        self.save_categories(categories, output_file)
        print(f'\nCategories saved to {output_file}')
        print(f'Total categories: {len(categories)}')
        for cat in categories:
            print(f"  - {cat['name']}")

if __name__ == '__main__':
    # Testing configurations
    date = '2026-02-13'
    generator = TaxonomyGenerator(min_categories=2, max_categories=6)

    input_file = f'../outputs/image_descriptions_{date}.jsonl'
    output_file = f'../outputs/activity_categories_{date}.json'
    
    generator.run(input_file, output_file)
