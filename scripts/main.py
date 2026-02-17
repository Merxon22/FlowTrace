from server_manager import ServerManager
from batch_image_processor import BatchImageProcessor
from activity_categorizer import ActivityCategorizer
from taxonomy_generator import TaxonomyGenerator
from report_generator import ReportGenerator

import argparse
import os
import json
from datetime import datetime, timedelta

if __name__ == '__main__':
    # Load configuration from config.json
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Parse optional date argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, required=False, help='Date for processing in YYYY-MM-DD format. Defaults to one day before.')
    args = parser.parse_args()

    SERVER_WAIT_TIME = 180  # How many times to retry server readiness check

    # Get settings from config
    CAPTURE_INTERVAL = config.get('capture_interval', 60)
    START_OF_DAY = config.get('start_of_day', 6)

    # If date is not specified, process previous day. If specified, process that date.
    if args.date:
        date = args.date
    elif datetime.now().hour >= START_OF_DAY:
        # If current time is past the start_of_day, set yesterday as the default date
        date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        # If current time is before the start_of_day, set the day before yesterday as the default date
        date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
    date_obj = datetime.strptime(date, '%Y-%m-%d')
    image_descriptions_path = os.path.join(config['output_dir'], f'image_descriptions_{date}.jsonl')
    activity_categories_path = os.path.join(config['output_dir'], f'activity_categories_{date}.json')

    start_time = datetime.now()
    print(f'Processing images between {date} {START_OF_DAY:02d}:00 (inclusive) and {(date_obj + timedelta(days=1)).strftime("%Y-%m-%d")} {START_OF_DAY:02d}:00 (exclusive)')

    photo_paths = []
    # Collect screenshots for the specified date, including those after midnight but before start_of_day
    todays_dir = os.path.join(config['screenshot_dir'], date)
    if os.path.exists(todays_dir):
        for filename in os.listdir(todays_dir):
            if filename.lower().endswith(('.jpg', '.jpeg')):
                time_part = filename.split('_')[0].split('-')[1]  # Filename format includes time as YYYYMMDD-HHMMSS_WXYZ.jpg
                hour = int(time_part[0:2])
                if hour >= START_OF_DAY:
                    photo_paths.append(os.path.join(todays_dir, filename))
    next_day_dir = os.path.join(config['screenshot_dir'], (date_obj + timedelta(days=1)).strftime('%Y-%m-%d'))
    if os.path.exists(next_day_dir):
        for filename in os.listdir(next_day_dir):
            if filename.lower().endswith(('.jpg', '.jpeg')):
                # Check if the time is before start_of_day
                time_part = filename.split('_')[0].split('-')[1]  # Filename format includes time as YYYYMMDD-HHMMSS_WXYZ.jpg
                hour = int(time_part[0:2])
                if hour < START_OF_DAY:
                    photo_paths.append(os.path.join(next_day_dir, filename))        

    # Process batch images (skip if the output already exists)
    if os.path.exists(image_descriptions_path):
        print(f'Image descriptions for {date} already exist at {image_descriptions_path}, skipping image processing step.')
    else:
        server = ServerManager(
            config['vlm_model_path'],
            config['vlm_parallel'],
            config['vlm_ctx_size'],
            config['vlm_proj_path']
        )
        server.start_server()
        if not server.wait_ready(max_retries=SERVER_WAIT_TIME, timeout=1):
            # If the server fails to start, we should not proceed with processing and should exit gracefully
            server.stop_server()
            exit(1)
        try:
            processor = BatchImageProcessor(max_workers=config['vlm_parallel'])
            output_log_path = image_descriptions_path
            processor.run_batch(photo_paths, output_log_path)
            print('Image batch processing completed successfully.')
        finally:
            # Ensure the server is stopped even if processing fails
            server.stop_server()
    
    # Process taxonomy analysis (skip if the output already exists)
    if os.path.exists(activity_categories_path):
        print(f'Activity categories for {date} already exist at {activity_categories_path}, skipping taxonomy generation step.')
    else:
        server = ServerManager(
            config['llm_model_path'],
            1,
            config['llm_ctx_size']
        )
        server.start_server()
        if not server.wait_ready(max_retries=SERVER_WAIT_TIME, timeout=1):
            server.stop_server()
            exit(1)
        try:
            num_active_hours = int(len(photo_paths) * (CAPTURE_INTERVAL / 3600.0))
            min_categories = 2      # Heuristically set minimum number of categories to 2 for very low activity days
            # Max categories: min=2 at 0 active hours, max=6 at 6 or more active hours, linear in between
            max_categories = int(round(min(6, max(min_categories, num_active_hours))))
            generator = TaxonomyGenerator(min_categories=min_categories, max_categories=max_categories)
            generator.run(image_descriptions_path, activity_categories_path)
        finally:
            # Ensure the server is stopped even if taxonomy generation fails
            server.stop_server()

    # Process activity categorization (run parallel LLM server)
    server = ServerManager(
        config['llm_model_path'],
        config['llm_category_parallel'],
        config['llm_ctx_size']
    )
    server.start_server()
    if not server.wait_ready(max_retries=SERVER_WAIT_TIME, timeout=1):
        server.stop_server()
        exit(1)

    try:

        # Skip category classification if the output already exists
        can_skip_categorization = False
        if os.path.exists(image_descriptions_path):
            # Load the jsonl file and ensure each entry has a category_id_field
            all_entries_have_category = True
            with open(image_descriptions_path, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line)
                    if 'category_id' not in entry:
                        all_entries_have_category = False
                        break
            if all_entries_have_category:
                print(f'Categorized activities for {date} already exist at {image_descriptions_path}, skipping activity categorization step.')
                can_skip_categorization = True
            else:
                print(f'Found existing image descriptions at {image_descriptions_path} but some entries are missing category_id, re-running categorization step.')
        if not can_skip_categorization:
            categorizer = ActivityCategorizer(
                activity_categories_path,
                image_descriptions_path,
                num_workers=config['llm_category_parallel']
            )
            categorizer.run(image_descriptions_path, batch_size=config['llm_category_batch_size'])

        # Generate report
        report_generator = ReportGenerator(
            date,
            categorized_path=image_descriptions_path,
            category_path=activity_categories_path,
            log_interval=CAPTURE_INTERVAL / 60.0,   # Convert capture interval to minute for plotting
            start_of_day=START_OF_DAY
        )
        figures, overview_json, final_report_entries = report_generator.generate_figures(num_workers=config['llm_category_parallel'])
    finally:
        server.stop_server()    # Stop the parallel LLM server, switch to a single-worker server for longer context size (PDF generation)

    server = ServerManager(
        config['llm_model_path'],
        1,
        config['llm_ctx_size']
    )
    server.start_server()
    if not server.wait_ready(max_retries=SERVER_WAIT_TIME, timeout=1):
        server.stop_server()
        exit(1)

    try:
        clean_summary = ReportGenerator.generate_summary(overview_json, final_report_entries)
        # Save report to desktop
        report_path = os.path.join(os.path.expanduser('~'), 'Desktop', f'activity_report_{date}.pdf')
        ReportGenerator.save_pdf(figures, overview_json, clean_summary, date, report_path, start_time, len(photo_paths))
        print(f'Report generated and saved to {report_path}')
    finally:
        server.stop_server()

    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    active_hours = len(photo_paths) / 60.0
    print(f'Spent {total_duration:.1f} seconds processing {len(photo_paths)} records for {date} ({active_hours:.1f} active hours). Avg: {total_duration / len(photo_paths):.2f}s/image')
