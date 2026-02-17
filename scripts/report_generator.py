import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
from wordcloud import WordCloud, STOPWORDS
from PIL import Image, ImageDraw
from fpdf import FPDF
import json
import textwrap
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
import matplotlib.font_manager as fm

REGULAR_FONT_PATH = '../NotoSansSC-Regular.ttf'
BOLD_FONT_PATH = '../NotoSansSC-Bold.ttf'

class ReportGenerator:
    '''Generates daily activity report figures, summaries, and PDF exports.'''

    def __init__(self, date: str, categorized_path: str, category_path: str,
                 log_interval: int = 1, start_of_day: int = 6):
        '''
        Args:
            date: Date string in 'YYYY-MM-DD' format.
            categorized_path: Path to the categorized activities JSONL file.
            category_path: Path to the activity categories JSON file.
            log_interval: Interval between screenshots in minutes.
            start_of_day: Hour considered the start of the day (for plotting).
        '''
        self.date = date
        self.log_interval = log_interval
        self.start_of_day = start_of_day

        # Load data
        self.activity_df = pd.read_json(categorized_path, lines=True)
        self.category_df = pd.read_json(category_path)

        self.active_minutes = len(self.activity_df) * self.log_interval
        self.active_hours = self.active_minutes / 60.0

        # Clean data
        self.activity_df['category_id'] = self.activity_df['category_id'].astype(int)
        self.category_df['name'] = self.category_df['name'].str.replace('&', 'and')

        # Smooth category_id using rolling mode (window=3)
        self.activity_df['smoothed_category_id'] = (
            self.activity_df['category_id']
            .rolling(window=3, min_periods=1, center=True)
            .apply(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
            .astype(int)
        )

        # Derived data
        self.df_by_category = self.activity_df.groupby('smoothed_category_id').size().sort_values(ascending=False)
        self.palette = sns.color_palette(palette='hls', n_colors=len(self.df_by_category))
        self.color_map = {cid: self.palette[i] for i, cid in enumerate(self.df_by_category.index)}

        plt.rcParams['figure.dpi'] = 200

        fm.fontManager.addfont(REGULAR_FONT_PATH)
        fm.fontManager.addfont(BOLD_FONT_PATH)
        self.font_name = fm.FontProperties(fname=REGULAR_FONT_PATH).get_name()
        plt.rcParams['font.sans-serif'] = [self.font_name]
        plt.rcParams['axes.unicode_minus'] = False


    # ── Helpers ──────────────────────────────────────────────────────────

    def _id_to_category_name(self, cat_id: int) -> str:
        return self.category_df.iloc[cat_id - 1]['name']

    @staticmethod
    def _auto_wrap_str(text: str, max_width: int = 30) -> str:
        return '\n'.join(textwrap.wrap(text, width=max_width))

    @staticmethod
    def _date_to_en(date_str: str) -> str:
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        year, month, day = date_str.split('-')
        month_name = months[int(month) - 1]
        dt = datetime(int(year), int(month), int(day))
        weekday = dt.strftime('%a')
        return f'{month_name} {int(day)}, {year}, {weekday}'

    @staticmethod
    def _minute_to_hours_minutes(minute: int) -> str:
        hours = minute // 60
        minutes = minute % 60
        return f'{hours}h{minutes}m' if hours > 0 else f'{minutes}m'

    # ── Figure: Pie Chart ────────────────────────────────────────────────

    def _create_pie_chart(self):
        fig, ax = plt.subplots(figsize=(16, 8))
        labels = [self._id_to_category_name(cid) for cid in self.df_by_category.index]
        active_minutes = self.active_minutes

        def autopct_format(pct):
            minutes = int(pct / 100 * active_minutes)
            return f'{self._minute_to_hours_minutes(minutes)}\n({pct:.0f}%)'

        pie_colors = [self.color_map[cid] for cid in self.df_by_category.index]
        wedges, texts, autotexts = ax.pie(
            self.df_by_category,
            autopct=autopct_format,
            colors=pie_colors,
            pctdistance=1.1,
            wedgeprops={'edgecolor': 'black', 'width': 0.25},
        )

        kw = dict(arrowprops=dict(arrowstyle="-"), zorder=0, va="center")
        for i, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1) / 2.0 + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = f"angle,angleA=0,angleB={ang}"
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
            ax.annotate(
                labels[i],
                xy=(x * 1.2, y * 1.2),
                xytext=(1.35 * np.sign(x), 1.35 * y),
                horizontalalignment=horizontalalignment,
                **kw,
            )

        ax.text(0, 0,
                f'Total Active: {self.active_hours:.1f}h\n({self._date_to_en(self.date)})',
                ha='center', va='center', fontsize=18, fontweight='bold')

        plt.close(fig)
        return fig

    # ── Figure: Hourly Usage ─────────────────────────────────────────────

    def _create_hourly_chart(self):
        logged_hours = []
        content_switching_hours = []
        last_category_id = None
        last_activity_end_time = None

        for _, row in self.activity_df.iterrows():
            time_object = row['timestamp']
            hour_decimal = time_object.hour + time_object.minute / 60.0 + time_object.second / 3600.0
            if hour_decimal < self.start_of_day:
                hour_decimal += 24.0
            logged_hours.append(hour_decimal)

            if (
                (row['smoothed_category_id'] != last_category_id and last_category_id is not None) or
                (last_activity_end_time is not None and
                 1800 > (time_object - last_activity_end_time).total_seconds() > 300)
            ):
                content_switching_hours.append(hour_decimal)
            last_category_id = row['smoothed_category_id']
            last_activity_end_time = time_object

        bin_interval = 0.25
        bins = np.arange(self.start_of_day, 24 + self.start_of_day + bin_interval, bin_interval)
        counts, _ = np.histogram(logged_hours, bins=bins)
        max_count_per_bin = (60 * bin_interval) // self.log_interval
        counts = counts / max_count_per_bin
        counts = np.convolve(counts, np.ones(2) / 2, mode='same')

        usage_palette = sns.color_palette('Blues', n_colors=len(bins) - 1)
        normalized_counts = counts / counts.max() if counts.max() > 0 else counts
        colors = [usage_palette[int(nc * (len(usage_palette) - 1))] for nc in normalized_counts]
        x_pos = list(np.arange(self.start_of_day, 24 + self.start_of_day, bin_interval) + bin_interval / 2)

        fig, ax = plt.subplots(figsize=(16, 4))
        for hour in range(self.start_of_day, 24 + self.start_of_day):
            if hour % 2 == 0:
                ax.axvspan(hour, hour + 1, alpha=0.1, color='grey')

        ax.bar(x_pos, counts, width=bin_interval, color=colors, edgecolor='black')
        ax.bar([-999], [0], color=usage_palette[int(len(usage_palette) * 0.75)],
               edgecolor='black', label='Screen Usage')

        ax.set_xlabel('Hour of Day')
        ax.set_xlim(self.start_of_day, 24 + self.start_of_day)
        ax.set_xticks(range(self.start_of_day, 25 + self.start_of_day))
        ax.set_xticklabels([
            f'{h % 12 if h % 12 != 0 else 12}{"AM" if h % 24 < 12 else "PM"}'
            for h in range(self.start_of_day, 25 + self.start_of_day)
        ])
        ax.set_ylabel('Screen time percentage\n(How much time you are working on screen)')
        ax.set_yticks(np.arange(0, 1.25, 0.25))
        ax.set_ylim(0, 1)
        ax.set_yticklabels([f'{tick * 100:.0f}%' for tick in ax.get_yticks()])

        if len(content_switching_hours) > 0:
            switching_counts, _ = np.histogram(content_switching_hours, bins=bins)
            max_switching_per_bin = (60 * bin_interval) // self.log_interval
            switching_counts = switching_counts / max_switching_per_bin
            ax.bar(x_pos, switching_counts, width=bin_interval * 0.5,
                   color='red', alpha=0.5, edgecolor='black')
            ax.bar([-999], [0], color='red', alpha=0.7, edgecolor='black', label='Focus Jump')

        ax2 = ax.twinx()
        if len(content_switching_hours) > 0:
            ax2.set_ylabel('Number of Focus Jumps\n(Switching tasks or entering a short break)')
            ax2.set_ylim(0, max_switching_per_bin + 2)
            ax2.tick_params(axis='y')

        ax.legend(loc='upper right')
        ax.set_title(f'Hourly Recap ({self._date_to_en(self.date)})')

        plt.close(fig)
        return fig

    # ── Figure: Word Cloud ───────────────────────────────────────────────

    def _create_word_cloud(self):
        text_data = (' '.join(self.activity_df['topic'].fillna('').astype(str)) + ' ' +
                     ' '.join(self.activity_df['detail'].fillna('').astype(str)))
        
        wc_width, wc_height = 2400, 1200

        img = Image.new('L', (wc_width, wc_height), 255)
        draw = ImageDraw.Draw(img)
        draw.ellipse((20, 20, wc_width - 20, wc_height - 20), fill=0)
        mask = np.array(img)

        # A custom set of stopwords that are not desired in a personal activity word cloud context
        custom_stopwords = {'user', 'User', 'file', 'File', 'window', 'Window'}

        wc = WordCloud(
            width=wc_width, height=wc_height,
            mask=mask,
            background_color='white',
            colormap='flare',
            max_words=200,
            min_word_length=3,
            stopwords=set(STOPWORDS).union(custom_stopwords),
            font_path=REGULAR_FONT_PATH,
        ).generate(text_data)

        fig = plt.figure(figsize=(16, 8))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()

        plt.close(fig)
        return fig

    # ── Chronological Log Helpers ────────────────────────────────────────

    def _get_chronological_log(self):
        broken_bars = {}
        chronological_json = []

        categories = self.activity_df['smoothed_category_id'].unique()
        for cat_id in categories:
            broken_bars[cat_id] = []

        last_activity_category = None
        last_activity_start_time = None
        last_activity_end_time = None

        aggregated_applications = []
        aggregated_actions = []
        aggregated_topics = []
        aggregated_details = []

        def generate_new_entry(start_time, end_time, category, applications, actions, topics, details):
            duration_second = (end_time - start_time).total_seconds()
            start_time_second = start_time.hour * 3600 + start_time.minute * 60 + start_time.second
            if start_time_second < self.start_of_day * 3600:
                start_time_second += 24 * 3600

            broken_bars[category].append((start_time_second / 3600, duration_second / 3600))
            chronological_json.append({
                "category_id": category,
                'start': start_time,
                'end': end_time,
                'duration': self._minute_to_hours_minutes(int(duration_second // 60)),
                'applications': ', '.join(sorted(set(applications))),
                'actions': ', '.join(sorted(set(actions))),
                'topics': ', '.join(sorted(set(topics))),
                'details': ' | '.join(sorted(set(details))),
            })
            aggregated_actions.clear()
            aggregated_applications.clear()
            aggregated_topics.clear()
            aggregated_details.clear()

        for _, row in self.activity_df.iterrows():
            category = row['smoothed_category_id']
            time_object = row['timestamp']

            if last_activity_category is None:
                last_activity_category = category
                last_activity_start_time = time_object
                last_activity_end_time = time_object + timedelta(minutes=self.log_interval)
            else:
                time_since_last = (time_object - last_activity_end_time).total_seconds()
                if category != last_activity_category or time_since_last > (self.log_interval * 60 + 5):
                    generate_new_entry(
                        last_activity_start_time, last_activity_end_time,
                        last_activity_category,
                        aggregated_applications, aggregated_actions,
                        aggregated_topics, aggregated_details,
                    )
                    last_activity_category = category
                    last_activity_start_time = time_object
                    last_activity_end_time = time_object + timedelta(minutes=self.log_interval)
                else:
                    last_activity_end_time = time_object + timedelta(minutes=self.log_interval)

            aggregated_actions.append(row['action'] if pd.notna(row['action']) else '')
            aggregated_applications.append(row['application'] if pd.notna(row['application']) else '')
            aggregated_topics.append(row['topic'] if pd.notna(row['topic']) else '')
            aggregated_details.append(row['detail'] if pd.notna(row['detail']) else '')

        if last_activity_category is not None:
            generate_new_entry(
                last_activity_start_time, last_activity_end_time,
                last_activity_category,
                aggregated_applications, aggregated_actions,
                aggregated_topics, aggregated_details,
            )

        return broken_bars, chronological_json

    def _concatenate_chronological(self, chronological_json, grace_period_minutes: int = 5):
        concatenated = []
        last_category_id = None
        last_start = None
        last_end = None
        agg_apps, agg_acts, agg_tops, agg_dets = [], [], [], []

        def _add(start, end, cat, apps, acts, tops, dets):
            concatenated.append({
                "category_name": self._id_to_category_name(cat),
                "category_id": cat,
                "start": start,
                "end": end,
                "duration": self._minute_to_hours_minutes(int((end - start).total_seconds() // 60)),
                "applications": ', '.join(sorted(set(apps))),
                "actions": ', '.join(sorted(set(acts))),
                "topics": ', '.join(sorted(set(tops))),
                "details": ' | '.join(sorted(set(dets))),
            })
            agg_apps.clear(); agg_acts.clear(); agg_tops.clear(); agg_dets.clear()

        for entry in chronological_json:
            cat_id = entry['category_id']
            start_time = entry['start']
            end_time = entry['end']

            if last_category_id is None:
                last_category_id = cat_id
                last_start = start_time
                last_end = end_time
            elif (cat_id == last_category_id and
                  (start_time - last_end).total_seconds() <= grace_period_minutes * 60):
                last_end = end_time
            else:
                _add(last_start, last_end, last_category_id, agg_apps, agg_acts, agg_tops, agg_dets)
                last_category_id = cat_id
                last_start = start_time
                last_end = end_time

            agg_apps.append(entry['applications'])
            agg_acts.append(entry['actions'])
            agg_tops.append(entry['topics'])
            agg_dets.append(entry['details'])

        if last_category_id is not None:
            _add(last_start, last_end, last_category_id, agg_apps, agg_acts, agg_tops, agg_dets)

        return concatenated

    # ── Figure: Chronological Timeline ───────────────────────────────────

    def _create_chronological_chart(self, broken_bars):
        categories = self.df_by_category.index.tolist()
        categories.reverse()
        y_height = 0.9

        fig, ax = plt.subplots(figsize=(16, 4))
        idx = 0
        for cat_id in categories:
            xranges = broken_bars[cat_id]
            inflated_xranges = []
            for (start, duration) in xranges:
                w = 0.002
                inflated_xranges.append((start - w, duration + 2 * w))

            ax.broken_barh(inflated_xranges, (idx - y_height / 2, y_height),
                           color=self.color_map[cat_id])
            ax.broken_barh(xranges, (len(categories) - y_height / 2, y_height),
                           color=self.color_map[cat_id])
            idx += 1

        for hour in range(self.start_of_day, 24 + self.start_of_day):
            if hour % 2 == 0:
                ax.axvspan(hour, hour + 1, alpha=0.1, color='grey')

        ax.set_xlim(self.start_of_day, 24 + self.start_of_day)
        ax.set_xticks(range(self.start_of_day, 24 + self.start_of_day + 1))
        ax.set_xticklabels([
            f'{h % 12 if h % 12 != 0 else 12}{"AM" if h % 24 < 12 else "PM"}'
            for h in range(self.start_of_day, 24 + self.start_of_day + 1)
        ])
        ax.set_ylim(-y_height / 2 - 0.5, len(categories) + y_height / 2 + 0.5)
        ax.set_yticks(range(len(categories) + 1))
        ax.set_yticklabels(
            [self._auto_wrap_str(self._id_to_category_name(cid), 50) for cid in categories] +
            [f'Aggregated ({self.active_hours:.1f}h)']
        )
        ax.set_title(f'Activity Timeline ({self._date_to_en(self.date)})')
        plt.tight_layout()

        plt.close(fig)
        return fig

    # ── Public: Generate Figures ─────────────────────────────────────────

    def generate_figures(self, num_workers: int = 4):
        """
        Generate all four report figures, the overview JSON, and final report entries.

        Args:
            num_workers: Number of parallel workers for LLM keyword summarization.

        Returns:
            tuple: (fig_pie, fig_hourly, fig_wc, fig_chronological),
                   overview_json, final_report_entries
        """
        print('Starting figure generation...')
        fig_pie = self._create_pie_chart()
        fig_hourly = self._create_hourly_chart()
        fig_wc = self._create_word_cloud()

        broken_bars, chronological_json = self._get_chronological_log()
        fig_chronological = self._create_chronological_chart(broken_bars)

        # Overview JSON
        overview_json = []
        for category, count in self.df_by_category.items():
            overview_json.append({
                'category_name': self._id_to_category_name(category),
                'duration': self._minute_to_hours_minutes(count * self.log_interval),
                'focus': self.category_df[self.category_df['id'] == category]['focus'].iloc[0],
            })

        # Concatenated chronological entries + final report entries
        concatenated = self._concatenate_chronological(chronological_json)

        # Generate LLM keyword summaries for each entry (parallel)
        print('Starting LLM keyword summarization for each entry...')
        client = OpenAI(base_url='http://localhost:8080/v1', api_key='sk-no-key-required')
        system_prompt = 'You are an expert personal assistant that helps summarize computer activity logs into brief summaries.'

        start_t = time.time()
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(self._generate_llm_summary, client, system_prompt, entry): i
                for i, entry in enumerate(concatenated)
            }
            for future in tqdm(as_completed(futures), total=len(futures),
                              desc='Generating summaries', unit='entry'):
                idx = futures[future]
                concatenated[idx]['activity_summary'] = future.result()
        elapsed = time.time() - start_t
        print(f'Spent {elapsed:.2f}s generating summaries for {len(concatenated)} entries. '
              f'Avg: {elapsed / len(concatenated):.4f}s/entry.')

        # Build final report entries
        final_report_entries = []
        for entry in concatenated:
            cat_name = entry['category_name']
            start = entry['start'].strftime('%H:%M')
            end = entry['end'].strftime('%H:%M')
            duration = entry['duration']
            keywords = entry.get('activity_summary', [])
            final_report_entries.append(
                f'[{start}-{end}] ({duration}) **{cat_name}**: Keywords: {", ".join(keywords)}'
            )

        return (fig_pie, fig_hourly, fig_chronological, fig_wc), overview_json, final_report_entries

    # ── LLM helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _generate_llm_summary(client, system_prompt, entry):
        user_prompt = f'''<entry>
"category_name": "{entry['category_name']}",
"applications": "{entry['applications']}",
"actions": "{entry['actions']}",
"topics": "{entry['topics']}",
"details": "{entry['details']}"
</entry>

The above is a log entry of my computer activity during a specific time period, including the category name, applications used, actions performed, topics worked on, and additional details.
Please provide a brief summary of the computer activity during this time period, capturing the main tasks and actions performed.
Your response should be in JSON format with the following field:
- activity_keywords: A list of keywords summarizing the main activities performed during the time period. Provide between 1 to 10 keywords that best capture the essence of the activities. Provide more keywords for more complex activities, and fewer keywords for simpler activities. Provide fewer keywords if the activity is vague or unclear.
'''
        response = client.chat.completions.create(
            model='qwen3-14b',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            response_format={
                'type': 'json_schema',
                'json_schema': {
                    'name': 'activity_summary',
                    'strict': True,
                    'schema': {
                        'type': 'object',
                        'properties': {
                            'activity_keywords': {
                                'type': 'array',
                                'items': {'type': 'string'},
                                'description': 'A list of keywords summarizing the main activities performed.',
                                'minItems': 1,
                                'maxItems': 6,
                            }
                        },
                        'required': ['activity_keywords'],
                        'additionalProperties': False,
                    }
                }
            },
            max_tokens=100,
        )
        result = json.loads(response.choices[0].message.content)
        return result['activity_keywords']

    # ── Public: Generate Summary ─────────────────────────────────────────

    @staticmethod
    def generate_summary(overview_json, final_report_entries) -> str:
        """
        Use LLM to generate a clean text summary from overview and report entries.

        Returns:
            str: Cleaned summary text ready for PDF.
        """
        print('Starting LLM summary generation for the entire day...')
        client = OpenAI(base_url='http://localhost:8080/v1', api_key='sk-no-key-required')

        system_prompt = (
            "You are a personal assistant that helps summarize daily computer activity logs into a concise report.\n"
            "User will provide you a list of their computer activities throughout the day, each with a timestamp, "
            "duration, category, and keywords. Summarize these activities into a brief report highlighting the main "
            "tasks and focus areas of the day.\n"
            "View the user's activities as a sequence of states: some are high-intensity technical states (coding/math), "
            "while others are low-intensity recovery states (gaming/browsing). "
            "Do not categorize leisure as a 'distraction' or a moral failure; treat it as a deliberate shift in cognitive load. "
            "Report the facts of the day—durations, timestamps, and themes—in a single, dense, unformatted paragraph. "
            "Be direct and dry, but remain neutral about the user's choices."
            "Be inclusive, objective, and honest about the user's productivity, leisure, and other "
            "aspects of their computer usage as long as they take up significant time.\n"
            "Rules:\n"
            "1. Provide a concise, plain-text summary of the day based strictly on the log data.\n"
            "2. No Markdown formatting (no bolding, no lists, no headers).\n"
            "3. Avoid devaluing the user's work.\n"
            "4. If the data shows long stretches of work, acknowledge it\n"
            "5. Keep the response to a single, unformatted short paragraph.\n"
            "6. When possible, adequately reference specific time periods from the log to back up your observations (prevent mentioning too short, too fragmented time periods).\n"
            "7. Focus on patterns and behaviors rather than individual, random, isolated events."
        )

        user_prompt = (
            f"<overview>\n{json.dumps(overview_json)}\n</overview>\n\n"
            f"<chronological_ordered_activities>\n"
            f"{chr(10).join(final_report_entries)}\n"
            f"</chronological_ordered_activities>\n"
        )

        response = client.chat.completions.create(
            model='qwen3-14b',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            max_tokens=1000,
        )

        summary = response.choices[0].message.content
        # Clean typographic characters
        clean = summary.replace('\u2013', '-').replace('\u2014', '-')
        clean = clean.replace('\u201c', '"').replace('\u201d', '"')
        clean = clean.replace('\u2018', "'").replace('\u2019', "'")
        return clean

    # ── Public: Save PDF ─────────────────────────────────────────────────

    @staticmethod
    def save_pdf(figures, overview_json, clean_summary: str, date_str: str, output_path: str, start_time: datetime, num_images: int):
        """
        Save the four figures and summary text to a PDF file.

        Args:
            figures: Tuple of (fig_pie, fig_hourly, fig_chronological, fig_wc).
            overview_json: The overview JSON data.
            clean_summary: Cleaned summary text.
            date_str: Date string in 'YYYY-MM-DD' format.
            output_path: Destination file path for the PDF.
        """
        date_en = ReportGenerator._date_to_en(date_str)
        titles = ['Time Allocation', 'Hourly Screen Usage', 'Activity Timeline', 'Theme Focus']

        class _ActivityReport(FPDF):
            def __init__(self, header_text):
                super().__init__(orientation='P', unit='mm', format='A4')
                self._header_text = header_text

            def header(self):
                self.set_font('helvetica', 'B', 15)
                self.cell(0, 10, self._header_text, border=False, ln=True, align='C')
                self.ln(5)

        pdf = _ActivityReport(date_en)
        pdf.add_font('CustomFont', '', REGULAR_FONT_PATH, uni=True)
        pdf.add_font('CustomFont', 'B', BOLD_FONT_PATH, uni=True)
        pdf.add_page()
        pdf.set_font("CustomFont", size=12)

        fig_idx = 0
        for fig, title in zip(figures, titles):
            if pdf.get_y() > 200:
                pdf.add_page()
            pdf.set_font("CustomFont", 'B', 10)
            pdf.cell(0, 10, title, ln=True)

            img_buf = io.BytesIO()
            fig.savefig(img_buf, format='png', bbox_inches='tight', dpi=150)
            img_buf.seek(0)
            pdf.image(img_buf, x=15, w=180)
            pdf.ln(4)
            print(f'Added figure "{title}" to PDF.')

            if fig_idx == 0:
                # After pie chart, add in each category's name and its focus.
                # Set the category's title in bold and in its corresponding color, and the focus in normal font below it.
                for category in overview_json:
                    cat_name = category['category_name']
                    focus = category['focus']
                    cat_color = tuple(int(c * 255) for c in sns.color_palette('hls', n_colors=len(overview_json))[overview_json.index(category)])
                    pdf.set_font("CustomFont", 'B', 10)
                    pdf.set_text_color(*cat_color)
                    pdf.cell(0, 8, cat_name, ln=True)
                    pdf.set_font("CustomFont", '', 8)
                    pdf.set_text_color(0, 0, 0)
                    pdf.multi_cell(0, 6, focus, ln=True)
                    pdf.ln(1)

            fig_idx += 1

        pdf.add_page()
        pdf.set_font("CustomFont", 'B', 14)
        pdf.cell(0, 10, "PC Usage Summary", ln=True)
        pdf.ln(5)
        pdf.set_font("CustomFont", '', 11)
        pdf.multi_cell(0, 8, clean_summary)

        # Append generation timestamp as the footnote
        now_time = datetime.now()
        process_time = (now_time - start_time).total_seconds()  # The time taken to generate the report (including LLM calls)
        process_min, process_sec = divmod(process_time, 60)
        timestamp_str = f'Generated on {now_time.strftime("%Y-%m-%d %H:%M:%S")}, took {int(process_min)}m{int(process_sec)}s to generate report for {num_images} images. Avg: {process_time / num_images:.2f}s/image.'

        pdf.ln(5)
        pdf.set_font("CustomFont", 'B', 8)
        pdf.multi_cell(0, 5, timestamp_str)

        pdf.output(output_path)

if __name__ == '__main__':
    pass