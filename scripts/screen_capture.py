import os
import ctypes
import time
import cv2
import sys
import json
import numpy as np
from skimage.metrics import structural_similarity as ssim
from datetime import datetime
import pygetwindow as gw
from PIL import ImageGrab, Image
from pynput import mouse, keyboard


class ScreenCapture:
    '''Main class for capturing and managing screenshots of active windows.'''
    
    def __init__(self, output_dir, interval=60, idle_threshold=0.95):
        '''
        Initialize ScreenCapture with configuration.
        
        Args:
            output_dir (str): Directory to save screenshots
            interval (int): Seconds between capture attempts
            idle_threshold (float): Similarity threshold for idle detection (0.0-1.0)
        '''
        self.output_dir = output_dir
        self.interval = interval
        self.idle_threshold = idle_threshold
        
        self.last_window_name = None
        self.last_screenshot = None
        self.last_input_time = time.time()
        self.screenshot_count = 0
        self.lock_file = None     # Prevent multiple instances of this script running
        
        self._setup_dpi_awareness()
        self._setup_input_listeners()
        self._acquire_lock()
    
    def _acquire_lock(self):
        '''Acquire a lock file to prevent multiple instances.'''
        lock_dir = os.path.dirname(os.path.abspath(__file__))
        lock_file_path = os.path.join(lock_dir, 'screen_capture.lock')

        try:
            self.lock_file = open(lock_file_path, 'w')
            import msvcrt
            msvcrt.locking(self.lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            self.lock_file.write(str(os.getpid()))
            self.lock_file.flush()
            print(f'Screen Capture lock acquired, PID: {os.getpid()}')
        except (IOError, OSError):
            print('Error: Another instance of Screen Capture is already running.')
            print('Exiting.')
            sys.exit(1)

    def _release_lock(self):
        '''Release the lock file.'''
        if self.lock_file:
            try:
                import msvcrt
                msvcrt.locking(self.lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                self.lock_file.close()
                os.remove(self.lock_file.name)
                print('Screen Capture lock released.')
            except Exception as e:
                print(f'Error releasing lock: {e}')

    def _setup_dpi_awareness(self):
        '''Make the process DPI aware to get correct screen coordinates.'''
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
        except:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except:
                pass
    
    def _setup_input_listeners(self):
        '''Set up mouse and keyboard listeners to track user input.'''
        self.mouse_listener = mouse.Listener(
            on_move=self._update_activity,
            on_click=self._update_activity,
            on_scroll=self._update_activity
        )
        self.keyboard_listener = keyboard.Listener(
            on_press=self._update_activity,
            on_release=self._update_activity
        )
        self.mouse_listener.start()
        self.keyboard_listener.start()
    
    def _update_activity(self, *args):
        '''Update the timestamp of last user activity.'''
        self.last_input_time = time.time()
    
    def get_active_window_name(self):
        '''
        Returns the name of the currently active window.
        Returns None if no window is active or an error occurs.
        '''
        try:
            active_window = gw.getActiveWindow()
            if active_window:
                return active_window.title
        except Exception as e:
            print(f'Error retrieving active window: {e}')
        return None
    
    def capture_active_window_screenshot(self):
        '''
        Captures a screenshot of the currently active window.
        Returns None if no window is active or an error occurs.
        '''
        try:
            active_window = gw.getActiveWindow()
            if active_window:
                left = max(0, active_window.left)
                top = max(0, active_window.top)
                right = active_window.right
                bottom = active_window.bottom
                
                screenshot = ImageGrab.grab(bbox=(left, top, right, bottom), all_screens=True)
                return screenshot
        except Exception as e:
            print(f'Error capturing screenshot: {e}')
        return None
    
    def is_idle_window(self, window_name1, window_name2, img1, img2):
        '''
        Determines if the window is idle by comparing window name, size, and pixel similarity.
        Returns True if idle (similarity > threshold), False otherwise.
        '''
        # Check if window names differ
        if window_name1 != window_name2:
            print(f'Window names differ: "{window_name1}" vs "{window_name2}"')
            return False
        
        # Check if either image is None
        if img1 is None or img2 is None:
            print(f'One of the images is None.')
            return False
        
        # Check if image sizes differ
        if img1.size != img2.size:
            print(f'Image sizes differ: {img1.size} vs {img2.size}')
            return False
        
        # Compare pixel similarity
        try:
            img1_cv = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
            img2_cv = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)

            # Resize while maintaining aspect ratio
            max_size = 960
            scale = max_size / max(img1_cv.shape[:2])
            new_size = (int(img1_cv.shape[1] * scale), int(img1_cv.shape[0] * scale))
            img1_cv = cv2.resize(img1_cv, new_size)
            img2_cv = cv2.resize(img2_cv, new_size)
            
            # Dynamically set win_size based on image dimensions
            min_dim = min(img1_cv.shape[:2])
            win_size = min(11, min_dim if min_dim % 2 == 1 else min_dim - 1)
            
            similarity_score = ssim(img1_cv, img2_cv, win_size=win_size, channel_axis=2)
            similarity = ((similarity_score + 1) / 2)
            print(f'Window idle similarity: {similarity:.4f} (threshold: {self.idle_threshold:.4f})')
            return similarity > self.idle_threshold
        except Exception as e:
            print(f'Error comparing images: {e}')
            return False
    
    def save_screenshot(self, window_name, screenshot):
        '''
        Saves a screenshot with the window name and timestamp.
        Returns the filepath of the saved screenshot, or None if an error occurs.
        '''
        date = datetime.now().strftime('%Y-%m-%d')  # E.g., '2026-02-03'
        image_folder = os.path.join(self.output_dir, date)
        os.makedirs(image_folder, exist_ok=True)

        try:
            # Cap the maximum dimensions
            max_width, max_height = 1920, 1080
            downscaled_screenshot = screenshot.copy()
            if downscaled_screenshot.width > max_width or downscaled_screenshot.height > max_height:
                downscaled_screenshot.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                screenshot = downscaled_screenshot

            # Generate timestamp
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

            # Sanitize window name for filename
            forbidden_chars = ['\\', '/', ':', '*', '?', '"', '<', '>', '|', '_']
            safe_window_name = ''.join(
                c if c.isalnum() or c not in forbidden_chars else ' ' 
                for c in window_name
            ).strip()
            
            # Limit filename length
            max_length = 128
            if len(safe_window_name) > max_length:
                safe_window_name = safe_window_name[:max_length]
            
            filename = f'{timestamp}_{safe_window_name}.jpg'
            filepath = os.path.join(image_folder, filename)

            screenshot.save(filepath, 'JPEG')
            print(f'Screenshot saved to {filepath}')
            return filepath
        except Exception as e:
            print(f'Error saving screenshot: {e}')
            return None
    
    def should_skip_window(self, window_name):
        '''Check if window should be skipped (system windows, etc).'''
        skip_windows = ['Program Manager', 'Windows Shell Experience Host', 'Task Switching', '']
        return window_name in skip_windows
    
    def is_no_user_input(self):
        '''Check if there has been no user input for the interval.'''
        return time.time() - self.last_input_time > self.interval
    
    def run(self):
        '''Main loop to capture screenshots.'''
        print(f'Capturing screenshot every {self.interval} seconds to {self.output_dir}')
        print('Press Ctrl+C to stop.')
        
        try:
            while True:
                time.sleep(self.interval)
                
                # Capture current window
                new_window_name = self.get_active_window_name()
                new_screenshot = self.capture_active_window_screenshot()

                # Skip system windows
                if self.should_skip_window(new_window_name) or new_window_name is None or new_screenshot is None:
                    print('No active window to capture.')
                    continue
                
                # Check if idle (both screenshot unchanged and no user input)
                screen_is_idle = self.is_idle_window(
                    self.last_window_name, 
                    new_window_name, 
                    self.last_screenshot, 
                    new_screenshot
                )
                no_input = self.is_no_user_input()
                
                if screen_is_idle and no_input:
                    print('Screen did not change and no user input detected, skipping screenshot.')
                    continue
                
                # Save the screenshot
                result = self.save_screenshot(new_window_name, new_screenshot)
                if result is not None:
                    self.screenshot_count += 1

                # Update tracking variables
                self.last_window_name = new_window_name
                self.last_screenshot = new_screenshot

        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        '''Stop listeners and cleanup.'''
        self.mouse_listener.stop()
        self.keyboard_listener.stop()
        self._release_lock()
        print(f'\nScreenshot capture stopped by user, total screenshots taken: {self.screenshot_count}')


if __name__ == '__main__':
    # Load configuration from config.json
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Create and run
    screen_capture = ScreenCapture(
        output_dir=config['screenshot_dir'],
        interval=config['capture_interval'],
        idle_threshold=config['image_similarity_threshold']
    )
    screen_capture.run()
    