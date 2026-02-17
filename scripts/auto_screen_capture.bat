@echo off
REM Run this file automatically on system startup to capture screenshots at regular intervals

echo Start automatic screen capture with interval %CAPTURE_INTERVAL% seconds...
..\venv\Scripts\python.exe screen_capture.py

