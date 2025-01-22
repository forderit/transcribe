@echo off
REM Batch file to run the simplified WebSocket server

REM Activate Python virtual environment (if applicable)
REM Uncomment the next line if you are using a virtual environment
REM call venv\Scripts\activate

REM Set Python path
set PYTHONPATH=.

REM Run the WebSocket server
python websocket_server.py

REM Pause to keep the terminal open
pause
