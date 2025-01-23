import asyncio
import websockets
import json
import assemblyai as aai
from assemblyai.extras import MicrophoneStream
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import socket
from dotenv import load_dotenv
from aiohttp import web
from aiohttp_cors import setup, ResourceOptions

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Set your AssemblyAI API key dynamically from environment variables
def get_assemblyai_api_key():
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        logger.warning("ASSEMBLYAI_API_KEY environment variable not found. Falling back to local key.")
        api_key = "your-local-api-key"  # Replace with a valid fallback key for local testing
    logger.info(f"Loaded API Key: {api_key}")
    return api_key

aai.settings.api_key = get_assemblyai_api_key()

class TranscriptionManager:
    def __init__(self, loop):
        self.transcriber = None
        self.microphone_stream = None
        self.websocket = None
        self.running = False  # Indicates whether transcription is active
        self.last_transcript_received = datetime.now()
        self.terminated = False  # Indicates if transcription has been stopped
        self.loop = loop
        self._executor = ThreadPoolExecutor()

    async def send_message(self, message):
        """Send a message to the WebSocket client."""
        if self.websocket and self.websocket.open:
            try:
                await self.websocket.send(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message: {e}")

    def on_data(self, transcript):
        """Handle transcription data."""
        if self.terminated:
            return

        if transcript.text == "":
            if (datetime.now() - self.last_transcript_received).total_seconds() > 5:
                logger.info("5 seconds of silence detected. Terminating transcription.")
                self.terminate_transcription()
            return

        if isinstance(transcript, aai.RealtimeFinalTranscript):
            asyncio.run_coroutine_threadsafe(
                self.send_message({"type": "final", "text": transcript.text}), self.loop
            )
        else:
            asyncio.run_coroutine_threadsafe(
                self.send_message({"type": "partial", "text": transcript.text}), self.loop
            )

        self.last_transcript_received = datetime.now()

    def on_error(self, error):
        """Handle transcription errors."""
        logger.error(f"AssemblyAI Error: {error}")
        asyncio.run_coroutine_threadsafe(
            self.send_message({"type": "error", "text": str(error)}), self.loop
        )

    def on_open(self, session):
        """Handle session start."""
        logger.info(f"Session opened: {session.session_id}")

    def terminate_transcription(self):
        """Terminate the transcription session."""
        if not self.terminated:
            logger.info("Terminating transcription session...")
            if self.transcriber:
                self.transcriber.close()
            self.terminated = True

    async def start_transcription(self, websocket):
        """Start transcription."""
        if self.running:
            logger.warning("Transcription is already running. Restarting...")
            await self.stop_transcription()

        self.websocket = websocket
        self.running = True
        self.terminated = False

        try:
            self.transcriber = aai.RealtimeTranscriber(
                sample_rate=16_000,
                on_data=self.on_data,
                on_error=self.on_error,
                on_open=self.on_open,
            )
            logger.info("Connecting to AssemblyAI...")
            self.transcriber.connect()

            self.microphone_stream = MicrophoneStream(sample_rate=16_000)
            logger.info("Starting transcription stream...")
            await self.loop.run_in_executor(
                self._executor,
                self.transcriber.stream,
                self.microphone_stream,
            )
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            await self.send_message({"type": "error", "text": str(e)})
            await self.stop_transcription()

    async def stop_transcription(self):
        """Stop transcription."""
        if not self.running:
            logger.info("No transcription is running to stop.")
            return

        logger.info("Stopping transcription...")
        self.running = False

        if self.microphone_stream:
            self.microphone_stream.close()
            self.microphone_stream = None

        if self.transcriber:
            self.transcriber.close()
            self.transcriber = None

        await self.send_message({"type": "status", "text": "Transcription stopped"})
        logger.info("Transcription fully stopped.")

async def handle_websocket(websocket, path, loop):
    """Handle WebSocket connections."""
    logger.info("New WebSocket connection")
    manager = TranscriptionManager(loop)

    try:
        async for message in websocket:
            logger.info(f"Received WebSocket message: {message}")
            if message == "start":
                await manager.start_transcription(websocket)
            elif message == "stop":
                logger.info("Stop command received.")
                await manager.stop_transcription()
            else:
                logger.warning(f"Unknown command received: {message}")
    except websockets.exceptions.ConnectionClosed:
        logger.info("WebSocket connection closed.")
    finally:
        await manager.stop_transcription()


def start_webserver(port, directory):
    """Start a simple web server to serve static files."""
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)
    httpd = HTTPServer(("", port), Handler)
    logger.info(f"Web server running on http://localhost:{port}")
    httpd.serve_forever()


async def main():
    """Start the WebSocket server."""
    loop = asyncio.get_running_loop()

    # Determine the host and port dynamically based on the environment
    host = "0.0.0.0" if os.getenv("RAILWAY_ENVIRONMENT") else "localhost"
    port = int(os.getenv("PORT", 8080))  # Default to 8080 for Railway

    # Find a free port for the web server.
    webserver_port = 8000
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", webserver_port))
                break
        except OSError:
            webserver_port += 1

    # Start the web server in a separate thread
    webserver_thread = threading.Thread(target=start_webserver, args=(webserver_port, os.path.dirname(os.path.abspath(__file__))))
    webserver_thread.daemon = True  # Makes sure the thread exits when the program exits
    webserver_thread.start()

    # Create an aiohttp web application for handling CORS
    app = web.Application()
    cors = setup(app, defaults={
        "*": ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    })

    # Add the websocket handler to the aiohttp app
    app.add_routes([web.get('/ws', lambda request: websockets.serve(
        lambda ws, path: handle_websocket(ws, path, loop),
        host,
        port,
    ))])

    # Register CORS for the route
    for route in list(app.router.routes()):
        cors.add(route)

    logger.info(f"WebSocket server running on ws://{host}:{port}")
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()

    await asyncio.Future()  # Run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user.")
