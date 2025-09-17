from time import time
import websocket
import json
import logging
from typing import Dict, Any
import base64
from io import BytesIO
from PIL import Image
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WebSocketClient:
    def __init__(self, server_url: str):
        self.ws = None
        self.connected = False
        self.messages = []
        self.latest_image_left = None
        self.latest_image_right = None
        self.server_url = server_url
        self._connection_lock = threading.Lock()
    
    def connect(self):
        def on_message(ws, message):
            """Handle incoming websocket message"""
            try:
                self._handle_message(message)            
            except Exception as e:
                self.messages.append(f"Error processing data: {str(e)}")
        
        def on_open(ws):
            with self._connection_lock:
                self.connected = True
            logger.info(f"Websocket connection open")
        
        def on_error(ws, error):
            with self._connection_lock:
                self.connected = False
            logger.error(f"Error msg: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            self.connected = False
            logger.info(f"Websocket connection is closed: {close_msg}")

        try:
            self.ws = websocket.WebSocketApp(
                self.server_url,
                on_open = on_open,
                on_message = on_message,
                on_error = on_error,
                on_close = on_close

            )
            # Run websocket in a separate thread
            ws_thread = threading.Thread(target=self.ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            
            #Added: Wait for connection with timeout
            timeout = 1  # seconds
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)

            return self.connected
        except Exception as e:
            logger.error(f"Exception occured: {e}")
            return False
        
    def disconnect(self):
        with self._connection_lock:
            if self.ws:
                self.ws.close()
            self.connected = False
    
    def send_message(self, payload):
        """Send message to Websocket server"""
        with self._connection_lock:
            is_connected = self.connected and self.ws is not None
            
        if is_connected:
            try:
                self.ws.send(payload)
                logger.info(f"Send message: {payload}")
            except Exception as e:
                logger.error(f"Send error: {str(e)}")
                with self._connection_lock:
                    self.connected = False
        else:
            logger.warning("Cannot send message: WebSocket is not connected.")
                    
    def _handle_message(self, message: str):
        """Process base64 image data"""
        try:
            logger.debug(f"received message: {message}") 
        except Exception as e:
            logger.error(f"Exception occured: {e}")
            

    
        

