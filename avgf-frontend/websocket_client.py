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
    def __init__(self):
        self.ws = None
        self.connected = False
        self.messages = []
        self.latest_image_left = None
        self.latest_image_right = None
    
    def connect(self, url):
        def _on_message(ws, message):
            """Handle incoming websocket message"""
            try:
                message_json = json.loads(message)
                image_base64 = message_json['image_base64']
                # implement message handling logic
            except json.JSONDecodeError:
                try:
                    self._handle_image_base64(message)
                except Exception as e:
                    self.messages.append(f"Error processing data: {str(e)}")
        
        def _on_open(ws):
            logger.info(f"Websocket connection open")
        
        def _on_error(ws, error):
            logger.error(f"Error msg: {error}")
        
        def _on_close(ws, close_status_code, close_msg):
            logger.info(f"Websocket connection is closed: {close_msg}")

        try:
            self.ws = websocket.WebSocketApp(
                url,
                on_open = _on_open,
                on_message = _on_message,
                on_error = _on_error,
                on_close = _on_close

            )
            # Run websocket in a separate thread
            ws_thread = threading.Thread(target=self.ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()

            return True
        except Exception as e:
            logger.error(f"Exception occured: {e}")
            return False
        
    def disconnect(self):
        if self.ws:
            self.ws.close()
        self.connected = False
    
    def send_message(self, message):
        """Send message to Websocket server"""
        if self.connected and self.ws:
            try:
                self.ws.send(message)
                logger.info(f"Send message: {message}")
            except Exception as e:
                logger.error(f"Send error: {str(e)}")
        
    def _handle_image_base64(self,image_base64: str):
            """Process base64 image data"""
            try:
                if image_base64.startswith('data:'):
                    image_base64 = image_base64.split(',',1)[1]
                
                #Decode base64 to image
                image_bytes = base64.b64decode(image_base64)
                image = Image.open(BytesIO(image_bytes))
            except Exception as e:
                logger.error(f"Exception occured: {e}")
            

    
        

