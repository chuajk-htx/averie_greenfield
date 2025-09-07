import os
import base64
import time
import logging
from watchdog.events import FileSystemEventHandler


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FileHandler(FileSystemEventHandler):
    def __init__(self, comm_client):
        self.comm_client = comm_client
        self.supported_formats = {".jpg", ".jpeg", ".png","bmp","webp"}

    def on_created(self, event):
        self._handle_event(event)
        
    def on_modified(self, event):
        self._handle_event(event)
    
    def _handle_event(self, event):
        if event.is_directory:
            return
        
        file_path = os.path.abspath(event.src_path)
        logger.info(f"Detected new file: {file_path}")
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext not in self.supported_formats:
            logger.warning(f"Unsupported file format: {file_ext}")
            return

        self._handle_image(file_path)
    
        

    def _handle_image(self, image_path, max_retries=5, retry_delay=0.5):
            for attempt in range(max_retries):
                if os.path.exists(image_path): 
                    try:
                        with open(image_path, "rb") as image_file:
                            filename = os.path.basename(image_path)
                            image_data = image_file.read()
                            image_base64 = base64.b64encode(image_data).decode("utf-8")
                            timestamp = os.path.getctime(image_path)
                            logger.info(f"filename:{filename}, timestamp:{timestamp}, image_base64:{image_base64[:10]}")
                            break
                    except Exception as e:
                        logger.error(f"Error reading file {image_path}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)

            success = self.comm_client.send_image(filename, image_base64, timestamp)
            if success:
                logger.info(f"Image sent to backend successfully: {filename}")
            else:
                logger.error(f"Image sent to backend failed: {filename}")
