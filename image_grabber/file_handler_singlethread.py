import os
import base64
import time
import logging
from watchdog.events import FileSystemEventHandler
from threading import Lock, Thread, Timer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FileHandler(FileSystemEventHandler):
    def __init__(self, comm_client, batch_size = 2, batch_timeout= 2.0):
        self.comm_client = comm_client
        self.supported_formats = {".jpg", ".jpeg", ".png","bmp","webp"}
        
        #Batch configuration
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        
        # Batch storage
        self.pending_files = []
        self.batch_timer = None    
        self.processed_files = set()
        
    def on_created(self, event):
        self._handle_event(event)
        
    def on_modified(self, event):
        self._handle_event(event)
    
    def _handle_event(self, event):
        if event.is_directory:
            return
        
        file_path = os.path.abspath(event.src_path)

        if file_path in self.processed_files:
            return
        
        self.processed_files.add(file_path)
        
        logger.info(f"Detected new file: {file_path}")
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext not in self.supported_formats:
            logger.warning(f"Unsupported file format: {file_ext}")
            return
        processed_file = self._process_file(file_path) 
        
        if processed_file:
           self._add_to_batch(processed_file)
                
    def _process_file(self, file_path, max_retries=5, retry_delay=0.5):
        """Process single file"""
        filename = os.path.basename(file_path)
        logger.info(f"Processing: {filename}")
        
        for attempt in range(max_retries):
            if not os.path.exists(file_path):
                logger.warning(f"File no longer exists: {filename}")
                return None
                        
            try:
                time.sleep(0.1)
                with open(file_path, "rb") as image_file:
                    filename = os.path.basename(file_path)
                    image_data = image_file.read()
                    image_base64 = base64.b64encode(image_data).decode("utf-8")
                    timestamp = os.path.getctime(file_path)
                    logger.info(f"filename:{filename}, timestamp:{timestamp}, image_base64:{image_base64[:10]}")
                    return {
                        'filename': filename,
                        'image_base64': image_base64,
                        'timestamp': timestamp
                    }
                    
            except (PermissionError, OSError) as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Retry  {attempt+1} for {filename}")
                    time.sleep(retry_delay)
                    continue
                else:    
                    logger.error(f"Error reading file {file_path}: {e}")
                    return None
        return None

    def _add_to_batch(self, processed_file):
        self.pending_files.append(processed_file)
        logger.info(f"Added to batch: {processed_file['filename']}"
                    f"(batch size: {len(self.pending_files)}/{self.batch_size})"
                    )
        if len(self.pending_files) >= self.batch_size:
            self._send_batch()
        else:
            #reset/restart timeout timer
            if self.batch_timer:
                self.batch_timer.cancel()
            self.batch_timer = Timer(self.batch_timeout, self._send_batch)
            self.batch_timer.start()
            
    def _send_batch(self):
        logger.info(f"Inside _send_batch")
        if not self.pending_files:
            return
        logging.info(f"Sending a batch of {len(self.pending_files)} files")
        
        if self.batch_timer:
            self.batch_timer.cancel()
            self.batch_timer = None
        
        try:
            for file_data in self.pending_files:
                success = self.comm_client.send_image(file_data['filename'],file_data['image_base64'],file_data['timestamp'])
            
            if not success:
                logger.error (f"Failed to send {file_data['filename']}")
                
            logger.info("Batch sent successfully") 
            self.pending_files.clear()
            
        except Exception as e:
            logger.error(f"Failed to send batch: {e}")
            
    def force_send_batch(self):
        """Manually trigger sending of current batch (useful for shutdown)"""
        if self.pending_files:
            logger.info("Force sending remaining batch...")
            self._send_batch_on_timeout()

    def get_status(self):
        """Get current processing status"""
        pending_count = len(self.pending_files)
        
        return {
            "queued_files": self.file_queue.qsize(),
            "pending_batch": pending_count,
            "batch_size_limit": self.batch_size,
            "batch_timeout": self.batch_timeout
        }