import datetime
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
        self.pending_messages = []
        self.batch_timer = None    
        self.processed_files = set()
        
    def on_created(self, event):
        if len(self.processed_files) >2:
            self.processed_files.clear()
            logger.info("Cleared processed_files set to manage memory")
        self._handle_event(event)
        
    def on_modified(self, event):
        if len(self.processed_files) >2:
            self.processed_files.clear()
            logger.info("Cleared processed_files set to manage memory")
        self._handle_event(event)
    
    def _handle_event(self, event):
        if event.is_directory:
            return
        
        filepath = os.path.abspath(event.src_path)
        
        if filepath in self.processed_files:
            return

        self.processed_files.add(filepath)
        
        logger.info(f"Detected new file: {filepath}")
        file_ext = os.path.splitext(filepath)[1].lower()

        if file_ext not in self.supported_formats:
            logger.warning(f"Unsupported file format: {file_ext}")
            return
        message_to_send = self._process_file(filepath) 
        
        if message_to_send:
           self._add_to_batch(message_to_send)
        
                
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
                        'image_base64': image_base64[:10],
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

    def _add_to_batch(self, message_to_send: dict):
        self.pending_messages.append(message_to_send)
        logger.info(f"Added to batch: {message_to_send['filename']}"
                    f"(batch size: {len(self.pending_messages)}/{self.batch_size})"
                    )
        if len(self.pending_messages) >= self.batch_size:
            self._send_batch()
        else:
            #reset/restart timeout timer
            if self.batch_timer:
                self.batch_timer.cancel()
            self.batch_timer = Timer(self.batch_timeout, self._send_batch)
            self.batch_timer.start()
            
    def _send_batch(self):
        logger.info(f"Inside _send_batch")
        if not self.pending_messages:
            return
        logging.info(f"Sending a batch of {len(self.pending_messages)} files")
        
        if self.batch_timer:
            self.batch_timer.cancel()
            self.batch_timer = None
        
        try:
            for file_data in self.pending_messages:
                success = self.comm_client.send_image(file_data['filename'],file_data['image_base64'],file_data['timestamp'])
            
            if not success:
                logger.error (f"Failed to send {file_data['filename']}")
                
            logger.info("Batch sent successfully") 
            self.pending_messages.clear()
            
            # for old_path in list(self.processed_files):
            #     self._rename_processed_files(old_path)
            # logger.info("Processed files renamed successfully")
            
        except Exception as e:
            logger.error(f"Failed to send batch: {e}")
            
    def force_send_batch(self):
        """Manually trigger sending of current batch (useful for shutdown)"""
        if self.pending_messages:
            logger.info("Force sending remaining batch...")
            self._send_batch_on_timeout()

    def get_status(self):
        """Get current processing status"""
        pending_count = len(self.pending_messages)
        
        return {
            "queued_files": self.file_queue.qsize(),
            "pending_batch": pending_count,
            "batch_size_limit": self.batch_size,
            "batch_timeout": self.batch_timeout
        }
    
    #Not implemented yet
    def _rename_processed_files(self, old_path: str):
        """Handle file renaming events"""
        if old_path in self.processed_files:
            if not os.path.exists(old_path):
                logger.warning(f"File does not exist for renaming: {old_path}")
                return
            
            dir_name, base_name = os.path.split(old_path)
            name, ext = os.path.splitext(base_name)
            new_name = f"{name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
            new_path = os.path.join(dir_name, new_name)
            
            try:
                os.rename(old_path, new_path)
                logger.info(f"File renamed from {old_path} to {new_path}")
                self.processed_files.remove(old_path)
                logger.info(f"Removed {old_path} from processed_files set")
            except Exception as e:
                logger.error(f"Error renaming file {old_path} to {new_path}: {e}")
                