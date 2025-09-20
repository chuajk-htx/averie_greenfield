# redis_image_handler.py
import asyncio
from dataclasses import dataclass
from enum import Enum
import json
import base64
import queue
import threading
import time
from io import BytesIO
from PIL import Image
from typing import Dict, Any, List, Optional
import logging
import os, sys
import redis
import streamlit as st

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parent_dir)

from redis_client.RedisClient import RedisPubSub

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProcessStatus(Enum):
    RECEIVED = "received"
    PROCESSING = "processing"  
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class ImageTask:
    task_id: str
    filename: str
    image_base64: str
    timestamp: float
    status: ProcessStatus
    result: Any = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'filename': self.filename,
            'image_base64': self.image_base64,
            'timestamp': self.timestamp,
            'status': self.status.value,
            'result': self.result,
            'error_message': self.error_message
        }
    

class RedisSubscriberManager:
    """Redis subscriber that continuously updates Streamlit session state with images"""
    
    def __init__(self, redis_config : Dict['str',Any], channels: list):
        self.redis_config = redis_config
        self.channels = channels
        self.subscriber_thread = None
        self.task_queue = queue.Queue()
        self.is_running = False
   
    def start_subscriber(self):
        """Start the Redis subscriber in a separate thread"""
        if not self.is_running:
            self.subscriber_thread = threading.Thread(
                target=self._subscriber_worker,
                daemon=True
            )
            self.subscriber_thread.start()
            self.is_running = True
    
    def stop_subscriber(self):
        """Stop the Redis subscriber thread"""
        if self.is_running:
            self.is_running = False
            if self.subscriber_thread is not None:
                self.subscriber_thread.join()
    
    @st.cache_resource
    def _get_redis_client(self):
        return RedisPubSub(**self.redis_config)
    
    def _subscriber_worker(self):
        """Worker function that runs in a separate thread to handle Redis subscription"""
        try:
            # with RedisPubSub(**self.redis_config) as redis_client:
            #     for channel in self.channels:
            #         redis_client.subscribe_with_timeout(channel, timeout=5.0)               
            #         logger.info(f"Subscribed to Redis channel: {channel}")
            #         for message in redis_client.subscribe_with_timeout(channel, timeout=30.0):
            #             if not self.is_running:
            #                 break
            #             logger.info(f"Received message on channel {channel}:{message}")
            #             self._process_message(channel,message)
            redis_client = self._get_redis_client()
            #pubsub = redis_client.pubsub()
            
            for channel in self.channels:
                redis_client.subscribe(channel)
                logger.info(f"Subscribed to Redis channel: {channel}")
            
            for message in redis_client.subscribe_with:
                if not self.is_running:
                    break
                if message['type'] == 'message':
                    self._process_message(message)
            
        except Exception as e:
            st.error(f"Error in Redis subscription: {e}")
        # finally:
        #     if 'pubsub' in locals():
        #         try:
        #             pubsub.close()
        #             logger.info("Redis pubsub connection closed")
        #         except Exception as e:
        #             logger.error(f"Error closing Redis pubsub: {e}")
    
    def _process_message(self, message: Dict[str, Any]):
        """Process incoming Redis message and update session state"""
        data = json.loads(message.get('data'))
        image_task = ImageTask(
            task_id=f"{data.get('filename','unknown')}",
            filename=data.get('filename', 'unknown'),
            image_base64=data.get('image_base64', ''),
            timestamp=data.get('timestamp', time.time()),
            status=ProcessStatus.RECEIVED,
            result = None,
            error_message = None
        )
        self._add_to_session_queue(image_task)
        
    
    def _add_to_session_queue(self, task:ImageTask):
        """Add a task to the session state queue"""
        if 'image_tasks' not in st.session_state:
            st.session_state['image_tasks'] = {}
            
        st.session_state['image_tasks'][task.task_id] = task.to_dict()
        logger.info(f"Added task {task.task_id} to session state")
        
        #re-render Streamlit app to reflect new state
        #st.rerun()

class AnalyticProcessor:
    """Process images from Redis and update their status"""
    
    def __init__(self):
        self.processing_queue = queue.Queue()
        self.results_queue = queue.Queue()
        self.is_running = False
        self.worker_thread = None
        self.lock= threading.Lock()
    
    def start_processing(self):
        """Start the image processing thread"""
        if not self.is_running:
            self.worker_thread = threading.Thread(
                target=self._processing_worker,
                daemon=True
            )
            self.worker_thread.start()
            self.is_running = True
    
    def stop_processing(self):
        """Stop the image processing thread"""
        if self.is_running:
            self.is_running = False
            if self.processor_thread is not None:
                self.processor_thread.join()
    
    def submit_task(self, task_id: str, image_base64: str):
        """Submit a task for processing"""
        self.processing_queue.put({
          'task_id': task_id,
          'image_base64': image_base64,
          'timestamp': time.time()
        })
    
    def get_completed_results(self) -> List[Dict[str, Any]]:
        """Retrieve all completed results"""
        results = []
        while True:
            try:
                logger.debug("Inside get_completed_results")
                result = self.results_queue.get_nowait()
                results.append(result)
            except queue.Empty:
                break
        return results
    
    def _processing_worker(self):
        """Worker function that processes images"""
        while self.is_running:
            try:
                task_data: ImageTask = self.processing_queue.get(timeout=1)
                logger.info(f"Processing task {task_data['task_id']}")
                result = self._process_image_analytics(task_data)
                self.results_queue.put({
                    'task_id': task_data.get('task_id','unknown'),
                    'result': result,
                    'status': ProcessStatus.COMPLETED.value if result.get('success') else ProcessStatus.ERROR.value,
                    'timestamp': time.time(),
                    })
                logger.info(f"Completed task {task['task_id']}")
                logger.info(f"Result: {result}")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.exception(f"Error processing image: {e}")
                self.results_queue.put({
                        'task_id': task_data.get('task_id','unknown'),
                        'result': None,
                        'status': ProcessStatus.ERROR.value,
                        'error_message': str(e),
                        'timestamp': time.time()
                    })
    
    def _process_image_analytics(self, task_data: ImageTask) -> ImageTask:
        """Simulate image processing and update task status"""
        try:
            #Actual analytics processing logic would go here
            image_base64 = task_data['image_base64']
            
            #Simulate processing (e.g., call your ML model here)
            time.sleep(3)  # Simulate processing delay
            
            return {
                'success': True,
                'contact_lens_detected': True,
                'confidence': 0.95,
            }
            
        except Exception as e:
            return {
                'success': False,
                'error_message': str(e)
            }
        