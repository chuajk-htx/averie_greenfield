# redis_image_handler.py
import asyncio
import json
import base64
import threading
import time
from io import BytesIO
from PIL import Image
from typing import Dict, Any, List, Optional
import logging
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamlitRedisImageSubscriber:
    """Redis subscriber that continuously updates Streamlit session state with images"""
    
    def __init__(self, redis_servicer):
        self.redis_servicer = redis_servicer
        self.subscription_task = None
        self.running = False
        self._loop = None
        self._thread = None
    
    def start_background_subscription(self, channel: str = 'upload_image'):
        """Start Redis subscription in a background thread"""
        if self.running:
            logger.info("Subscription already running")
            return
        
        # Initialize Streamlit session state
        if 'redis_images' not in st.session_state:
            st.session_state.redis_images = []
        if 'redis_messages' not in st.session_state:
            st.session_state.redis_messages = []
        if 'subscription_status' not in st.session_state:
            st.session_state.subscription_status = "stopped"
        
        # Start background thread
        self._thread = threading.Thread(
            target=self._run_subscription_thread,
            args=(channel,),
            daemon=True
        )
        self._thread.start()
        
        logger.info("Started Redis subscription in background")
    
    def _run_subscription_thread(self, channel: str):
        """Run Redis subscription in separate thread with its own event loop"""
        try:
            # Create new event loop for this thread
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            
            # Run the subscription
            self._loop.run_until_complete(self._subscription_coroutine(channel))
            
        except Exception as e:
            logger.error(f"Subscription thread error: {e}")
            st.session_state.subscription_status = f"error: {str(e)}"
        finally:
            if self._loop:
                self._loop.close()
    
    async def _subscription_coroutine(self, channel: str):
        """The actual Redis subscription coroutine"""
        self.running = True
        st.session_state.subscription_status = "running"
        
        try:
            logger.info(f"Starting Redis subscription on channel: {channel}")
            
            async for sub_message in self.redis_servicer.run_async(
                channel=channel,
                timeout=30
            ):
                try:
                    # Process the message
                    await self._process_redis_message(sub_message)
                    
                except Exception as e:
                    logger.error(f"Error processing Redis message: {e}")
                    # Continue subscription even if one message fails
                    continue
                    
        except asyncio.CancelledError:
            logger.info("Redis subscription cancelled")
            st.session_state.subscription_status = "cancelled"
        except Exception as e:
            logger.error(f"Redis subscription error: {e}")
            st.session_state.subscription_status = f"error: {str(e)}"
        finally:
            self.running = False
            logger.info("Redis subscription ended")
    
    async def _process_redis_message(self, message: Dict[str, Any]):
        """Process individual Redis message and update session state"""
        try:
            logger.info(f"Processing Redis message: {message.get('type', 'unknown')}")
            
            # Store raw message
            st.session_state.redis_messages.append({
                'message': message,
                'timestamp': time.time(),
                'processed': False
            })
            
            # Keep only last 50 messages
            if len(st.session_state.redis_messages) > 50:
                st.session_state.redis_messages = st.session_state.redis_messages[-50:]
            
            # Process image if present
            if message.get('type') == 'image_upload' and 'image_base64' in message:
                image_info = await self._process_image_data(message)
                if image_info:
                    # Add to images list for display
                    st.session_state.redis_images.append(image_info)
                    
                    # Keep only last 10 images to manage memory
                    if len(st.session_state.redis_images) > 10:
                        st.session_state.redis_images = st.session_state.redis_images[-10:]
            
            # Mark as processed
            if st.session_state.redis_messages:
                st.session_state.redis_messages[-1]['processed'] = True
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            raise
    
    async def _process_image_data(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process base64 image data for Streamlit display"""
        try:
            image_base64 = message.get('image_base64', '')
            
            if not image_base64:
                return None
            
            # Remove data URL prefix if present
            if image_base64.startswith('data:'):
                image_base64 = image_base64.split(',', 1)[1]
            
            # Decode base64 to image
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_bytes))
            
            # Create image info for Streamlit
            image_info = {
                'image': image,
                'filename': message.get('filename', 'unknown.jpg'),
                'timestamp': message.get('timestamp', time.time()),
                'side': message.get('side', 'center'),
                'metadata': message.get('metadata', {}),
                'message_id': message.get('client_id', 'unknown')
            }
            
            logger.info(f"Processed image: {image_info['filename']} ({image.size})")
            return image_info
            
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            return None
    
    def stop_subscription(self):
        """Stop the Redis subscription"""
        if self.subscription_task:
            self.subscription_task.cancel()
        
        self.running = False
        st.session_state.subscription_status = "stopped"
        logger.info("Redis subscription stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get subscription status"""
        return {
            'running': self.running,
            'thread_alive': self._thread.is_alive() if self._thread else False,
            'images_count': len(st.session_state.get('redis_images', [])),
            'messages_count': len(st.session_state.get('redis_messages', [])),
            'status': st.session_state.get('subscription_status', 'unknown')
        }