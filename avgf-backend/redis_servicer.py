import logging
import asyncio
import os, sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parent_dir)
from redis_client.RedisClient import RedisPubSub

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RedisServicer:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        
    def run(self, channel: str, timeout: float=1.0):  
        logger.info("Running Redis subscription in sync mode")
        with RedisPubSub(host=self.host, port=self.port) as redis_client:
            try:
                for message in redis_client.subscribe_with_timeout("upload_image", timeout=timeout):
                    print(f"Received: {message}")
                    yield message['image_base64'][:25]
            except KeyboardInterrupt:
                logger.info("Program interrupted by user")
            except Exception as e:
                logger.exception(f"Error in sync subscription: {e}")
    
    async def run_async(self, channel:str, timeout: float):
        logger.info(f"Starting async Redis subscription to channgel '{channel}'")
        
        async with RedisPubSub(host=self.host, port=self.port) as redis_client:
            try:                
                async for message in redis_client.subscribe_async_with_timeout(channel=channel, timeout=timeout):
                    logger.info("Waiting for message...")
                    yield message['image_base64'][:25]
            except KeyboardInterrupt as e:
                logger.exception(f"Redis subscriber stopped by user")
            except asyncio.CancelledError:
                logger.info("Task was cancelled")
                raise
            except Exception as e:
                logger.exception(f"Other exceptions: {e}")
                raise
        
