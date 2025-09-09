import redis
import redis.asyncio as aioredis
import json
import logging
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RedisPubSub:
    def __init__(self, host='localhost', port=6379, db=0):
        self.redis_host = host
        self.redis_port = port
        self._sync_client = None
        self._async_client = None
        self.pubsub = None
        self.pubsub_async = None
        self.redis_db = db
        logger.info(f"Instantiated RedisPubSub for {self.redis_host}:{self.redis_port}")

    def _get_sync_client(self) -> redis.Redis:
        if self._sync_client is None:
            self._sync_client = redis.Redis(host=self.redis_host, port=self.redis_port, db=self.redis_db)
        return self._sync_client
    
    def _get_async_client(self) -> aioredis.Redis:
        if self._async_client is None:
            self._async_client = aioredis.Redis(host=self.redis_host, port=self.redis_port, db=0)
        return self._async_client
    # Context manager methods for synchronous usage
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    # Async context manager methods
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_async()
        return False 
        
    ################################################
    #              Synchronous Methods            #
    ################################################    
        
    def publish(self, channel: str, payload: dict) -> bool:
        try:
            client = self._get_sync_client()
            message = json.dumps(payload)           
            result = client.publish(channel, message)
            logger.debug(f"Published to {channel}, {result} subscribers received")
            return True
        except Exception as e:
            logger.exception(f"Error publishing message to Redis: {str(e)}")
    
    def subscribe_with_timeout(self, channel: str, timeout :float=1):
        client = self._get_sync_client()
        pubsub = None
        try:
            self.pubsub = client.pubsub()
            self.pubsub.subscribe(channel)
            logger.info(f"Subscribed to channel '{channel}' (sync)")
            
            while True:
                try:
                    # Wait up to 1 second for a message
                    message = self.pubsub.get_message(timeout=timeout)
                    
                    if message is None:
                        # Timeout occurred - no message received
                        # This gives Python a chance to process Ctrl+C
                        continue
                        
                    # Skip subscription confirmation messages
                    if message['type'] == 'subscribe':
                        logger.info(f"Subscribed to {message['channel'].decode('utf-8')}")
                        continue
                        
                    # Process actual messages
                    if message['type'] == 'message':
                        try:
                            data = json.loads(message['data'].decode('utf-8'))
                            yield data
                        except (json.JSONDecodeError, UnicodeDecodeError) as e:
                            logger.exception(f"Error decoding message: {e}")
                        
                except KeyboardInterrupt:
                    logger.exception("Subscription stopped by user")
        except Exception as e:
            logger.exception(f"Error subscribing to Redis channel '{channel}': {str(e)}")
        finally:
            if self.pubsub:
                try:
                    self.pubsub.unsubscribe(channel)
                    self.pubsub.close()
                    logger.info("Redis subscriber is closed")
                except Exception as e:
                    logger.error(f"Error closing pubsub: {e}")
    
    ################################################
    #              Asynchronous Methods            # 
    ################################################
    async def publish_async(self, channel: str, payload: dict):
        try:
            client = self._get_async_client()
            message = json.dumps(payload)
            result = await client.publish(channel, message)
            logger.debug(f"Published to {channel}, {result} subscribers received")
            return True
        except Exception as e:
            print(f"Error publishing message to Redis: {str(e)}")
        finally:
            if client:
                client.close()
       
    async def subscribe_async_with_timeout(self, channel: str, timeout=1.0):
        client = self._get_async_client()
        this_timeout = timeout
        try:
            self.pubsub_async = client.pubsub()
            await self.pubsub_async.subscribe(channel)
            logger.info(f"Subscribed to channel '{channel}' (async)")
            
            while True:
                try:
                    message = await asyncio.wait_for(
                        self.pubsub_async.get_message(),
                        timeout=this_timeout
                    )
                    if message is None:
                        continue
                    if message and message['type'] == 'subscribe':
                        logger.info(f"Subscribed to {message['channel'].decode('utf-8')}")
                        continue
                    if message and message['type'] == 'message':
                        try:
                            yield json.loads(message['data'].decode('utf-8'))
                        except (json.JSONDecodeError, UnicodeDecodeError) as e:
                            logger.error(f"Error decoding message from '{channel}': {e}")
                
                except asyncio.TimeoutError as e:
                    #Timeout is normal, continue listening
                    continue
        except KeyboardInterrupt:
            logger.info("Subscription stopped by user")
        except Exception as e:
            logger.error(f"Error in async subscription: {e}")
        finally:
            if self.pubsub_async:
                try:
                    await self.pubsub_async.unsubscribe(channel)
                    await self.pubsub_async.close()
                    logger.info(f"Unsubscribed from channel '{channel}' (async)")
                except Exception as e:
                    logger.error(f"Error closing subscriber: {e}")
    
    ################################################
    #              Cleanup Methods                 # 
    ################################################
    
    def close(self):
        try:
            if self.pubsub:
                self.pubsub.close()
                self.pubsub=None
            if self._sync_client:
                self._sync_client.close()
                self._sync_client = None
                logger.info("Sync Redis client closed")
        except Exception as e:
            logger.error(f"Error closing sync resources: {e}")
    
    def close_async(self):
        try:
            if self.pubsub_async:
                self.pubsub_async.close()
                self.pubsub_async=None
            if self._async_client:
                self._async_client.close()
                self._async_client = None
                logger.info("Async Redis client closed")
        except Exception as e:
            logger.error(f"Error closing async resources: {e}")