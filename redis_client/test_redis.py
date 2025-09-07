from RedisClient import RedisPubSub
import logging
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Running in sync mode")
    redis_client = RedisPubSub()
    messages = redis_client.subscribe_with_timeout("upload_image")
    try:
        for message in messages:
            print(message)
    except KeyboardInterrupt as e:
        logger.exception(f"Program interrupted by user")
    finally:
        redis_client.unsubscribe("upload_image")

async def main_async():
    logger.info("Running in async mode")
    redis_client = RedisPubSub()
    async def listen_for_messages():
        async for message in redis_client.subscribe_async_with_timeout("upload_image"):
            logger.info("Waiting for message...")
            print(message)
    try:
       asyncio.wait_for(listen_for_messages(),timeout=0)
    except KeyboardInterrupt as e:
        logger.exception(f"Redis subscriber stopped by user")
    except asyncio.CancelledError:
        logger.info("Task was cancelled")
    except Exception as e:
        logger.exception(f"Other exceptions: {e}")
    finally:
        try:
            if redis_client.pubsub_async:
                redis_client.pubsub_async.close()
            if redis_client._async_client:
                redis_client._async_client.close()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        await redis_client.unsubscribe_async("upload_image")

if __name__=="__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")