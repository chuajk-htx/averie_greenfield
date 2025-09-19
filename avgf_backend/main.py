import asyncio
import json
import os
from dotenv import load_dotenv
import logging
#from contact_lens_detection.contact_lens_detection import AnalyzeImageAsync

from redis_servicer import RedisServicer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()


#instantiate RedisClient
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", 6379))
redis_servicer = RedisServicer(redis_host, redis_port)

async def handle_redis_subscription():
    try:
        async for sub_message in redis_servicer.run_async(
            channel='upload_image',
            timeout=30
        ):
            try:
                logger.info(f"'Message': {sub_message}")
            except json.JSONDecodeError:
                await logger.error("Invalid JSON format")
            except Exception as e:
                logger.error(f"Error yielding message: {str(e)}")
    except asyncio.CancelledError:
        logger.info(f"Redis subscription cancelled")
        raise
    except Exception as e:
        logger.error(f"Error in Redis subscription: {str(e)}")

async def main():
    # Start the Redis subscription handler
    await handle_redis_subscription()


if __name__ == "__main__":
    asyncio.run(main())