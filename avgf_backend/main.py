import asyncio
import base64
import json
import os
from dotenv import load_dotenv
import logging
#from contact_lens_detection.contact_lens_detection import AnalyzeImageAsync

from redis_servicer import RedisServicer
from comm_client2 import CommClient

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

received_image_files_dir = os.getenv("IMAGE_FILES_DIR", "./received_image_files")
os.makedirs(received_image_files_dir, exist_ok=True)

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

def handle_comm_client_message(message):
    # Setup message handler
    def handle_received_message(message):
        filename = message.get('filename', 'unknown')
        file_base = os.path.splitext(filename)[0]
        file_ext = os.path.splitext(filename)[1]
        image_base64 = message.get('image_base64', '')
        image_base64_trunc = image_base64[:25] + "..." if len(image_base64) > 25 else image_base64
        logger.info(f"Message content (first 25 chars): {image_base64_trunc}")
        message_timestamp = message.get('timestamp', 'unknown')
        logger.info(f"Message timestamp: {message_timestamp}")
        new_filename = f"{file_base}_{int(message_timestamp*1000)}{file_ext}"
        new_filepath = os.path.join(received_image_files_dir, new_filename)
        with open(new_filepath, "wb") as img_file:
            img_file.write(base64.b64decode(image_base64))
        

    client = CommClient("redis", redis_host, redis_port)
    client.set_message_handler(handle_received_message)

    # Option 1: Continuous receiving
    client.start_receiving("upload_image")

    # Option 2: Single message
    #message = client.receive_single_message("upload_image", timeout=10.0)

    # Option 3: Async receiving
    #await client.start_receiving_async("upload_image")

async def main():
    # Start the Redis subscription handler
    #await handle_redis_subscription()
    handle_comm_client_message("test")


if __name__ == "__main__":
    asyncio.run(main())