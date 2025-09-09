import os
import base64
import time
import logging
from watchdog.observers import Observer
from dotenv import load_dotenv
from file_handler_singlethread import FileHandler
from comm_client import CommClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    load_dotenv()
    WATCH_FOLDER = os.getenv("WATCH_FOLDER", ".\watch_folder")
    HOST = os.getenv("HOST", "localhost")
    PORT = os.getenv("PORT", "6379")

    #Ensure watch folder exists
    os.makedirs(WATCH_FOLDER, exist_ok=True)
    
    logger.info(f"Starting Image Grabber Service")
    logger.info(f"Watching folder: {WATCH_FOLDER}")
    
    #Set up file system watcher
    this_client = CommClient(CommType="redis", host=HOST, port=PORT)
    event_handler = FileHandler(comm_client=this_client)
    observer = Observer()
    observer.schedule(event_handler, WATCH_FOLDER, recursive=False)

    try:
        observer.start()
        logger.info("Image Grabber Service started. Press Ctrl-C to stop")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Image Grabber Service shutting down...")
        observer.stop()
    finally:
        observer.join()


if __name__ == "__main__":
    main()
