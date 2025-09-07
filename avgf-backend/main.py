import asyncio
import json
import base64
import os, sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from connection_manager import ConnectionManager
from dotenv import load_dotenv
import uvicorn
import logging
import uuid
from contact_lens_detection.contact_lens_detection import AnalyzeImageAsync

from redis_servicer import RedisServicer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

#instantiate FastAPI app
app = FastAPI()

#instantiate ConnectionManager
connection_manager = ConnectionManager()

#instantiate RedisClient
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", 6379))
redis_servicer = RedisServicer(redis_host, redis_port)

@app.websocket("/analyze")
async def websocket_endpoint(websocket: WebSocket):
    client_id = str(uuid.uuid4())[:8]  # Generate a short unique client ID
    
    try: 
        await connection_manager.connect(websocket, client_id)
        logger.info(f"Client {client_id} connected")    
        subscription_task = asyncio.create_task(handle_redis_subscription(client_id,websocket))
        try:
            await subscription_task
        except asyncio.CancelledError:
            logger.info(f"Subscription task cancelled for client {client_id}")
    except WebSocketDisconnect:
        connection_manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"Unexpected error for client {client_id}: {str(e)}")
        try:
            await send_error(f"Server error: {str(e)}",client_id)
        except:
            pass
    finally:
        # Cleanup
        if subscription_task and not subscription_task.done():
            subscription_task.cancel()
            try:
                await subscription_task
            except asyncio.CancelledError:
                pass
        connection_manager.disconnect(client_id)
        logger.info(f"Cleanup completed for client {client_id}")
    
async def send_error(error_message: str, client_id: str):
    try:
        logger.error(f"Error for client {client_id}: {error_message}")
        message= {
            "origin": "error",
            "data": error_message,
        }
        await connection_manager.send_message(message, client_id)
    except Exception as e:
        logger.error(f"Failed to send error message to client {client_id}: {e}")
    
async def handle_redis_subscription(client_id: str, websocket: WebSocket):
    try:
        async for sub_message in redis_servicer.run_async(
            channel='upload_image',
            timeout=30
        ):
            try:
                logger.info(f"Sending message to client {client_id}: {sub_message}")
                await connection_manager.send_message(sub_message, client_id)
            except json.JSONDecodeError:
                await send_error("Invalid JSON format",client_id)
            except Exception as e:
                logger.error(f"Error sending message to client {client_id}: {str(e)}")
    except asyncio.CancelledError:
        logger.info(f"Redis subscription cancelled for client {client_id}")
        raise
    except Exception as e:
        logger.error(f"Error in Redis subscription for client {client_id}: {str(e)}")
        try:
            await send_error(f"Subscription error: {str(e)}",client_id)
        except:
            pass 


if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )