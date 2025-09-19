import asyncio
import json
import base64
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from connectionmanager import ConnectionManager
from dotenv import load_dotenv
import uvicorn
import logging
from backend_redis import RedisMesssageBroker
from MessageTypes import MessageTypes
from contact_lens_detector import DetectContactLens


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


UPLOAD_DIR = "./received_images"

Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

#instantiate FastAPI app
app = FastAPI()

#instantiate ConnectionManager
connection_manager = ConnectionManager()

#instantiate RedisMessageBroker
redis_message_broker = RedisMesssageBroker()

#instantiate

load_dotenv()

image_key = os.getenv("IMAGE_KEY", "base64_data")
filename_key = os.getenv("FILENAME_KEY", "filename")
timestamp_key = os.getenv("TIMESTAMP_KEY", "timestamp")

@app.websocket("/image")
async def websocket_endpoint(websocket: WebSocket):
    
    ws_connected = await connection_manager.connect(websocket, client_id)
    try:
        while ws_connected:
            client_id = str(uuid.uuid4())[:8]  # Generate a short unique client ID
            redis_sub_messages = await redis_message_broker.subscribe()
            for message in redis_sub_messages:
                try:
                    data = message['data']
                    send_image_to_frontend(data,client_id)
                    

                except json.JSONDecodeError:
                    error_message = f"Json Decode Exception: {e}"
                    logger.exception(error_message)
                    await send_error(error_message)
                except Exception as e:
                    await send_error(f"Error processing message: {str(e)}, {client_id}")
    
    
    # ping_task = asyncio.create_task(ping_client(websocket, client_id))
        
    except WebSocketDisconnect:
        connection_manager.disconnect(client_id)
        ping_task.cancel()

async def send_error(error_message: str, client_id: str):
    logger.error(f"Error for client {client_id}: {error_message}")
    await connection_manager.send_message(
        message={
            "error": error_message,
            "client_id": client_id},
        client_id=client_id
    )


# async def ping_client(websocket: WebSocket, client_id: str):
#     try:
#         while True:
#             await asyncio.sleep(30)  # Ping every 30 seconds
#             await connection_manager.send_message(json.dumps({"type": "ping"}))
#     except WebSocketDisconnect:
#         connection_manager.disconnect(client_id)
#         ping_task.cancel()
#     except Exception as e:
#         print(f"Error in pinging client {client_id}: {e}")
#         connection_manager.disconnect(client_id)
#         ping_task.cancel()


# async def handle_message(message: dict, client_id: str):
#     """
#     Handle incoming messages based on their type.
#     """
#     msg_type = message.get("type")
#     if msg_type == "image":
#         await handle_image_message(message, client_id)
#     elif msg_type == "save":
#         await handle_save_image(message, client_id)
#     else:
#         error_message = f"Unknown message type: {msg_type}"
#         await send_error(error_message, client_id)    

def send_image_to_frontend(message_data: dict, client_id):
    this_message = {"origin":MessageTypes.SHOW_IMAGE,"data":message_data}
    connection_manager.send_message(message=this_message,client_id=client_id)
    


async def handle_image_message():
    try:


        
        await connection_manager.send_message(
            message={
                "message": "Connected", 
                "client_id": client_id
                }, 
            client_id=client_id)
        filename = message.get(filename_key)
        base64_data = message.get(image_key)
        timestamp = message.get(timestamp_key)
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        if not filename or not base64_data:
            await send_error(client_id, "Missing filename or base64_data in the message")
            return
        
        print(f"Received and processing image from {client_id}: {filename} at {timestamp}")
        
        try:
            image_bytes = base64.b64decode(base64_data)
            detection_outcome = process_image(image_bytes)
        except Exception as e:
            error_message = f"Error decoding base64 data: {e}"
            await send_error(error_message, client_id)
            return
    except Exception as e:
        error_message = f"Error processing grpc message:{e}"
        await send_error(error_message, client_id)
        return

# async def handle_save_image(message: str, client_id: str):
#     try:
#         filename = message.get(filename_key)
#         base64_data = message.get(image_key)
#         timestamp = message.get(timestamp_key)
#         file_extension = Path(filename).suffix.lower()
#         base_name = Path(filename).stem
#         unique_filename = f"{base_name}_{int(time.time())}{file_extension}"
#         save_path = os.path.join(UPLOAD_DIR, unique_filename)
        
#         with open(save_path, "wb") as image_file:
#             image_file.write(image_bytes)
        
#         print(f"Image saved to {save_path}")
        
#         await connection_manager.send_message(
#             message={
#                 "message": "Image received and saved",
#                 "file_path": save_path
#             },
#             client_id=client_id
#         )
#     except Exception as e:
#         error_message = f"Error saving image: str({e})"
#         await send_error(error_message, client_id)

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )