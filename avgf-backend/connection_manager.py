import asyncio
import json
import base64
import os
import time
from fastapi import WebSocket
from fastapi.responses import JSONResponse
from typing import Dict
from pathlib import Path
import logging

UPLOAD_DIR = "./received_images"

Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str) -> bool:
        try:
            await websocket.accept()
            self.active_connections[client_id] = websocket
            logger.info(f"Client {client_id} connected.")
            return True
        except Exception as e:
            logger.exception(f"Websocket connection failed: {e}")
            return False

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected.")

    async def send_message(self, message: Dict, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(json.dumps(message))

    async def broadcast(self, message: str):
        for client_id in self.active_connections.keys():
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.exception(f"Error sending message to {client_id}: {e}")