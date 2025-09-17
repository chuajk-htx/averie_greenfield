import asyncio
import json
import base64
import os
import time
from fastapi import WebSocket
from fastapi.responses import JSONResponse
from typing import Any, Dict
from pathlib import Path
import logging
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket) -> str | None:
        """Accept connection and return client ID"""
        try:
            await websocket.accept()
            client_id = str(uuid.uuid4())[:8]  # Generate a short unique client ID
            self.active_connections[client_id] = websocket
            logger.info(f"Client {client_id} connected.")
            return client_id
        except Exception as e:
            logger.exception(f"Websocket connection failed: {e}")
            return None

    def disconnect(self, client_id: str):
        """Remove client from active connections"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected.")

    async def send_message(self, message: Any, client_id: str) -> bool:
        """Send message to a specific client"""
        if(type(message) != dict):
            logger.error("Message must be in JSON format (dict)")
            return False
        if(type(message) != str):
            message = json.dumps(message)
        if client_id in self.active_connections:
            try:           
                websocket = self.active_connections[client_id]
                await websocket.send_text(message)
                return True
            except Exception as e:
                logger.exception(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)
                return False
        return False
    
    async def broadcast(self, message: str):
        disconnect_clients = []
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.exception(f"Error sending message to {client_id}: {e}")
                disconnect_clients.append(client_id)
        for client_id in disconnect_clients:
            self.disconnect(client_id)
    
    def get_connection_count(self) -> int:
        """Return the number of active connections"""
        return len(self.active_connections)