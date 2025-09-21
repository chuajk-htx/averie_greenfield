import sys
import os
import json
import asyncio
from typing import Callable, Optional, Any, Dict

# Add the parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from grpc_client.GrpcClient import ImageGrpcClient
from redis_client.RedisClient import RedisPubSub
from websocket_client.WebsocketClient import WebSocketClient

class CommClient:
    def __init__(self, CommType: str, host: str, port: str):
        self.comm_type = CommType.lower()
        self._message_handler: Optional[Callable] = None
        self._is_receiving = False
        
        try:
            if self.comm_type not in ["grpc", "redis", "websocket"]:
                raise ValueError("Unsupported communication type. Use 'gRPC', 'Redis', or 'WebSocket'.")
        except ValueError as ve:
            print(str(ve))
            return 
            
        if self.comm_type == "grpc":
            grpc_server_address = f"{host}:{port}"
            self._client = ImageGrpcClient(server_address=grpc_server_address)
        elif self.comm_type == "redis":
            redis_host = host
            redis_port = int(port)
            self._client = RedisPubSub(host=redis_host, port=redis_port)
        elif self.comm_type == "websocket":
            websocket_server_url = f"ws://{host}:{port}/analyze"
            self._client = WebSocketClient(server_url=websocket_server_url)
        
    def send_image(self, filename: str, image_base64: str, timestamp: float) -> bool:
        try:
            if self.comm_type == "grpc":
                return self._client.send_image(filename, image_base64, timestamp)
            elif self.comm_type == "redis":
                payload = {
                    "filename": filename,
                    "image_base64": image_base64,
                    "timestamp": timestamp
                }
                self._client.publish(channel="upload_image", payload=payload)
            elif self.comm_type == "websocket":
                self._client.connect()
                payload = {
                    "filename": filename,
                    "image_base64": image_base64,
                    "timestamp": timestamp
                }
                self._client.send_message(payload=json.dumps(payload))
            return True
        except Exception as e:
            print(f"Error sending image: {str(e)}")
            return False
    
    async def send_image_async(self, filename: str, image_base64: str, timestamp: float) -> bool:
        try:
            if self.comm_type == "grpc":
                return self._client.send_image(filename, image_base64, timestamp)
            elif self.comm_type == "redis":
                payload = {
                    "filename": filename,
                    "image_base64": image_base64,
                    "timestamp": timestamp
                }
                await self._client.publish_async(channel="upload_image", payload=payload)
            elif self.comm_type == "websocket":
                if not self._client.is_connected():
                    await self._client.connect_async()
                payload = {
                    "filename": filename,
                    "image_base64": image_base64,
                    "timestamp": timestamp
                }
                await self._client.send_message_async(payload=json.dumps(payload))
            return True
        except Exception as e:
            print(f"Error sending image: {str(e)}")
            return False

    def set_message_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Set the callback function to handle received messages"""
        self._message_handler = handler

    def start_receiving(self, channel: str = "upload_image") -> bool:
        """Start receiving messages synchronously"""
        if self._message_handler is None:
            print("Error: No message handler set. Use set_message_handler() first.")
            return False
            
        try:
            self._is_receiving = True
            
            if self.comm_type == "grpc":
                # For gRPC, assuming there's a receive_messages method
                for message in self._client.receive_messages():
                    if not self._is_receiving:
                        break
                    self._message_handler(message)
                        
            elif self.comm_type == "redis":
                # Use the actual RedisPubSub methods
                for message in self._client.subscribe_with_timeout(channel, timeout=1.0):
                    if not self._is_receiving:
                        break
                    self._message_handler(message)
                            
            elif self.comm_type == "websocket":
                if not self._client.is_connected():
                    self._client.connect()
                
                while self._is_receiving:
                    message = self._client.receive_message()
                    if message:
                        try:
                            # Parse JSON message if possible
                            payload = json.loads(message) if isinstance(message, str) else message
                            self._message_handler(payload)
                        except json.JSONDecodeError:
                            self._message_handler(message)
                            
            return True
        except Exception as e:
            print(f"Error receiving messages: {str(e)}")
            return False

    async def start_receiving_async(self, channel: str = "upload_image") -> bool:
        """Start receiving messages asynchronously"""
        if self._message_handler is None:
            print("Error: No message handler set. Use set_message_handler() first.")
            return False
            
        try:
            self._is_receiving = True
            
            if self.comm_type == "grpc":
                # For gRPC, assuming there's an async receive method
                async for message in self._client.receive_messages_async():
                    if not self._is_receiving:
                        break
                    self._message_handler(message)
                        
            elif self.comm_type == "redis":
                # Use the actual RedisPubSub async methods
                async for message in self._client.subscribe_async_with_timeout(channel, timeout=1.0):
                    if not self._is_receiving:
                        break
                    self._message_handler(message)
                            
            elif self.comm_type == "websocket":
                if not self._client.is_connected():
                    await self._client.connect_async()
                
                while self._is_receiving:
                    message = await self._client.receive_message_async()
                    if message:
                        try:
                            payload = json.loads(message) if isinstance(message, str) else message
                            self._message_handler(payload)
                        except json.JSONDecodeError:
                            self._message_handler(message)
                    await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
                            
            return True
        except Exception as e:
            print(f"Error receiving messages: {str(e)}")
            return False

    def stop_receiving(self):
        """Stop receiving messages"""
        self._is_receiving = False
        
        try:
            if self.comm_type == "redis":
                # The RedisPubSub class handles cleanup internally
                # when the generator exits, so we just need to signal stop
                pass
        except Exception as e:
            print(f"Error stopping message reception: {str(e)}")

    def receive_single_message(self, channel: str = "upload_image", timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """Receive a single message with timeout"""
        try:
            if self.comm_type == "grpc":
                # Assuming gRPC client has a method to receive single message
                return self._client.receive_single_message(timeout=timeout)
                
            elif self.comm_type == "redis":
                # For single message, we need to create a temporary subscription
                try:
                    # Use the generator to get just one message
                    for message in self._client.subscribe_with_timeout(channel, timeout=timeout):
                        return message  # Return the first message received
                    return None  # No message received within timeout
                except Exception:
                    return None
                
            elif self.comm_type == "websocket":
                if not self._client.is_connected():
                    self._client.connect()
                
                message = self._client.receive_message(timeout=timeout)
                if message:
                    try:
                        return json.loads(message) if isinstance(message, str) else message
                    except json.JSONDecodeError:
                        return message
                return None
                
        except Exception as e:
            print(f"Error receiving single message: {str(e)}")
            return None

    async def receive_single_message_async(self, channel: str = "upload_image", timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """Receive a single message asynchronously with timeout"""
        try:
            if self.comm_type == "grpc":
                return await self._client.receive_single_message_async(timeout=timeout)
                
            elif self.comm_type == "redis":
                # For single message async, use the async generator
                try:
                    async for message in self._client.subscribe_async_with_timeout(channel, timeout=timeout):
                        return message  # Return the first message received
                    return None  # No message received within timeout
                except Exception:
                    return None
                
            elif self.comm_type == "websocket":
                if not self._client.is_connected():
                    await self._client.connect_async()
                
                message = await self._client.receive_message_async(timeout=timeout)
                if message:
                    try:
                        return json.loads(message) if isinstance(message, str) else message
                    except json.JSONDecodeError:
                        return message
                return None
                
        except Exception as e:
            print(f"Error receiving single message: {str(e)}")
            return None

    def close(self):
        """Close the connection and cleanup resources"""
        self.stop_receiving()
        try:
            if hasattr(self._client, 'close'):
                self._client.close()
        except Exception as e:
            print(f"Error closing client: {str(e)}")