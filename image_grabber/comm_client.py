import sys
import os

# Add the parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #os.path.dirname gets the path to directory where the current script is
sys.path.insert(0, parent_dir)
from grpc_client.GrpcClient import ImageGrpcClient
from redis_client.RedisClient import RedisPubSub

class CommClient:
    def __init__(self, CommType: str, host: str, port: str):
        self.comm_type = CommType.lower()
        try:
            if self.comm_type not in ["grpc", "redis"]:
                raise ValueError("Unsupported communication type. Use 'gRPC' or 'Redis'.")
        except ValueError as ve:
            print(str(ve))
            return 
        if self.comm_type == "grpc":
            grpc_server_address = f"{host}:{port}"
            self._client = ImageGrpcClient(server_address=grpc_server_address)
        if self.comm_type == "redis":
            redis_host = host
            redis_port = int(port)
            self._client = RedisPubSub(host=redis_host, port=redis_port)
        
    def send_image(self, filename: str, image_base64: str, timestamp: float) -> bool:
        try:
            if self.comm_type == "grpc":
                    return self._client.send_image(filename, image_base64, timestamp)
            if self.comm_type == "redis":
                    payload = {
                        "filename": filename,
                        "image_base64": image_base64,
                        "timestamp": timestamp
                    }
                    self._client.publish(channel="upload_image", payload=payload)
            return True
        except Exception as e:
            print(f"Error sending image: {str(e)}")
            return False
    
    async def send_image_async(self, filename: str, image_base64: str, timestamp: float) -> bool:
        try:
            if self.comm_type == "grpc":
                    return self._client.send_image(filename, image_base64, timestamp)
            if self.comm_type == "redis":
                    payload = {
                        "filename": filename,
                        "image_base64": image_base64,
                        "timestamp": timestamp
                    }
                    await self._client.publish_async(channel="upload_image", payload=payload)
            return True
        except Exception as e:
            print(f"Error sending image: {str(e)}")
            return False
        

    def close(self):
        self._client.close()
            
        # Add any necessary cleanup for Redis client if needed