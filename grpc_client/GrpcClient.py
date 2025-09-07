# image_grabber_client.py
import logging
import grpc
import os
import sys

root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,root_dir)
                           
import image_grabber_pb2
import image_grabber_pb2_grpc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageGrpcClient:
    def __init__(self, server_address):
        self.server_address = server_address
        self.channel = None
        self.stub = None
        self._connect()

    def _connect(self):
        self.channel = grpc.insecure_channel(self.server_address)
        self.stub = image_grabber_pb2_grpc.ImageServiceStub(self.channel)
        logger.info(f"Connected to gRPC server at {self.server_address}")

    def send_image(self, filename, image_base64, timestamp):
        try:
            request = image_grabber_pb2.ImageRequest(
                filename=filename,
                image_base64=image_base64,
                timestamp=timestamp
            )
            response = self.stub.SendImage(request)
            return response.success
        except grpc.RpcError as e:
            logger.error(f"gRPC error: {e.code()} - {e.details()}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return False
    
    def close(self):
        if self.channel:
            self.channel.close()