import os
import base64
import logging
from concurrent import futures
import grpc
import image_grabber_pb2
import image_grabber_pb2_grpc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'   )
logger = logging.getLogger(__name__)

class ImageGrabberGrpcService(image_grabber_pb2_grpc.ImageGrabberServicer):    
    def __init__(self, upload_dir="./received_images"):
        self.upload_dir = upload_dir
        os.makedirs(self.upload_dir, exist_ok=True)
        self.redis_client = RedisMessageBroker() # Initialize Redis client here

    def SendImage(self, request, context):
        try:
            logger.info(f"Received image with filename: {request.filename}")
            image_data = base64.b64decode(request.image_base64)
            filename = request.filename if request.filename else "received_image"
            filepath = os.path.join(self.upload_dir, filename)
            
            with open(filepath, "wb") as image_file:
                image_file.write(image_data)
            
            logger.info(f"Received and saved image: {filepath}")
            return image_grabber_pb2.ImageResponse(status="success", message=f"Image saved as {filename}")
        except Exception as e:
            logger.error(f"Error saving image: {str(e)}")
            context.set_details(f"Error saving image: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return image_grabber_pb2.ImageResponse(status="failure", message=str(e))
    
    def _save_image_to_file(self, toSave: bool, image_base64 : str, filename: str, timestamp: str):
        if toSave:
            try:
                image_data = base64.b64decode(image_base64)
                filename = filename if filename else f"image_{timestamp}.jpg"
                filepath = os.path.join(self.upload_dir, filename)
                
                with open(filepath, "wb") as image_file:
                    image_file.write(image_data)
                
                logger.info(f"Image saved to {filepath}")
            except Exception as e:
                logger.error(f"Error saving image: {str(e)}")
        else:
            logger.info("Image saving is disabled.")
        