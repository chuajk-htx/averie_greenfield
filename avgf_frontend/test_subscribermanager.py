import base64
from io import BytesIO
import logging
import time
from live_image_processor import RedisSubscriberManager, AnalyticProcessor, ProcessStatus
import streamlit as st
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 0
}
CHANNELS = ['upload_image']

def initialize_components():
    """Initialize Components"""
    if 'subscriber_manager' not in st.session_state:
        st.session_state['subscriber_manager'] = RedisSubscriberManager(REDIS_CONFIG, CHANNELS)
        st.session_state['subscriber_manager'].start_subscriber()
    if 'analytics_processor' not in st.session_state:
        st.session_state['analytics_processor'] = AnalyticProcessor()
        st.session_state['analytics_processor'].start_processing()
    
    if 'image_tasks' not in st.session_state:
        st.session_state['image_tasks'] = {}

def process_analytics():
    """Process analytics for images in session state"""
    if 'analytics_processor' not in st.session_state:
        return
    results = st.session_state['analytics_processor'].get_completed_results()
    print(f"Length of results: {len(results)}")
    for result in results:
        task_id = result['task_id']
        if task_id in st.session_state['image_tasks']:
            st.session_state['image_tasks'][task_id]['status'] = result['status']
            if result['status'] == ProcessStatus.COMPLETED.value:
                st.session_state.image_tasks[task_id]['result'] = result['result']
            elif result['status'] == ProcessStatus.ERROR.value:
                st.session_state.image_tasks[task_id]['error_message'] = result.get('error_message')

def display_image_with_analytics(task_id:str, task_data: dict):
    
    col1, col2 = st.columns([2,1])
    
    with col1:
        try:
            image_bytes = BytesIO(base64.b64decode(task_data['image_base64']))
            image = Image.open(image_bytes)
            st.image(image, 
                     caption=f"Filename: {task_data['filename']}\nStatus: {task_data['status']}",
                     use_column_width=True
                     )
        except Exception as e:
            st.error(f"Error displaying image: {e}")
    
    with col2:
        status = task_data['status']
        if status == ProcessStatus.RECEIVED.value:
            st.info("Starting analysis...")
            st.session_state['analytics_processor'].submit_task(task_id, task_data['image_base64'])
        elif status == ProcessStatus.PROCESSING.value:
            st.warning("Analysis in progress...")
        elif status == ProcessStatus.COMPLETED.value:
            st.success("Analysis completed!")
            if task_data.get('result'):
                st.json(task_data['result'])
        elif status == ProcessStatus.ERROR.value:
            st.error(f"Error during analysis: {task_data.get('error_message', 'Unknown error')}")

def test_main():
    st.title("Live Image Processor Test")
    # Initialize RedisSubscriberManager
    initialize_components()
    
    # Start the subscriber
    #process_analytics()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh"):
            st.rerun()
            process_analytics()
    
    #Display images
    if st.session_state.get('image_tasks'):
        sorted_tasks = sorted(st.session_state['image_tasks'].items(),
                              key=lambda item: item[1]['timestamp'], 
                              reverse=True
                              )
        for task_id, task_data in sorted_tasks:
            with st.container():
                st.markdown("---")
                display_image_with_analytics(task_id, task_data)
    
    else:
        st.info("No images received yet. Waiting for images on Redis channel...")
   
   
    # Stop the subscriber
    #redis_manager.stop_subscriber()
    
   
    
if __name__ == "__main__":
    test_main()