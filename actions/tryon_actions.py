"""Background actions for try-on processing"""
from typing import Dict
import logging
from services.tryon_service import process_virtual_tryon
from services.email_service import send_completion_email_sync, send_error_email_sync
from config import FRONTEND_URL

logger = logging.getLogger(__name__)


async def process_and_send_email(
    user_id: str, 
    email: str, 
    garment_images: Dict[str, str], 
    person_image: str, 
    subscription_type: str,
    collection: str
):
    """Process try-on and send email (runs in background)"""
    try:
        processed_images = await process_virtual_tryon(user_id, garment_images, person_image)
        logger.info(f"Processing completed for user: {user_id}")
        
        # Send completion email
        send_completion_email_sync(email, user_id, processed_images, subscription_type, collection)
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error processing try-on for user {user_id}: {error_message}")
        
        # Send error email
        send_error_email_sync(email, user_id, error_message, subscription_type, collection, FRONTEND_URL)

