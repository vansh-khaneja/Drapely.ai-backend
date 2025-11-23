"""Virtual try-on processing service"""
from typing import Dict
from PIL import Image
import logging
from services.image_service import download_image
from services.modal_service import process_tryon_batch_with_modal
from services.cloudinary_service import upload_to_cloudinary
from services.garment_description_service import generate_garment_description
from config import MODAL_ENDPOINT

logger = logging.getLogger(__name__)


async def process_virtual_tryon(user_id: str, garment_images: Dict[str, str], person_image: str) -> Dict[str, str]:
    """Download images, process try-on using batch endpoint, and save results in batches of 2"""
    logger.info(f"Starting try-on processing for user: {user_id}")
    
    # Download person image
    logger.info(f"Downloading person image from: {person_image}")
    person_img = await download_image(person_image)
    
    # Download all garment images and generate descriptions
    garment_img_dict = {}
    garment_descriptions = {}
    for product_id, garment_url in garment_images.items():
        logger.info(f"Downloading garment {product_id} from: {garment_url}")
        garment_img = await download_image(garment_url)
        garment_img_dict[product_id] = garment_img
        
        # Generate description for this garment using OpenAI
        logger.info(f"Generating description for garment {product_id}...")
        description = await generate_garment_description(garment_img)
        garment_descriptions[product_id] = description
        logger.info(f"Generated description for {product_id}: {description}")
    
    total_garments = len(garment_img_dict)
    processed_images = {}
    
    # If more than 2 images, process in batches of 2
    if total_garments > 2:
        logger.info(f"Processing {total_garments} garments in batches of 2")
        
        # Convert dict to list of tuples for easier batching
        garment_items = list(garment_img_dict.items())
        batch_size = 2
        
        # Process in batches
        for batch_num in range(0, total_garments, batch_size):
            batch = garment_items[batch_num:batch_num + batch_size]
            batch_dict = dict(batch)
            batch_product_ids = [product_id for product_id, _ in batch]
            
            logger.info(f"Processing batch {batch_num // batch_size + 1}: {len(batch_dict)} garments (products: {batch_product_ids})")
            
            # Get descriptions for this batch
            batch_descriptions = {product_id: garment_descriptions[product_id] for product_id in batch_dict.keys()}
            
            # Process this batch
            if MODAL_ENDPOINT:
                result_images = await process_tryon_batch_with_modal(
                    person_img, batch_dict, MODAL_ENDPOINT, batch_descriptions
                )
            else:
                logger.warning("Modal endpoint not configured. Using placeholder.")
                # Fallback: return original person image for each product
                result_images = {product_id: person_img.copy() for product_id in batch_dict.keys()}
            
            # Upload batch results to Cloudinary immediately
            for product_id, result_img in result_images.items():
                secure_url = await upload_to_cloudinary(result_img, user_id, product_id)
                processed_images[product_id] = secure_url
                logger.info(f"Uploaded processed image for product: {product_id} (batch {batch_num // batch_size + 1})")
            
            logger.info(f"Completed batch {batch_num // batch_size + 1} of {(total_garments + batch_size - 1) // batch_size}")
    else:
        # Process all at once if 2 or fewer images
        logger.info(f"Processing {total_garments} garments in single batch")
        
        if MODAL_ENDPOINT:
            result_images = await process_tryon_batch_with_modal(
                person_img, garment_img_dict, MODAL_ENDPOINT, garment_descriptions
            )
        else:
            logger.warning("Modal endpoint not configured. Using placeholder.")
            # Fallback: return original person image for each product
            result_images = {product_id: person_img.copy() for product_id in garment_img_dict.keys()}
        
        # Upload all processed images to Cloudinary
        for product_id, result_img in result_images.items():
            secure_url = await upload_to_cloudinary(result_img, user_id, product_id)
            processed_images[product_id] = secure_url
            logger.info(f"Uploaded processed image for product: {product_id}")
    
    logger.info(f"Completed processing for user: {user_id} - {len(processed_images)} images processed")
    return processed_images

