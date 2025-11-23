"""Modal API service for virtual try-on processing"""
import httpx
import zipfile
import random
from PIL import Image
from io import BytesIO
from typing import Dict
from fastapi import HTTPException
import logging
from config import MODAL_ENDPOINT

logger = logging.getLogger(__name__)


async def process_tryon_batch_with_modal(
    person_img: Image.Image, 
    garment_images: Dict[str, Image.Image], 
    endpoint: str,
    garment_descriptions: Dict[str, str] = None
) -> Dict[str, Image.Image]:
    """Call Modal batch endpoint to process multiple garments with one person image"""
    try:
        # Convert person image to bytes (read content, not buffer object)
        person_buffer = BytesIO()
        person_img.save(person_buffer, format="PNG")
        person_bytes = person_buffer.getvalue()
        
        # Convert all garment images to bytes
        garment_files = []
        for product_id, garment_img in garment_images.items():
            garment_buffer = BytesIO()
            garment_img.save(garment_buffer, format="PNG")
            garment_bytes = garment_buffer.getvalue()
            garment_files.append((product_id, garment_bytes))
        
        # Prepare files for multipart form data (matching test_tryon_api.py format)
        # httpx needs tuple format: (field_name, (filename, file_content, content_type))
        # IMPORTANT: human_image first, then garment_images (same order as test file)
        files = [("human_image", ("person.png", person_bytes, "image/png"))]
        logger.info(f"Prepared human_image (person/avatar): {len(person_bytes)} bytes")
        
        for product_id, garment_bytes in garment_files:
            files.append(("garment_images", (f"{product_id}.png", garment_bytes, "image/png")))
            logger.info(f"Prepared garment_image for product {product_id}: {len(garment_bytes)} bytes")
        
        # Generate random seed for each request (ensures unique results)
        random_seed = random.randint(0, 2**31 - 1)
        
        # Prepare form data (as strings, matching test file format)
        data = {
            "auto_mask": "true",
            "auto_crop": "false",
            "denoise_steps": "30",
            "seed": str(random_seed)
        }
        logger.info(f"Request parameters: seed={data['seed']} (random), denoise_steps={data['denoise_steps']}, auto_mask={data['auto_mask']}, auto_crop={data['auto_crop']}")
        
        # Add garment descriptions if provided
        if garment_descriptions:
            # Convert descriptions dict to JSON array format (matching modal_deploy.py format)
            import json
            # Create descriptions list in the same order as garment_files
            descriptions_list = []
            product_id_order = []
            for product_id, _ in garment_files:
                product_id_order.append(product_id)
                if product_id in garment_descriptions:
                    descriptions_list.append(garment_descriptions[product_id])
                    logger.info(f"Mapped description for product {product_id}: {garment_descriptions[product_id][:80]}..." if len(garment_descriptions[product_id]) > 80 else f"Mapped description for product {product_id}: {garment_descriptions[product_id]}")
                else:
                    # Fallback to default if description missing
                    default_desc = "a beautiful garment, professional fashion photography, high quality"
                    descriptions_list.append(default_desc)
                    logger.warning(f"No description found for product {product_id}, using default")
            
            # Send as JSON array string (modal_deploy.py expects this format)
            data["garment_descriptions"] = json.dumps(descriptions_list)
            logger.info(f"Added {len(descriptions_list)} garment descriptions to request in order: {product_id_order}")
            for idx, (product_id, desc) in enumerate(zip(product_id_order, descriptions_list)):
                logger.info(f"  [{idx + 1}] Product {product_id}: {desc[:100]}..." if len(desc) > 100 else f"  [{idx + 1}] Product {product_id}: {desc}")
        
        logger.info(f"Sending to Modal: 1 human_image + {len(garment_files)} garment_images")
        logger.info(f"Request data being sent: {data}")
        
        # Make request to Modal batch endpoint (matching test file timeout)
        async with httpx.AsyncClient(timeout=1200.0) as client:  # 20 minutes like test file
            logger.info(f"Calling Modal batch endpoint: {endpoint}/tryon/batch")
            response = await client.post(f"{endpoint}/tryon/batch", files=files, data=data)
            response.raise_for_status()
            
            logger.info(f"Received response from Modal, extracting ZIP file")
            # Extract zip file
            zip_buffer = BytesIO(response.content)
            result_images = {}
            
            with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
                # Map each file in zip to product_id
                file_list = zip_file.namelist()
                logger.info(f"ZIP contains {len(file_list)} files: {file_list}")
                for idx, (product_id, _) in enumerate(garment_files):
                    # Find corresponding output file (usually output_1_*.png, output_2_*.png, etc.)
                    matching_files = [f for f in file_list if f.startswith(f"output_{idx + 1}_") and f.endswith(".png")]
                    if matching_files:
                        img_data = zip_file.read(matching_files[0])
                        result_images[product_id] = Image.open(BytesIO(img_data)).convert("RGB")
                        logger.info(f"Extracted image for product {product_id} from {matching_files[0]}")
                    else:
                        logger.warning(f"Could not find output image for product {product_id} (index {idx + 1})")
            
            return result_images
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error from Modal: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Modal API error: {e.response.text}")
    except Exception as e:
        logger.error(f"HTTP call to Modal batch endpoint failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Try-on processing failed: {str(e)}")

