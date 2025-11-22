from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Depends, Security
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Dict
import uvicorn
import logging
import os
import httpx
from PIL import Image
from io import BytesIO
from pathlib import Path
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv
import resend

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Create output directory for processed images (temporary, before Cloudinary upload)
OUTPUT_DIR = Path("processed_images")
OUTPUT_DIR.mkdir(exist_ok=True)

# Modal app endpoint
MODAL_ENDPOINT = os.getenv("MODAL_ENDPOINT")

# Frontend URL for email links
FRONTEND_URL = os.getenv("FRONTEND_URL")

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# Configure Resend
RESEND_API_KEY = os.getenv("RESEND_API_KEY")
RESEND_FROM_EMAIL = os.getenv("RESEND_FROM_EMAIL", "onboarding@resend.dev")

# API Key for authentication
API_KEY = os.getenv("API_KEY")
security = HTTPBearer()


class TrialRequest(BaseModel):
    user_id: str
    email: str
    garment_images: Dict[str, str]  # {"product_id": "image_url"}
    person_image: str  # Person image URL


class PremiumRequest(BaseModel):
    user_id: str
    email: str
    garment_images: Dict[str, str]  # {"product_id": "image_url"}
    person_image: str  # Person image URL


async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify API key from Bearer token"""
    if not API_KEY:
        logger.warning("API_KEY not configured. Authentication disabled.")
        return True
    
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True


async def download_image(url: str) -> Image.Image:
    """Download image from URL and return PIL Image"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
            logger.info(f"Downloaded image from: {url}")
            return img
    except Exception as e:
        logger.error(f"Error downloading image from {url}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to download image from URL: {str(e)}")


async def process_tryon_batch_with_modal(person_img: Image.Image, garment_images: Dict[str, Image.Image], endpoint: str) -> Dict[str, Image.Image]:
    """Call Modal batch endpoint to process multiple garments with one person image"""
    try:
        import zipfile
        
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
        
        # Prepare form data (as strings, matching test file format)
        data = {
            "auto_mask": "true",
            "auto_crop": "false",
            "denoise_steps": "30",
            "seed": "42"
        }
        
        logger.info(f"Sending to Modal: 1 human_image + {len(garment_files)} garment_images")
        
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


def get_email_template(user_id: str, email: str, processed_images: Dict[str, str], plan_type: str, frontend_url: str) -> str:
    """Generate HTML email template for try-on completion matching website theme"""
    
    # Generate product links/buttons
    products_html = ""
    for product_id, image_url in processed_images.items():
        products_html += f"""
        <div style="margin: 15px 0; padding: 20px; background: #fff; border-radius: 8px; border: 1px solid #f0f0f0;">
            <p style="margin: 0 0 10px 0; color: #333; font-size: 16px; font-weight: 600;">Product: {product_id}</p>
            <a href="{image_url}" target="_blank" style="display: inline-block; padding: 12px 24px; background: linear-gradient(135deg, #ff6b9d 0%, #ff4757 100%); color: white; text-decoration: none; border-radius: 6px; font-weight: 600; font-size: 14px;">View Try-On Result</a>
        </div>
        """
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Virtual Try-On Complete - DRAPELY.ai</title>
    </head>
    <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; line-height: 1.6; color: #333; margin: 0; padding: 0; background-color: #fafafa;">
        <table width="100%" cellpadding="0" cellspacing="0" style="background-color: #fafafa; padding: 40px 20px;">
            <tr>
                <td align="center">
                    <table width="600" cellpadding="0" cellspacing="0" style="background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.08);">
                        <!-- Header with Logo -->
                        <tr>
                            <td style="background: linear-gradient(135deg, #fff5f7 0%, #ffeef2 100%); padding: 40px 30px; text-align: center; border-bottom: 2px solid #ff6b9d;">
                                <div style="margin-bottom: 20px;">
                                    <img src="https://res.cloudinary.com/dnkrqpuqk/image/upload/v1/logo2.2k" alt="DRAPELY.ai Logo" style="max-width: 180px; height: auto; display: block; margin: 0 auto;" />
                                </div>
                                <div style="display: inline-block; background: #ff6b9d; color: white; padding: 6px 16px; border-radius: 20px; font-size: 12px; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 10px;">
                                    // FUTURISTIC
                                </div>
                                <h1 style="color: #1a1a2e; margin: 10px 0; font-size: 32px; font-weight: 700; letter-spacing: -0.5px;">
                                    Where AI Meets Style
                                </h1>
                            </td>
                        </tr>
                        
                        <!-- Main Content -->
                        <tr>
                            <td style="padding: 40px 30px;">
                                <p style="font-size: 18px; color: #1a1a2e; margin: 0 0 20px 0; font-weight: 600;">
                                    ✨ Your Virtual Try-On is Complete!
                                </p>
                                
                                <p style="font-size: 16px; color: #555; margin: 0 0 25px 0; line-height: 1.8;">
                                    Fashion that learns you. We've successfully processed <strong style="color: #ff6b9d;">{len(processed_images)}</strong> garment(s) for your personalized try-on experience.
                                </p>
                                
                                <!-- Plan Badge -->
                                <div style="background: linear-gradient(135deg, #ffeef2 0%, #fff5f7 100%); padding: 12px 20px; border-radius: 8px; margin: 25px 0; border-left: 4px solid #ff6b9d;">
                                    <p style="margin: 0; color: #1a1a2e; font-size: 14px; font-weight: 600;">
                                        Plan: <span style="color: #ff6b9d;">{plan_type.title()}</span> • Total Processed: <span style="color: #ff6b9d;">{len(processed_images)}</span>
                                    </p>
                                </div>
                                
                                <!-- Try-On Results Section -->
                                <div style="margin: 30px 0;">
                                    <h2 style="color: #1a1a2e; font-size: 22px; font-weight: 700; margin: 0 0 20px 0; padding-bottom: 10px; border-bottom: 2px solid #ffeef2;">
                                        Your Try-On Results
                                    </h2>
                                    {products_html}
                                </div>
                                
                                <!-- CTA Button -->
                                <div style="text-align: center; margin: 40px 0 30px 0;">
                                    <a href="{FRONTEND_URL}/products" style="display: inline-block; padding: 16px 40px; background: linear-gradient(135deg, #ff6b9d 0%, #ff4757 100%); color: white; text-decoration: none; border-radius: 8px; font-weight: 700; font-size: 16px; box-shadow: 0 4px 15px rgba(255, 107, 157, 0.3); transition: transform 0.2s;">
                                        View All Products
                                    </a>
                                </div>
                                
                                <!-- Footer -->
                                <div style="margin-top: 40px; padding-top: 30px; border-top: 1px solid #f0f0f0; text-align: center;">
                                    <p style="color: #888; font-size: 14px; margin: 0 0 10px 0;">
                                        Thank you for using <strong style="color: #ff6b9d;">DRAPELY.ai</strong> Virtual Try-On service!
                                    </p>
                                    <p style="color: #aaa; font-size: 12px; margin: 0;">
                                        If you have any questions, please don't hesitate to contact us.
                                    </p>
                                </div>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
    </body>
    </html>
    """
    return html


def send_completion_email_sync(email: str, user_id: str, processed_images: Dict[str, str], plan_type: str):
    """Send email notification when try-on processing is complete (synchronous for background task)"""
    if not RESEND_API_KEY:
        logger.warning("RESEND_API_KEY not configured. Skipping email notification.")
        return
    
    try:
        # Set API key (per Resend docs)
        resend.api_key = RESEND_API_KEY
        
        subject = f"Your Virtual Try-On Results Are Ready! ({plan_type.title()} Plan)"
        html_content = get_email_template(user_id, email, processed_images, plan_type, FRONTEND_URL)
        
        # Send email via Resend (per FastAPI docs)
        params: resend.Emails.SendParams = {
            "from": RESEND_FROM_EMAIL,
            "to": [email],
            "subject": subject,
            "html": html_content,
        }
        
        email_response = resend.Emails.send(params)
        logger.info(f"Email sent successfully to {email}. Email ID: {email_response.get('id')}")
        
    except Exception as e:
        logger.error(f"Failed to send email to {email}: {str(e)}")
        # Don't raise exception - email failure shouldn't break the API response


def send_error_email_sync(email: str, user_id: str, error_message: str, plan_type: str, frontend_url: str):
    """Send error email notification when try-on processing fails"""
    if not RESEND_API_KEY:
        logger.warning("RESEND_API_KEY not configured. Skipping error email notification.")
        return
    
    try:
        # Set API key (per Resend docs)
        resend.api_key = RESEND_API_KEY
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Virtual Try-On Processing Error</title>
        </head>
        <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; line-height: 1.6; color: #333; margin: 0; padding: 0; background-color: #fafafa;">
        <table width="100%" cellpadding="0" cellspacing="0" style="background-color: #fafafa; padding: 40px 20px;">
            <tr>
                <td align="center">
                    <table width="600" cellpadding="0" cellspacing="0" style="background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.08);">
                        <!-- Header with Logo -->
                        <tr>
                            <td style="background: linear-gradient(135deg, #fff5f7 0%, #ffeef2 100%); padding: 40px 30px; text-align: center; border-bottom: 2px solid #ff6b9d;">
                                <div style="margin-bottom: 20px;">
                                    <img src="https://res.cloudinary.com/dnkrqpuqk/image/upload/v1/logo2.2k" alt="DRAPELY.ai Logo" style="max-width: 180px; height: auto; display: block; margin: 0 auto;" />
                                </div>
                                <h1 style="color: #1a1a2e; margin: 10px 0; font-size: 28px; font-weight: 700;">⚠️ Processing Error</h1>
                            </td>
                        </tr>
                        
                        <!-- Main Content -->
                        <tr>
                            <td style="padding: 40px 30px;">
                
                <p style="font-size: 16px; color: #555;">Hello,</p>
                
                <p style="font-size: 16px; color: #555;">
                    We encountered an error while processing your virtual try-on request. 
                    Please try again or contact support if the issue persists.
                </p>
                
                <div style="background: #fff5f7; padding: 15px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #ff6b9d;">
                    <p style="margin: 0; color: #1a1a2e;">
                        <strong>Plan:</strong> <span style="color: #ff6b9d;">{plan_type.title()}</span><br>
                        <strong>Error:</strong> {error_message}
                    </p>
                </div>
                
                                <!-- CTA Button -->
                                <div style="text-align: center; margin: 30px 0;">
                                    <a href="{frontend_url}/products" style="display: inline-block; padding: 16px 40px; background: linear-gradient(135deg, #ff6b9d 0%, #ff4757 100%); color: white; text-decoration: none; border-radius: 8px; font-weight: 700; font-size: 16px; box-shadow: 0 4px 15px rgba(255, 107, 157, 0.3);">
                                        Try Again
                                    </a>
                                </div>
                                
                                <!-- Footer -->
                                <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #f0f0f0; text-align: center;">
                                    <p style="color: #888; font-size: 14px; margin: 0 0 10px 0;">
                                        Please try submitting your request again.
                                    </p>
                                    <p style="color: #aaa; font-size: 12px; margin: 0;">
                                        If the problem continues, please contact our support team.
                                    </p>
                                </div>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
        </body>
        </html>
        """
        
        params: resend.Emails.SendParams = {
            "from": RESEND_FROM_EMAIL,
            "to": [email],
            "subject": f"Virtual Try-On Processing Error ({plan_type.title()} Plan)",
            "html": html_content,
        }
        
        email_response = resend.Emails.send(params)
        logger.info(f"Error email sent successfully to {email}. Email ID: {email_response.get('id')}")
        
    except Exception as e:
        logger.error(f"Failed to send error email to {email}: {str(e)}")


async def upload_to_cloudinary(image: Image.Image, user_id: str, product_id: str) -> str:
    """Upload processed image to Cloudinary and return secure URL"""
    try:
        # Save image to temporary buffer
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        
        # Upload to Cloudinary
        # public_id format: {product_id}_{user_id}
        public_id = f"{product_id}_{user_id}"
        
        logger.info(f"Uploading to Cloudinary: public_id={public_id}")
        response = cloudinary.uploader.upload(
            buffer,
            folder="ecommerce-products/users",
            public_id=public_id,
            resource_type="image",
            format="png"
        )
        
        secure_url = response["secure_url"]
        logger.info(f"Uploaded to Cloudinary: {secure_url}")
        return secure_url
    except Exception as e:
        logger.error(f"Error uploading to Cloudinary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload image to Cloudinary: {str(e)}")


async def process_and_send_email(user_id: str, email: str, garment_images: Dict[str, str], person_image: str, plan_type: str):
    """Process try-on and send email (runs in background)"""
    try:
        processed_images = await process_virtual_tryon(user_id, garment_images, person_image)
        logger.info(f"Processing completed for user: {user_id}")
        
        # Send completion email
        send_completion_email_sync(email, user_id, processed_images, plan_type)
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error processing try-on for user {user_id}: {error_message}")
        
        # Send error email
        send_error_email_sync(email, user_id, error_message, plan_type, FRONTEND_URL)


async def process_virtual_tryon(user_id: str, garment_images: Dict[str, str], person_image: str) -> Dict[str, str]:
    """Download images, process try-on using batch endpoint, and save results"""
    logger.info(f"Starting try-on processing for user: {user_id}")
    
    # Download person image
    logger.info(f"Downloading person image from: {person_image}")
    person_img = await download_image(person_image)
    
    # Download all garment images
    garment_img_dict = {}
    for product_id, garment_url in garment_images.items():
        logger.info(f"Downloading garment {product_id} from: {garment_url}")
        garment_img = await download_image(garment_url)
        garment_img_dict[product_id] = garment_img
    
    # Process all garments in batch using Modal
    if MODAL_ENDPOINT:
        logger.info(f"Processing {len(garment_img_dict)} garments in batch via Modal")
        result_images = await process_tryon_batch_with_modal(person_img, garment_img_dict, MODAL_ENDPOINT)
    else:
        logger.warning("Modal endpoint not configured. Using placeholder.")
        # Fallback: return original person image for each product
        result_images = {product_id: person_img.copy() for product_id in garment_img_dict.keys()}
    
    # Upload all processed images to Cloudinary
    processed_images = {}
    for product_id, result_img in result_images.items():
        secure_url = await upload_to_cloudinary(result_img, user_id, product_id)
        processed_images[product_id] = secure_url
        logger.info(f"Uploaded processed image for product: {product_id}")
    
    logger.info(f"Completed batch processing for user: {user_id}")
    return processed_images


@app.post("/api/v1/trial")
async def trial_virtual_tryon(
    request: TrialRequest, 
    background_tasks: BackgroundTasks,
    verified: bool = Depends(verify_api_key)
):
    """Trial endpoint - max 2-3 images"""
    # Log request details
    logger.info(f"TRIAL ENDPOINT HIT")
    logger.info(f"User ID: {request.user_id}")
    logger.info(f"User Email: {request.email}")
    logger.info(f"Person Image URL: {request.person_image}")
    logger.info(f"Garment Images ({len(request.garment_images)}):")
    for product_id, image_url in request.garment_images.items():
        logger.info(f"  - Product ID: {product_id}, Image URL: {image_url}")
    
    total_images = len(request.garment_images) + 1  # +1 for person_image
    
    if total_images > 3:
        raise HTTPException(status_code=400, detail="Trial plan: max 3 total images allowed (garment + person)")
    
    # Return success immediately (fire and forget)
    # Processing will happen in background and email will be sent when done
    background_tasks.add_task(
        process_and_send_email,
        request.user_id,
        request.email,
        request.garment_images,
        request.person_image,
        "trial"
    )
    
    return {
        "success": True,
        "message": "Request received. Processing in background. You will receive an email with results shortly.",
        "user_id": request.user_id
    }


@app.post("/api/v1/premium")
async def premium_virtual_tryon(
    request: PremiumRequest, 
    background_tasks: BackgroundTasks,
    verified: bool = Depends(verify_api_key)
):
    """Premium endpoint - more images allowed"""
    # Log request details
    logger.info(f"PREMIUM ENDPOINT HIT")
    logger.info(f"User ID: {request.user_id}")
    logger.info(f"User Email: {request.email}")
    logger.info(f"Person Image URL: {request.person_image}")
    logger.info(f"Garment Images ({len(request.garment_images)}):")
    for product_id, image_url in request.garment_images.items():
        logger.info(f"  - Product ID: {product_id}, Image URL: {image_url}")
    
    # Return success immediately (fire and forget)
    # Processing will happen in background and email will be sent when done
    background_tasks.add_task(
        process_and_send_email,
        request.user_id,
        request.email,
        request.garment_images,
        request.person_image,
        "premium"
    )
    
    return {
        "success": True,
        "message": "Request received. Processing in background. You will receive an email with results shortly.",
        "user_id": request.user_id
    }


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": str(exc.body)}
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

