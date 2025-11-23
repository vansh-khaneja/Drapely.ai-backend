"""FastAPI application entry point"""
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import uvicorn
import logging
from pathlib import Path
import traceback

# Import schemas
from schemas import TryOnRequest

# Import utilities
from utils.auth import verify_api_key

# Import actions
from actions.tryon_actions import process_and_send_email

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


@app.post("/api/v1/tryon")
async def virtual_tryon(
    request: TryOnRequest, 
    background_tasks: BackgroundTasks,
    verified: bool = Depends(verify_api_key)
):
    """Unified try-on endpoint - accepts subscription type and collection"""
    # Log request details
    logger.info(f"TRY-ON ENDPOINT HIT")
    logger.info(f"User ID: {request.user_id}")
    logger.info(f"User Email: {request.email}")
    logger.info(f"Subscription Type: {request.subscription_type}")
    logger.info(f"Collection: {request.collection}")
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
        request.subscription_type,
        request.collection
    )
    
    return {
        "success": True,
        "message": "Request received. Processing in background. You will receive an email with results shortly.",
        "user_id": request.user_id,
        "subscription_type": request.subscription_type,
        "collection": request.collection
    }


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": str(exc.body)}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle any unhandled exceptions"""
    error_traceback = traceback.format_exc()
    logger.error(f"Unhandled exception: {str(exc)}\n{error_traceback}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
