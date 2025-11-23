"""Configuration and environment variables"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Modal app endpoint
MODAL_ENDPOINT = os.getenv("MODAL_ENDPOINT")

# Frontend URL for email links
FRONTEND_URL = os.getenv("FRONTEND_URL")

# Cloudinary configuration
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")

# Resend configuration
RESEND_API_KEY = os.getenv("RESEND_API_KEY")
RESEND_FROM_EMAIL = os.getenv("RESEND_FROM_EMAIL", "onboarding@resend.dev")

# API Key for authentication
API_KEY = os.getenv("API_KEY")

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Logo URL
LOGO_URL = "https://res.cloudinary.com/dnkrqpuqk/image/upload/v1763804717/logo2.2k_orepqx.png"

