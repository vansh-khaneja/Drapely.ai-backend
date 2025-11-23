"""Pydantic models for request/response"""
from pydantic import BaseModel, Field
from typing import Dict, Literal


class TryOnRequest(BaseModel):
    user_id: str
    email: str
    garment_images: Dict[str, str]  # {"product_id": "image_url"}
    person_image: str  # Person image URL
    subscription_type: Literal["trial", "premium"] = Field(..., description="Subscription type: trial or premium")
    collection: str = Field(..., description="Collection name to process and mention in email")

