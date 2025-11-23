"""Service for generating garment descriptions using OpenAI"""
from openai import OpenAI
from PIL import Image
from io import BytesIO
import base64
import logging
from config import OPENAI_API_KEY

logger = logging.getLogger(__name__)


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")


async def generate_garment_description(image: Image.Image) -> str:
    """Generate detailed garment description using OpenAI Vision API"""
    try:
        if not OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY not configured. Using default description.")
            return "a beautiful garment, professional fashion photography, high quality"
        
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Convert image to base64
        base64_image = image_to_base64(image)
        
        # Create prompt for concise garment description (must be under 50 words to fit CLIP's 77 token limit)
        prompt = """Analyze this garment image and provide a concise, natural language description for virtual try-on generation.
        
        Describe in 2-3 sentences: garment type, color, style, material/texture, and key features (collar, sleeves, pattern, fit).
        
        IMPORTANT: 
        - Use plain text only (NO markdown, NO bullet points, NO formatting)
        - Keep it under 50 words
        - Write as a natural sentence, not a list
        - Focus on visual details that matter for try-on
        
        Example format: "A gray cable knit sweater with a turtleneck collar, loose fit, chunky knit texture, and ribbed cuffs." """
        
        # Use standard OpenAI Chat Completions API with vision
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    },
                ],
            }],
            max_tokens=100  # Reduced from 300 to encourage conciseness
        )
        
        description = response.choices[0].message.content.strip()
        
        # Clean up markdown formatting if present
        import re
        # Remove markdown bold/italic
        description = re.sub(r'\*\*([^*]+)\*\*', r'\1', description)
        description = re.sub(r'\*([^*]+)\*', r'\1', description)
        # Remove markdown headers
        description = re.sub(r'^#+\s*', '', description, flags=re.MULTILINE)
        # Remove bullet points and dashes at start of lines
        description = re.sub(r'^[\-\*]\s*', '', description, flags=re.MULTILINE)
        # Remove extra whitespace
        description = ' '.join(description.split())
        
        logger.info(f"Generated garment description: {description}")
        print(f"Generated garment description: {description}")  # Print as requested
        
        # Return clean description without redundant prefixes (prompt already adds "a beautiful female model wearing")
        # Just add quality descriptors at the end
        enhanced_description = f"{description}, professional fashion photography, high quality, detailed texture"
        
        return enhanced_description
        
    except Exception as e:
        logger.error(f"Error generating garment description: {str(e)}")
        # Fallback to default description
        default_description = "a beautiful garment, professional fashion photography, high quality"
        logger.warning(f"Using default description: {default_description}")
        print(f"Error generating description, using default: {default_description}")
        return default_description

