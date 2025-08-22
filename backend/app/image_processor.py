import base64
import io
from PIL import Image
from typing import Optional, Tuple, Union
import aiohttp
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Service for processing images for multi-modal AI"""
    
    def __init__(self, max_size: Tuple[int, int] = (1024, 1024)):
        self.max_size = max_size
        self.supported_formats = {'JPEG', 'PNG', 'GIF', 'WEBP'}
    
    async def process_image(self, image_data: str) -> str:
        """Process image from URL or base64 string"""
        if image_data.startswith(('http://', 'https://')):
            return await self._process_url_image(image_data)
        else:
            return await self._process_base64_image(image_data)
    
    async def _process_url_image(self, url: str) -> str:
        """Download and process image from URL"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise ValueError(f"Failed to download image: HTTP {response.status}")
                    
                    image_bytes = await response.read()
                    return await self._process_image_bytes(image_bytes)
        except Exception as e:
            logger.error(f"Error processing URL image: {str(e)}")
            raise
    
    async def _process_base64_image(self, base64_data: str) -> str:
        """Process base64 encoded image"""
        try:
            # Remove data URL prefix if present
            if ',' in base64_data:
                base64_data = base64_data.split(',')[1]
            
            image_bytes = base64.b64decode(base64_data)
            return await self._process_image_bytes(image_bytes)
        except Exception as e:
            logger.error(f"Error processing base64 image: {str(e)}")
            raise
    
    async def _process_image_bytes(self, image_bytes: bytes) -> str:
        """Process raw image bytes"""
        try:
            # Open image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode not in ('RGB', 'RGBA'):
                image = image.convert('RGB')
            
            # Check format
            format_name = image.format or 'JPEG'
            if format_name not in self.supported_formats:
                raise ValueError(f"Unsupported image format: {format_name}")
            
            # Resize if necessary
            if image.size[0] > self.max_size[0] or image.size[1] > self.max_size[1]:
                image.thumbnail(self.max_size, Image.Resampling.LANCZOS)
                logger.info(f"Resized image from {image.size} to fit {self.max_size}")
            
            # Convert to base64
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=85, optimize=True)
            buffer.seek(0)
            
            base64_image = base64.b64encode(buffer.read()).decode('utf-8')
            return f"data:image/jpeg;base64,{base64_image}"
            
        except Exception as e:
            logger.error(f"Error processing image bytes: {str(e)}")
            raise
    
    def validate_image_size(self, base64_data: str, max_mb: float = 10) -> bool:
        """Validate image size is within limits"""
        try:
            # Calculate size in MB
            size_bytes = len(base64.b64decode(base64_data.split(',')[1] if ',' in base64_data else base64_data))
            size_mb = size_bytes / (1024 * 1024)
            
            return size_mb <= max_mb
        except:
            return False
    
    async def extract_image_metadata(self, image_data: str) -> dict:
        """Extract metadata from image"""
        try:
            if image_data.startswith(('http://', 'https://')):
                async with aiohttp.ClientSession() as session:
                    async with session.get(image_data) as response:
                        image_bytes = await response.read()
            else:
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
            
            image = Image.open(io.BytesIO(image_bytes))
            
            return {
                "format": image.format,
                "mode": image.mode,
                "size": image.size,
                "width": image.width,
                "height": image.height
            }
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            return {}