from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import base64
from app.services.image_processor import ImageProcessor
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["upload"])

image_processor = ImageProcessor()

@router.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    """Upload and process a single image"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file content
        content = await file.read()
        
        # Convert to base64
        base64_image = base64.b64encode(content).decode('utf-8')
        
        # Process image
        processed_image = await image_processor.process_image(base64_image)
        
        # Extract metadata
        metadata = await image_processor.extract_image_metadata(base64_image)
        
        return JSONResponse(content={
            "filename": file.filename,
            "content_type": file.content_type,
            "size": len(content),
            "processed_image": processed_image,
            "metadata": metadata
        })
        
    except Exception as e:
        logger.error(f"Error uploading image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload/images")
async def upload_images(files: List[UploadFile] = File(...)):
    """Upload and process multiple images"""
    if len(files) > 4:
        raise HTTPException(status_code=400, detail="Maximum 4 images allowed")
    
    results = []
    for file in files:
        try:
            # Validate file type
            if not file.content_type.startswith('image/'):
                results.append({
                    "filename": file.filename,
                    "error": "File must be an image"
                })
                continue
            
            # Read file content
            content = await file.read()
            
            # Convert to base64
            base64_image = base64.b64encode(content).decode('utf-8')
            
            # Process image
            processed_image = await image_processor.process_image(base64_image)
            
            results.append({
                "filename": file.filename,
                "processed_image": processed_image,
                "size": len(content)
            })
            
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {str(e)}")
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return JSONResponse(content={"images": results})