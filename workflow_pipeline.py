from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, List
import logging
import traceback
import sys
import asyncio
import os
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
from AI_Model.src.utils.exceptions import CustomException



class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., min_length=1, max_length=5000, description="User message")

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    reply: str
    status: str = "success"

class MessageResponse(BaseModel):
    """Generic message response"""
    message: str
    status: str = "success"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
workflow = None
pipeline = None
vision_pipeline = None

class AudioPipeline:
    """Manages the audio workflow execution"""  
    def __init__(self):
        from AI_Model.audio_model.workflow.graph_builder import MultiGraphWorkflow
        self.audio_workflow_instance = MultiGraphWorkflow()
    
    def process_audio(self, audio_files: List[UploadFile]) -> str:
        """
        Docstring for process_audio
        
        """
        temp_audio_paths: List[str] = []
        try:
            if not audio_files:
                raise ValueError("No audio files provided.")

            for audio in audio_files:
                audio.file.seek(0)
                suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(audio.file.read())
                    temp_audio_paths.append(tmp_file.name)

            initial_state = {
                "audio_file": temp_audio_paths,
            }

            logger.info(f"Processing audio files: {[file.filename for file in audio_files]}...")
            result = self.audio_workflow_instance.invoke(initial_state)
            # Extract response from result
            bot_response = self._extract_response(result)
            #logger.info("✓ Audio processed successfully")
            return str(bot_response)
        
        except Exception as e:
            error_msg = f"Error processing audio: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise CustomException(e, sys)
        finally:
            for path in temp_audio_paths:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass
    def _extract_response(self, result: dict) -> Optional[str]:
        """
        Extract the final response from workflow result
        
        Args:
            result: The output from workflow.invoke()"""
        if isinstance(result, dict):
            # Try different possible keys where response might be stored
            possible_keys = ["final_response", "final_output"]
             
            for key in possible_keys:
                if key in result and result[key]:
                    value = result[key]
                    # If it's a list of messages, get the last one
                    if isinstance(value, list) and len(value) > 0:
                        last_msg = value[-1]
                        if isinstance(last_msg, dict) and "content" in last_msg:
                            return last_msg["content"]
                        return str(last_msg)
                    return str(value)

        # Fallback
        return None

class VisionPipeline:
    """Manages the vision workflow execution"""

    def __init__(self):
        from AI_Model.vision_model.workflow.graph_builder_vision import MultiGraphWorkflow 
        self.vision_workflow_instance = MultiGraphWorkflow()
        
    
    def process_images(self, images: List[Image.Image], query: str = '') -> str:
        """
        Docstring for process_images
        
        """
        try:
            initial_state = {
                'image' : images,
                'query': query,
            }
            
            logger.info(f"Processing image with query: {query}...")
            result = self.vision_workflow_instance.invoke(initial_state)
            # Extract response from result
            bot_response = self._extract_response(result)
            #logger.info("✓ Image processed successfully")
            return str(bot_response)
        
        except Exception as e:
            error_msg = f"Error processing image: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise CustomException(e, sys)
        
    def _extract_response(self, result: dict) -> Optional[str]:
        """
        Extract the final response from workflow result
        
        Args:
            result: The output from workflow.invoke()
            
        Returns:
            The bot's response string
        """
        if isinstance(result, dict):
            # Try different possible keys where response might be stored
            possible_keys = ["final_response", "final_output"]
             
            for key in possible_keys:
                if key in result and result[key]:
                    value = result[key]
                    # If it's a list of messages, get the last one
                    if isinstance(value, list) and len(value) > 0:
                        last_msg = value[-1]
                        if isinstance(last_msg, dict) and "content" in last_msg:
                            return last_msg["content"]
                        return str(last_msg)
                    return str(value)

        # Fallback
        return None

class ChatbotPipeline:
    """Manages the chatbot workflow execution"""
    
    def __init__(self, compiled_workflow):
        self.workflow = compiled_workflow
        self.conversation_history: List[dict] = []

    
    def process_message(self, user_input: str) -> Optional[str]:
        """
        Process user message through the LangGraph workflow
        
        Args:
            user_input: User's message
            
        Returns:
            Bot response or error message
        """
        try:
            initial_state = {
                "query": user_input,
                "conversation_history": self.conversation_history,
                "use_rag": True,  
                "messages": [],
                "current_step": "input_processing"
            }
            
            logger.info(f"Processing input: {user_input[:50]}...")
            result = self.workflow.invoke(initial_state)
            
            # Extract response from result
            bot_response = self._extract_response(result)
            
            # Update conversation history
            self.conversation_history.append({
                "query": user_input,
                "bot": bot_response
            })
            
            logger.info("✓ Message processed successfully")
            return bot_response
            
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise CustomException(e, sys)
    
    def _extract_response(self, result: dict) -> str:
        """
        Extract the final response from workflow result
        
        Args:
            result: The output from workflow.invoke()
            
        Returns:
            The bot's response string
        """
        if isinstance(result, dict):
            # Try different possible keys where response might be stored
            possible_keys = ["final_response", "validated_response"]
            
            for key in possible_keys:
                if key in result and result[key]:
                    value = result[key]
                    # If it's a list of messages, get the last one
                    if isinstance(value, list) and len(value) > 0:
                        last_msg = value[-1]
                        if isinstance(last_msg, dict) and "content" in last_msg:
                            return last_msg["content"]
                        return str(last_msg)
                    return str(value)
        
        # Fallback
        return "No response generated"
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []
        logger.info("Conversation history reset")
        return 
    def get_history(self) -> List[dict]:
        """Get conversation history"""
        return self.conversation_history

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages startup and shutdown events
    """
    # Startup
    global workflow, pipeline, vision_pipeline
    
    logger.info("=" * 50)
    logger.info("Starting AI Chatbot Server")
    logger.info("=" * 50)
    
    try:
        from AI_Model.src.workflow.graph_builder import build_complete_workflow
        workflow = build_complete_workflow()
        pipeline = ChatbotPipeline(workflow)
        vision_pipeline = VisionPipeline()
        logger.info("✓ Workflow compiled successfully")
    except Exception as e:
        logger.error(f"✗ Error compiling workflow: {e}")
        logger.error(traceback.format_exc())
        workflow = None
        pipeline = None
        vision_pipeline = None
  
    if workflow is None:
        logger.error("CRITICAL: Workflow failed to compile!")
        logger.error("Check your graph_builder.py and node definitions")
    else:
        logger.info("✓ Workflow ready")
        logger.info("Available endpoints:")
        logger.info("  POST   /api/chat      - Send message")
        logger.info("=" * 50)
    
    try:
        yield
    except asyncio.CancelledError:
        logger.info("Lifespan cancelled during shutdown")
        raise
    finally:
        logger.info("Shutting down server...")

app = FastAPI(
    title="AI Chatbot API",
    description="FastAPI backend for AI Chatbot with LangGraph workflow",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/upload-audio/")
async def upload_audio(files: List[UploadFile] = File(default=[])):
    """
    Upload an audio file.
    Includes validation for content type and file size.
    """
    audio_pipeline = None
    if not files:
        raise HTTPException(status_code=400, detail="At least one audio file is required")

    # 1. Validate Content Type
    allowed_types = ["audio/mpeg", "audio/wav", "audio/ogg", "audio/x-wav", "audio/mp3"]
    for file in files:
        if file.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}. Allowed: {allowed_types}")
    
        # 2. Validate filename
        if file.filename is None:
            raise HTTPException(status_code=400, detail="Filename is required")
    
    if audio_pipeline is None:
        audio_pipeline = AudioPipeline()
    response = audio_pipeline.process_audio(files)

    return {
        "filenames": [uploaded_file.filename for uploaded_file in files],
        "response": response
           }

@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a video file.
    Similar to audio, but with different mime types and size limits.
    """
    # 1. Validate Content Type
    allowed_types = ["video/mp4", "video/mpeg", "video/quicktime", "video/x-msvideo"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Invalid video type: {file.content_type}")

    # 2. Validate filename
    if file.filename is None:
        raise HTTPException(status_code=400, detail="Filename is required")

    return {
        "message": "Video uploaded successfully",
        "filename": file.filename,
        "size": file.size
    }

@app.post("/upload-photo/")
async def upload_photo(files: List[UploadFile] = File(default=[]), query: str = ''):
    """
    Upload multiple photos and perform processing on each.
    """
    results = []
    global vision_pipeline
    if vision_pipeline is None:
        vision_pipeline = VisionPipeline()
    images = []
    for file in files:
        contents = await file.read()
        images.append(Image.open(io.BytesIO(contents)))
        
    result = vision_pipeline.process_images(images, query)
        
    results.append({
        "filename": file.filename,
        "result": result
    })

    return results
    

@app.post(
    "/api/chat",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="Send a message to the chatbot",
    responses={
        200: {"description": "Message processed successfully"},
        400: {"description": "Invalid request (empty message)"},
        500: {"description": "Server error"}
    }
)
async def chat(request: ChatRequest):
    """
    Main chat endpoint - sends user message to chatbot
    
    Args:
        request: ChatRequest containing the user message
        
    Returns:
        ChatResponse: Bot's reply and status
        
    Raises:
        HTTPException: If message is empty or workflow not initialized
    """
    try:
        user_message = request.message.strip()
        
        if not user_message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Please enter a message"
            )
        
        if workflow is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Workflow not initialized. Check server logs."
            )
        
        # Process through pipeline
        response = pipeline.process_message(user_message) #type: ignore
        
        return ChatResponse(reply=response  , status="success")#type: ignore
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Server error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn

    try:
        uvicorn.run(
            "workflow_pipeline:app",  # Change "main" to your file name if different
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
