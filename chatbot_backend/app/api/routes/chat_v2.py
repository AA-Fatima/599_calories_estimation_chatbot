from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# These will be initialized in main.py
app_state: Dict = {}

class ChatRequest(BaseModel):
    message: str
    country: str = "lebanon"

class IngredientResponse(BaseModel):
    usda_fdc_id: Optional[int] = 0
    name: str
    weight_g: float
    calories: float
    protein_g: Optional[float] = 0
    carbs_g: Optional[float] = 0
    fats_g: Optional[float] = 0

class ChatResponse(BaseModel):
    food_item: str
    total_calories: float
    total_weight_g: float
    protein_g: float
    carbs_g: float
    fats_g: float
    ingredients: List[Dict]
    found: bool
    source: Optional[str] = None

def get_gpt_parser():
    return app_state.get("gpt_parser")

def get_calculator_v2():
    return app_state.get("calculator_v2")

def get_missing_logger():
    return app_state.get("missing_logger")

@router.post("/message", response_model=ChatResponse)
async def chat_message(request: ChatRequest):
    """Main chat endpoint using GPT parser and v2 calculator"""
    
    try:
        gpt_parser = get_gpt_parser()
        calculator = get_calculator_v2()
        missing_logger = get_missing_logger()
        
        if not gpt_parser or not calculator:
            logger.error("Services not initialized")
            raise HTTPException(status_code=500, detail="Services not initialized")
        
        # Parse with GPT
        logger.info(f"Processing message: {request.message}")
        gpt_result = await gpt_parser.parse_query(request.message, request.country)
        logger.info(f"GPT parsed result: {gpt_result}")
        
        # Calculate calories
        result = calculator.calculate(gpt_result)
        logger.info(f"Calculation result: {result.get('food_item')} = {result.get('total_calories')} kcal")
        
        # Log if not found
        if not result['found'] and missing_logger:
            missing_logger.log(
                dish_name=result['food_item'],
                country=request.country,
                user_query=request.message,
                gpt_result=gpt_result
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
