from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging
import json
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

router = APIRouter()

# These will be initialized in main.py
app_state = {}

class Dish(BaseModel):
    id: Optional[int] = None
    dish_name: str
    country: str
    total_calories: Optional[float] = 0
    weight_g: Optional[float] = 0
    ingredients: Optional[List] = []

class MissingDish(BaseModel):
    timestamp: str
    dish_name: str
    user_query: str
    country: str
    gpt_ingredients: str

def get_missing_logger():
    return app_state.get("missing_logger")

@router.get("/dishes", response_model=List[Dish])
async def get_dishes():
    """Get all dishes"""
    try:
        # Load dishes from Excel
        dishes_path = "app/data/raw/dishes.xlsx"
        
        if not Path(dishes_path).exists():
            return []
        
        df = pd.read_excel(dishes_path)
        dishes = []
        
        for idx, row in df.iterrows():
            dishes.append({
                "id": idx,
                "dish_name": str(row.get('dish name', row.get('dish_name', ''))),
                "country": str(row.get('Country', row.get('country', ''))),
                "total_calories": float(row.get('calories', 0)) if not pd.isna(row.get('calories', 0)) else 0,
                "weight_g": float(row.get('weight (g)', row.get('weight_g', 0))) if not pd.isna(row.get('weight (g)', row.get('weight_g', 0))) else 0,
                "ingredients": []
            })
        
        return dishes
    
    except Exception as e:
        logger.error(f"Error loading dishes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/missing-dishes", response_model=List[MissingDish])
async def get_missing_dishes():
    """Get all missing dishes"""
    try:
        missing_logger = get_missing_logger()
        if not missing_logger:
            return []
        
        missing = missing_logger.get_missing_dishes()
        return missing
    
    except Exception as e:
        logger.error(f"Error loading missing dishes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/dishes", response_model=Dish)
async def add_dish(dish: Dish):
    """Add a new dish"""
    try:
        dishes_path = "app/data/raw/dishes.xlsx"
        
        # Load existing data
        if Path(dishes_path).exists():
            df = pd.read_excel(dishes_path)
        else:
            df = pd.DataFrame()
        
        # Add new dish
        new_row = {
            'dish name': dish.dish_name,
            'Country': dish.country,
            'calories': dish.total_calories,
            'weight (g)': dish.weight_g,
            'ingredients': json.dumps(dish.ingredients) if dish.ingredients else '[]'
        }
        
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_excel(dishes_path, index=False)
        
        dish.id = len(df) - 1
        return dish
    
    except Exception as e:
        logger.error(f"Error adding dish: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/dishes/{dish_id}", response_model=Dish)
async def update_dish(dish_id: int, dish: Dish):
    """Update an existing dish"""
    try:
        dishes_path = "app/data/raw/dishes.xlsx"
        
        if not Path(dishes_path).exists():
            raise HTTPException(status_code=404, detail="Dishes file not found")
        
        df = pd.read_excel(dishes_path)
        
        if dish_id >= len(df):
            raise HTTPException(status_code=404, detail="Dish not found")
        
        # Update the dish
        df.at[dish_id, 'dish name'] = dish.dish_name
        df.at[dish_id, 'Country'] = dish.country
        df.at[dish_id, 'calories'] = dish.total_calories
        df.at[dish_id, 'weight (g)'] = dish.weight_g
        
        df.to_excel(dishes_path, index=False)
        
        dish.id = dish_id
        return dish
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating dish: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/dishes/{dish_id}")
async def delete_dish(dish_id: int):
    """Delete a dish"""
    try:
        dishes_path = "app/data/raw/dishes.xlsx"
        
        if not Path(dishes_path).exists():
            raise HTTPException(status_code=404, detail="Dishes file not found")
        
        df = pd.read_excel(dishes_path)
        
        if dish_id >= len(df):
            raise HTTPException(status_code=404, detail="Dish not found")
        
        # Remove the dish
        df = df.drop(dish_id).reset_index(drop=True)
        df.to_excel(dishes_path, index=False)
        
        return {"message": "Dish deleted successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting dish: {e}")
        raise HTTPException(status_code=500, detail=str(e))
