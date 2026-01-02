from typing import List, Dict, Optional, Tuple
from rapidfuzz import fuzz, process
import logging

logger = logging.getLogger(__name__)

class DatasetSearch:
    """Search dishes and USDA ingredients by name"""
    
    def __init__(self, dishes: List[Dict], usda_foods: List[Dict]):
        self.dishes = dishes
        self.usda_foods = usda_foods
        
        # Build name index
        self.dish_names = {d['dish_name'].lower(): d for d in dishes}
        self.usda_names = {f['description'].lower(): f for f in usda_foods}
    
    def find_dish(self, dish_name: str) -> Optional[Dict]:
        """Find dish by name (fuzzy match)"""
        
        dish_lower = dish_name.lower().strip()
        
        # Exact match
        if dish_lower in self.dish_names:
            logger.info(f"✅ Exact dish match: {dish_name}")
            return self.dish_names[dish_lower]
        
        # Fuzzy match
        matches = process.extract(
            dish_lower,
            self.dish_names.keys(),
            scorer=fuzz.WRatio,
            limit=1
        )
        
        if matches and matches[0][1] >= 80:
            match_name = matches[0][0]
            logger.info(f"✅ Fuzzy dish match: {dish_name} → {match_name} ({matches[0][1]})")
            return self.dish_names[match_name]
        
        logger.warning(f"❌ Dish not found: {dish_name}")
        return None
    
    def find_ingredient(self, ingredient_name: str) -> Optional[Dict]:
        """Find USDA ingredient by name"""
        
        ing_lower = ingredient_name.lower().strip()
        
        # Exact match
        if ing_lower in self.usda_names:
            return self.usda_names[ing_lower]
        
        # Fuzzy match (high threshold for accuracy)
        matches = process.extract(
            ing_lower,
            self.usda_names.keys(),
            scorer=fuzz.WRatio,
            limit=1
        )
        
        if matches and matches[0][1] >= 85:
            match_name = matches[0][0]
            logger.info(f"✅ Found ingredient: {ingredient_name} → {match_name}")
            return self.usda_names[match_name]
        
        logger.warning(f"❌ Ingredient not found: {ingredient_name}")
        return None
    
    def find_ingredients_batch(self, ingredient_names: List[str]) -> List[Tuple[str, Optional[Dict]]]:
        """Find multiple ingredients"""
        results = []
        for name in ingredient_names:
            ing = self.find_ingredient(name)
            results.append((name, ing))
        return results
