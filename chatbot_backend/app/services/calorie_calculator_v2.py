from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

# Default ingredient weight when adding items
DEFAULT_ADDED_INGREDIENT_WEIGHT_G = 30

class CalorieCalculatorV2:
    """Calculate calories and macros from ingredients"""
    
    def __init__(self, dataset_search):
        self.search = dataset_search
    
    def calculate(self, gpt_result: Dict) -> Dict:
        """
        Calculate total calories, protein, carbs, fats
        
        Returns:
        {
            "food_item": "chicken shawarma wrap",
            "total_calories": 650.0,
            "total_weight_g": 350.0,
            "protein_g": 45.0,
            "carbs_g": 55.0,
            "fats_g": 20.0,
            "ingredients": [...],
            "found": true
        }
        """
        
        food_item = gpt_result.get('food_item', '')
        
        # Step 1: Try to find complete dish
        dish = self.search.find_dish(food_item)
        
        if dish:
            return self._calculate_from_dish(dish, gpt_result)
        
        # Step 2: Calculate from GPT ingredients
        gpt_ingredients = gpt_result.get('ingredients', [])
        
        if gpt_ingredients:
            return self._calculate_from_ingredients(food_item, gpt_ingredients, gpt_result)
        
        # Step 3: Not found
        return self._not_found_result(food_item)
    
    def _calculate_from_dish(self, dish: Dict, gpt_result: Dict) -> Dict:
        """Calculate from existing dish with modifications"""
        
        ingredients = dish.get('ingredients', [])
        modifications = gpt_result.get('modifications', {})
        
        # Apply modifications
        ingredients = self._apply_modifications(ingredients, modifications)
        
        # Calculate totals
        totals = self._sum_nutrition(ingredients)
        
        return {
            "food_item": dish['dish_name'],
            "total_calories": totals['calories'],
            "total_weight_g": totals['weight'],
            "protein_g": totals['protein'],
            "carbs_g": totals['carbs'],
            "fats_g": totals['fats'],
            "ingredients": ingredients,
            "found": True,
            "source": "dishes"
        }
    
    def _calculate_from_ingredients(self, food_item: str, gpt_ingredients: List[Dict], gpt_result: Dict) -> Dict:
        """Calculate from GPT-provided ingredients"""
        
        found_ingredients = []
        
        for gpt_ing in gpt_ingredients:
            name = gpt_ing.get('name', '')
            weight_g = gpt_ing.get('weight_g', 100)
            
            # Search USDA
            usda_food = self.search.find_ingredient(name)
            
            if usda_food:
                nutrients = usda_food.get('foodNutrients', [])
                
                # Extract per 100g values
                calories_per_100 = self._get_nutrient(nutrients, 'Energy', 'KCAL')
                protein_per_100 = self._get_nutrient(nutrients, 'Protein')
                carbs_per_100 = self._get_nutrient(nutrients, 'Carbohydrate, by difference')
                fats_per_100 = self._get_nutrient(nutrients, 'Total lipid (fat)')
                
                # Scale to actual weight
                scale = weight_g / 100.0
                
                found_ingredients.append({
                    "usda_fdc_id": usda_food.get('fdcId'),
                    "name": usda_food.get('description'),
                    "weight_g": weight_g,
                    "calories": round(calories_per_100 * scale, 1),
                    "protein_g": round(protein_per_100 * scale, 1),
                    "carbs_g": round(carbs_per_100 * scale, 1),
                    "fats_g": round(fats_per_100 * scale, 1)
                })
        
        # Calculate totals
        totals = self._sum_nutrition(found_ingredients)
        
        return {
            "food_item": food_item,
            "total_calories": totals['calories'],
            "total_weight_g": totals['weight'],
            "protein_g": totals['protein'],
            "carbs_g": totals['carbs'],
            "fats_g": totals['fats'],
            "ingredients": found_ingredients,
            "found": len(found_ingredients) > 0,
            "source": "gpt_ingredients"
        }
    
    def _apply_modifications(self, ingredients: List[Dict], modifications: Dict) -> List[Dict]:
        """Remove/add ingredients based on modifications"""
        
        remove_items = [r.lower() for r in modifications.get('remove', [])]
        add_items = modifications.get('add', [])
        
        # Remove ingredients
        filtered = []
        for ing in ingredients:
            name_lower = ing.get('name', '').lower()
            should_remove = any(rem in name_lower for rem in remove_items)
            if not should_remove:
                filtered.append(ing)
            else:
                logger.info(f"🗑️ Removed: {ing.get('name')}")
        
        # Add ingredients (search USDA)
        for add_item in add_items:
            usda_food = self.search.find_ingredient(add_item)
            if usda_food:
                nutrients = usda_food.get('foodNutrients', [])
                # Add with default weight
                calories_per_100 = self._get_nutrient(nutrients, 'Energy', 'KCAL')
                protein_per_100 = self._get_nutrient(nutrients, 'Protein')
                carbs_per_100 = self._get_nutrient(nutrients, 'Carbohydrate, by difference')
                fats_per_100 = self._get_nutrient(nutrients, 'Total lipid (fat)')
                
                scale = DEFAULT_ADDED_INGREDIENT_WEIGHT_G / 100.0
                
                filtered.append({
                    "usda_fdc_id": usda_food.get('fdcId'),
                    "name": usda_food.get('description'),
                    "weight_g": DEFAULT_ADDED_INGREDIENT_WEIGHT_G,
                    "calories": round(calories_per_100 * scale, 1),
                    "protein_g": round(protein_per_100 * scale, 1),
                    "carbs_g": round(carbs_per_100 * scale, 1),
                    "fats_g": round(fats_per_100 * scale, 1)
                })
                logger.info(f"➕ Added: {add_item}")
        
        return filtered
    
    def _sum_nutrition(self, ingredients: List[Dict]) -> Dict:
        """Sum all nutrition values"""
        
        total_calories = sum(ing.get('calories', 0) for ing in ingredients)
        total_weight = sum(ing.get('weight_g', 0) for ing in ingredients)
        total_protein = sum(ing.get('protein_g', 0) for ing in ingredients)
        total_carbs = sum(ing.get('carbs_g', 0) for ing in ingredients)
        total_fats = sum(ing.get('fats_g', 0) for ing in ingredients)
        
        return {
            "calories": round(total_calories, 1),
            "weight": round(total_weight, 1),
            "protein": round(total_protein, 1),
            "carbs": round(total_carbs, 1),
            "fats": round(total_fats, 1)
        }
    
    def _get_nutrient(self, nutrients: List[Dict], name: str, unit: str = 'G') -> float:
        """Extract nutrient value from USDA data"""
        for nutrient in nutrients:
            nutrient_info = nutrient.get('nutrient', nutrient)
            nutrient_name = nutrient_info.get('nutrientName', nutrient_info.get('name', ''))
            
            # Use more specific matching to avoid false matches
            # e.g., don't match "fat" when looking for "Total lipid (fat)"
            name_lower = name.lower()
            nutrient_name_lower = nutrient_name.lower()
            
            # Exact match or starts with the nutrient name
            if name_lower == nutrient_name_lower or nutrient_name_lower.startswith(name_lower):
                unit_name = nutrient_info.get('unitName', '')
                if unit.upper() in unit_name.upper() or not unit_name:
                    value = nutrient.get('amount', nutrient.get('value', 0))
                    return float(value) if value else 0.0
        return 0.0
    
    def _not_found_result(self, food_item: str) -> Dict:
        """Return not found result"""
        return {
            "food_item": food_item,
            "total_calories": 0,
            "total_weight_g": 0,
            "protein_g": 0,
            "carbs_g": 0,
            "fats_g": 0,
            "ingredients": [],
            "found": False
        }
