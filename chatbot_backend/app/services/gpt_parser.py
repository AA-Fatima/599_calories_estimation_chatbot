from openai import OpenAI
from typing import Dict, List, Optional
import json
import logging

logger = logging.getLogger(__name__)

class GPTParser:
    """Parse user queries using OpenAI GPT-4"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    async def parse_query(self, user_message: str, country: str = "") -> Dict:
        """
        Parse user message and return structured JSON
        
        Returns:
        {
            "food_item": "chicken shawarma wrap",
            "modifications": {
                "remove": ["french fries", "mayonnaise"],
                "add": ["extra tomato"]
            },
            "ingredients": [
                {"name": "Chicken, broilers or fryers, breast, meat only, raw", "weight_g": 150},
                {"name": "Bread, pita, white, enriched", "weight_g": 70},
                {"name": "Tomatoes, raw", "weight_g": 50}
            ],
            "quantity_multiplier": 1.0
        }
        """
        
        system_prompt = f"""You are a food calorie assistant. Parse user queries about food and return ONLY valid JSON.

CRITICAL RULES:
1. Use EXACT USDA ingredient names (e.g., "Chicken, broilers or fryers, breast, meat only, raw", "Tomatoes, raw", "Oil, olive, salad or cooking")
2. Return ONLY the JSON object, no other text
3. Estimate realistic weights in grams
4. If user mentions country context: {country or 'any'}
5. Detect modifications (without, no, remove, add, extra, with)

USDA Format Examples:
- "Apples, raw" NOT "apple"
- "Bananas, raw" NOT "banana"  
- "Chicken, broilers or fryers, breast, meat only, raw" NOT "chicken breast"
- "Tomatoes, raw" NOT "tomato"
- "Onions, raw" NOT "onion"
- "Rice, white, long-grain, regular, cooked" NOT "rice"
- "Oil, olive, salad or cooking" NOT "olive oil"
- "Bread, pita, white, enriched" NOT "pita bread"

Return format:
{{
  "food_item": "dish name",
  "modifications": {{
    "remove": ["ingredient1"],
    "add": ["ingredient2"]
  }},
  "ingredients": [
    {{"name": "USDA exact name", "weight_g": number}}
  ],
  "quantity_multiplier": 1.0
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            logger.info(f"✅ GPT parsed: {result.get('food_item')}")
            return result
            
        except Exception as e:
            logger.error(f"GPT parsing failed: {e}")
            # Fallback: simple parsing
            return self._fallback_parse(user_message)
    
    def _fallback_parse(self, text: str) -> Dict:
        """Simple fallback if GPT fails"""
        return {
            "food_item": text.strip(),
            "modifications": {"remove": [], "add": []},
            "ingredients": [],
            "quantity_multiplier": 1.0
        }
