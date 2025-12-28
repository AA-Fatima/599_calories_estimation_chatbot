import logging
import json
from typing import Dict, Optional, Tuple
from openai import AsyncOpenAI
import asyncio

from app.config import settings

logger = logging.getLogger(__name__)


class FallbackService: 
    """
    Fallback to ChatGPT when food not found in database
    """
    
    SYSTEM_PROMPT = """You are a nutrition expert specializing in Arabic and Middle Eastern cuisine. 

When asked about calories in a food item, provide:
1. Estimated total calories per 100g
2. Typical serving size in grams
3. Breakdown of main ingredients with calories (if it's a dish)
4. Confidence level (high/medium/low)

IMPORTANT: 
- Be specific to the country if mentioned (e.g., Egyptian vs Lebanese versions differ)
- Consider typical serving sizes in that region
- If unsure, indicate this is an estimate
- Use common ingredient portions

Respond in this EXACT JSON format:
{
    "food_name": "Name of the food",
    "total_calories": 250,
    "weight_g": 100,
    "calories_per_100g": 250,
    "typical_serving_g": 150,
    "ingredients": [
        {"name": "Ingredient 1", "weight_g": 50, "calories": 100},
        {"name": "Ingredient 2", "weight_g": 30, "calories": 60}
    ],
    "confidence": "medium",
    "notes": "Any relevant notes about regional variations or uncertainty"
}

If it's a single ingredient (like apple, chicken), ingredients array can have just one item.
"""

    def __init__(self):
        self.openai_client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize OpenAI client"""
        if settings.OPENAI_API_KEY:
            self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            logger.info("✅ OpenAI client initialized")
        else:
            logger.warning("⚠️ OpenAI API key not found - fallback disabled")
    
    async def get_calories_from_gpt(
        self,
        food_name: str,
        country: str = "",
        modifications: Optional[Dict] = None
    ) -> Tuple[Optional[Dict], bool]:
        """
        Get calorie estimate from ChatGPT
        
        Returns:
            Tuple of (result_dict, success)
        """
        
        if not self.openai_client:
            logger.error("OpenAI client not initialized")
            return None, False
        
        # Build the query
        query = f"What are the calories in {food_name}"
        if country:
            query += f" as prepared in {country}"
        if modifications:
            if modifications.get('remove'):
                query += f" without {', '.join(modifications['remove'])}"
            if modifications.get('add'):
                query += f" with added {', '.join(modifications['add'])}"
        
        query += "?  Provide detailed nutritional information."
        
        try:
            logger.info(f"🤖 Asking ChatGPT:  {query}")
            
            response = await self.openai_client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content":  query}
                ],
                temperature=0.3,
                max_tokens=1000,
                response_format={"type": "json_object"}  # Force JSON response
            )
            
            content = response.choices[0].message.content
            logger.info(f"📥 GPT Response: {content[: 200]}...")
            
            # Parse JSON response
            result = self._parse_gpt_response(content, food_name)
            
            if result:
                logger.info(f"✅ GPT found:  {result.get('food_name')} = {result.get('total_calories')} kcal")
                return result, True
            else: 
                logger.error("Failed to parse GPT response")
                return None, False
            
        except Exception as e: 
            logger.error(f"❌ ChatGPT query failed: {str(e)}")
            return None, False
    
    def _parse_gpt_response(self, content: str, food_name: str) -> Optional[Dict]:
        """Parse ChatGPT JSON response"""
        try:
            # Parse JSON
            result = json.loads(content)
            
            # Validate required fields
            if 'total_calories' not in result and 'calories_per_100g' not in result:
                logger.error("GPT response missing calorie information")
                return None
            
            # Normalize the response
            normalized = {
                'food_name': result.get('food_name', food_name),
                'total_calories': result.get('calories_per_100g', result.get('total_calories', 0)),
                'weight_g': result.get('weight_g', 100),
                'ingredients': result.get('ingredients', []),
                'confidence': result.get('confidence', 'medium'),
                'notes':  result.get('notes', ''),
                'is_approximate':  True,
                'source': 'chatgpt'
            }
            
            # Ensure we have positive calories
            if normalized['total_calories'] <= 0:
                logger.error(f"Invalid calories from GPT: {normalized['total_calories']}")
                return None
            
            return normalized
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse GPT JSON: {e}")
            logger.error(f"Content was: {content}")
            return None
        except Exception as e:
            logger.error(f"Error parsing GPT response: {e}")
            return None