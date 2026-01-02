# GPT-Powered Calorie Calculator - Implementation Guide

## Overview

This implementation adds a GPT-4 powered calorie calculator system with the following features:

1. **GPT Parser** - Uses OpenAI GPT-4 to parse user queries into structured JSON
2. **Dataset Search** - Fuzzy search for dishes and USDA ingredients by name
3. **Calorie Calculator V2** - Calculates calories with macros (protein, carbs, fats)
4. **Missing Dish Logger** - Logs dishes not found in CSV format
5. **Admin Dashboard** - Web UI for managing dishes and viewing missing items

## Architecture

```
User Query → GPT-4 Parser → Structured JSON
                              ↓
                    Dataset Search (Fuzzy Match)
                              ↓
                    Calorie Calculator V2
                              ↓
              Calculate: calories + protein + carbs + fats
                              ↓
                    Apply Modifications (add/remove)
                              ↓
                         Return Result
                              ↓
                    Log Missing Dishes (if not found)
```

## Backend Components

### 1. GPT Parser (`app/services/gpt_parser.py`)

Parses user queries using OpenAI GPT-4:

```python
from app.services.gpt_parser import GPTParser

parser = GPTParser(api_key="your-openai-api-key")
result = await parser.parse_query("chicken shawarma without fries", country="lebanon")

# Returns:
{
    "food_item": "chicken shawarma wrap",
    "modifications": {
        "remove": ["french fries"],
        "add": []
    },
    "ingredients": [
        {"name": "Chicken, broilers or fryers, breast, meat only, raw", "weight_g": 150},
        {"name": "Bread, pita, white, enriched", "weight_g": 70}
    ],
    "quantity_multiplier": 1.0
}
```

### 2. Dataset Search (`app/services/dataset_search.py`)

Fuzzy search for dishes and ingredients:

```python
from app.services.dataset_search import DatasetSearch

search = DatasetSearch(dishes_list, usda_foods_list)

# Find dish
dish = search.find_dish("shawarma")

# Find ingredient
ingredient = search.find_ingredient("Apples, raw")

# Batch search
results = search.find_ingredients_batch(["apple", "banana", "chicken"])
```

### 3. Calorie Calculator V2 (`app/services/calorie_calculator_v2.py`)

Calculates calories and macros:

```python
from app.services.calorie_calculator_v2 import CalorieCalculatorV2

calculator = CalorieCalculatorV2(dataset_search)
result = calculator.calculate(gpt_result)

# Returns:
{
    "food_item": "chicken shawarma wrap",
    "total_calories": 650.0,
    "total_weight_g": 350.0,
    "protein_g": 45.0,
    "carbs_g": 55.0,
    "fats_g": 20.0,
    "ingredients": [...],
    "found": true,
    "source": "dishes"
}
```

### 4. Missing Dish Logger (`app/services/missing_dish_logger.py`)

Logs dishes not found in dataset:

```python
from app.services.missing_dish_logger import MissingDishLogger

logger = MissingDishLogger(log_file="missing_dishes.csv")
logger.log(
    dish_name="unknown dish",
    country="lebanon",
    gpt_result=gpt_result
)

# Get all missing dishes
missing = logger.get_missing_dishes()
```

## API Endpoints

### Chat V2 (GPT-Powered)

**POST** `/api/chat_v2/message`

Request:
```json
{
    "message": "chicken shawarma without fries",
    "country": "lebanon"
}
```

Response:
```json
{
    "food_item": "chicken shawarma wrap",
    "total_calories": 650.0,
    "total_weight_g": 350.0,
    "protein_g": 45.0,
    "carbs_g": 55.0,
    "fats_g": 20.0,
    "ingredients": [
        {
            "name": "Chicken, broilers or fryers, breast, meat only, raw",
            "weight_g": 150,
            "calories": 165,
            "protein_g": 31,
            "carbs_g": 0,
            "fats_g": 3.6
        }
    ],
    "found": true,
    "source": "dishes"
}
```

### Admin API

**GET** `/api/admin/dishes`
- Returns all dishes

**GET** `/api/admin/missing-dishes`
- Returns all logged missing dishes

**POST** `/api/admin/dishes`
- Add a new dish

**PUT** `/api/admin/dishes/{id}`
- Update an existing dish

**DELETE** `/api/admin/dishes/{id}`
- Delete a dish

## Frontend Components

### Admin Dashboard

Access at: `http://localhost:4200/admin`

Features:
- View all dishes in a table
- Add new dishes with modal form
- Edit existing dishes
- Delete dishes
- View missing dishes log
- Tab navigation between dishes and missing items

## Setup Instructions

### Backend Setup

1. Install dependencies:
```bash
cd chatbot_backend
pip install -r requirements.txt
```

2. Set OpenAI API key in `.env`:
```
OPENAI_API_KEY=your-api-key-here
```

3. Run the server:
```bash
python run.py
```

### Frontend Setup

1. Install dependencies:
```bash
cd chatbot_frontend
npm install
```

2. Run development server:
```bash
npm start
```

3. Access admin dashboard:
```
http://localhost:4200/admin
```

## Testing

### Test Backend Services

```bash
cd chatbot_backend
python -c "from app.main import app; print('✅ Backend imports successfully')"
```

### Test API Endpoints

```bash
# Test chat_v2 endpoint
curl -X POST http://localhost:8000/api/chat_v2/message \
  -H "Content-Type: application/json" \
  -d '{"message": "apple", "country": "lebanon"}'

# Test admin endpoints
curl http://localhost:8000/api/admin/dishes
curl http://localhost:8000/api/admin/missing-dishes
```

## Example Queries

### Simple Ingredient
```
Input: "apple"
Output: Apples, raw → 52 kcal per 100g
```

### Dish with Modifications
```
Input: "chicken shawarma wrap without fries"
Output: Chicken Shawarma Wrap (minus fries) → 550 kcal
```

### Custom Weight
```
Input: "banana 150g"
Output: Bananas, raw → 133 kcal (150g)
```

### Arabic Query
```
Input: "تفاحة كبيرة"
Output: GPT translates → apple → 95 kcal
```

## Key Features

### 1. Smart Parsing
- GPT-4 understands natural language
- Detects modifications (without, no, add, extra)
- Extracts quantity multipliers
- Translates Arabic to English

### 2. Fuzzy Matching
- Finds dishes even with typos
- Matches partial names
- High threshold for accuracy

### 3. Macro Tracking
- Calculates protein, carbs, fats
- Scales nutrients by weight
- Sums totals from all ingredients

### 4. Missing Dish Logging
- Logs dishes not found
- Stores GPT ingredients for review
- CSV format for easy import
- Admin dashboard view

### 5. Modification Support
- Remove ingredients
- Add ingredients
- Recalculates totals automatically

## Success Criteria

✅ User query → GPT → JSON with ingredients  
✅ Search local dataset by ingredient names  
✅ Calculate calories + protein + carbs + fats  
✅ Handle modifications (add/remove)  
✅ Log missing dishes  
✅ Admin dashboard to manage dishes  
✅ Fallback logic if GPT fails  

## File Structure

```
chatbot_backend/
├── app/
│   ├── api/routes/
│   │   ├── chat_v2.py          # New GPT-powered chat endpoint
│   │   └── admin.py            # Admin API for dish management
│   ├── services/
│   │   ├── gpt_parser.py       # GPT-4 query parser
│   │   ├── dataset_search.py   # Fuzzy search service
│   │   ├── calorie_calculator_v2.py  # V2 calculator with macros
│   │   └── missing_dish_logger.py    # CSV logging
│   └── main.py                 # Updated with new services

chatbot_frontend/
├── src/app/
│   ├── features/
│   │   └── admin-dashboard/
│   │       ├── admin-dashboard.component.ts
│   │       ├── admin-dashboard.component.html
│   │       └── admin-dashboard.component.scss
│   ├── core/services/
│   │   └── admin.service.ts    # Admin API service
│   └── app.routes.ts           # Updated with admin route
```

## Next Steps

1. Set OpenAI API key in production environment
2. Test GPT parser with real queries
3. Populate dishes.xlsx with more dishes
4. Review and add missing dishes from admin dashboard
5. Add authentication to admin dashboard
6. Add ingredient breakdown to admin UI
7. Export missing dishes to Excel for bulk import

## Support

For issues or questions, please refer to the main repository documentation or create an issue on GitHub.
