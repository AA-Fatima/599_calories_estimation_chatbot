# ML Features Testing Guide

## Overview
This guide provides comprehensive testing instructions for the newly integrated machine learning features in the Arabic Calorie Chatbot.

## 🎯 ML Features Added

### 1. Intent Classification (Zero-Shot)
- **Model**: `facebook/bart-large-mnli`
- **Purpose**: Classify user intent (query_food, modify_remove, modify_add, greeting, help)
- **Fallback**: Rule-based classification if ML fails or confidence < threshold

### 2. Semantic Search
- **Model**: `paraphrase-multilingual-MiniLM-L12-v2`
- **Purpose**: Find foods using semantic similarity (cosine similarity)
- **Fallback**: Fuzzy matching if semantic model unavailable

### 3. Named Entity Recognition (NER)
- **Model**: `Davlan/bert-base-multilingual-cased-ner-hrl`
- **Purpose**: Extract food items from queries (supports Arabic and English)
- **Fallback**: Rule-based extraction if NER fails

## 📋 Prerequisites

### Install Dependencies
```bash
cd chatbot_backend
pip install -r requirements.txt
```

This will install:
- `transformers>=4.35.0` - For ML models
- `torch>=2.0.0` - PyTorch for model inference
- `accelerate>=0.24.0` - For efficient model loading
- Other existing dependencies (fastapi, sentence-transformers, etc.)

### Configuration
Check `app/config.py` for ML settings:
```python
USE_ML_INTENT_CLASSIFICATION = True  # Enable/disable intent classification
USE_SEMANTIC_SEARCH = True           # Enable/disable semantic search
USE_NER_EXTRACTION = True            # Enable/disable NER
ML_CONFIDENCE_THRESHOLD = 0.5        # Minimum confidence for ML predictions
```

## 🧪 Testing Scenarios

### Test 1: Intent Classification

#### Test Queries
```python
test_cases = [
    # Greeting Intent
    ("Hello", "GREETING"),
    ("Hi there", "GREETING"),
    ("marhaba", "GREETING"),
    
    # Help Intent
    ("How do I use this?", "HELP"),
    ("What can you do?", "HELP"),
    ("help me", "HELP"),
    
    # Query Food Intent
    ("How many calories in shawarma?", "QUERY_FOOD"),
    ("calories in hummus", "QUERY_FOOD"),
    ("كم سعرة حرارية في الشاورما؟", "QUERY_FOOD"),
    
    # Modify Remove Intent
    ("chicken without sauce", "MODIFY_REMOVE"),
    ("shawarma bidun tahini", "MODIFY_REMOVE"),
    ("burger no cheese", "MODIFY_REMOVE"),
    
    # Modify Add Intent
    ("shawarma with extra garlic", "MODIFY_ADD"),
    ("add cheese to my burger", "MODIFY_ADD"),
]
```

#### How to Test
```bash
cd chatbot_backend
python3 -c "
from app.core.nlp_engine import NLPEngine

nlp = NLPEngine()
queries = ['Hello', 'How many calories in shawarma?', 'chicken without sauce']

for query in queries:
    result = nlp.parse_query(query)
    print(f'Query: {query}')
    print(f'Intent: {result.intent}')
    print(f'Food Items: {result.food_items}')
    print()
"
```

### Test 2: Semantic Search

#### Test Queries (with typos and variations)
```python
test_searches = [
    # Exact matches
    "shawarma",
    "hummus",
    "falafel",
    
    # Typos
    "shwarma",  # missing 'a'
    "humus",    # missing 'm'
    "flafel",   # typo
    
    # Variations
    "chicken shawarma",
    "beef shawarma",
    "arabic bread",
    "pita bread",
    
    # Arabic
    "شاورما",
    "حمص",
    "فلافل",
]
```

#### Expected Results
- Semantic search should find correct dishes even with typos
- Should rank dishes by similarity score
- Country-specific dishes should be prioritized when country is specified

#### How to Test
After starting the server, use the API:
```bash
curl -X POST http://localhost:8000/api/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "message": "shwarma",
    "session_id": "test-123",
    "country": "lebanon"
  }'
```

### Test 3: NER Extraction

#### Test Queries
```python
ner_test_cases = [
    # Simple food items
    "I want chicken",
    "Show me hummus calories",
    
    # Multiple items
    "chicken and rice",
    "shawarma with hummus",
    
    # Arabic food names
    "كبسة",
    "مندي دجاج",
    
    # Mixed language
    "I ate كبسة today",
    
    # With modifiers
    "grilled chicken without skin",
    "fried rice with vegetables",
]
```

#### Expected Results
- NER should extract food entities from the query
- Should combine with rule-based extraction for better coverage
- Should handle both Arabic and English food names

### Test 4: Fallback Mechanisms

#### Test ML Disabled
```python
# In config.py, temporarily set:
USE_ML_INTENT_CLASSIFICATION = False
USE_SEMANTIC_SEARCH = False
USE_NER_EXTRACTION = False

# Test that the app still works with rule-based methods
```

#### Test Low Confidence
```python
# Set a high threshold to force fallback:
ML_CONFIDENCE_THRESHOLD = 0.9

# Test ambiguous queries that should fall back to rules
test_queries = [
    "xyz food",  # Nonsensical - should use rules
    "???",       # Invalid - should use rules
]
```

### Test 5: Performance Benchmarks

#### Startup Time
```bash
time python3 -c "
from app.main import app
import asyncio

async def test():
    async with app.router.lifespan_context(app):
        print('Startup complete')

asyncio.run(test())
"
```

**Expected**: < 10 seconds

#### Query Processing Time
```bash
# Time a single query
time curl -X POST http://localhost:8000/api/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "message": "shawarma",
    "session_id": "test-perf",
    "country": "lebanon"
  }'
```

**Expected**: < 2 seconds per query

### Test 6: Edge Cases

#### Edge Case Tests
```python
edge_cases = [
    # Empty query
    "",
    
    # Very long query
    "shawarma " * 100,
    
    # Special characters
    "!@#$%^&*()",
    
    # Numbers only
    "123456",
    
    # Mixed scripts
    "chicken الدجاج poulet",
    
    # Emojis
    "🍗🍖🥙",
]
```

#### Expected Behavior
- Should not crash
- Should gracefully fall back to existing logic
- Should log appropriate warnings

## 📊 Success Criteria

### Intent Classification
- ✅ Accuracy > 85% on test queries
- ✅ Confidence scores logged for all predictions
- ✅ Fallback to rules when confidence < threshold
- ✅ No crashes on edge cases

### Semantic Search
- ✅ Finds correct dishes with typos (edit distance ≤ 2)
- ✅ Handles variations (e.g., "chicken shawarma" matches "shawarma")
- ✅ Country-specific boosting works
- ✅ Falls back to fuzzy matching if needed

### NER Extraction
- ✅ Extracts food items from complex queries
- ✅ Handles both Arabic and English
- ✅ Combines with rule-based for better coverage
- ✅ Falls back gracefully if NER fails

### Performance
- ✅ Startup time < 10 seconds
- ✅ Query processing < 2 seconds
- ✅ Memory usage reasonable (~2GB max)

### Stability
- ✅ All ML failures handled gracefully
- ✅ No breaking changes to existing API
- ✅ Backward compatible with existing clients

## 🔧 Troubleshooting

### Issue: Models not loading
**Solution**: Check disk space and memory. Models total ~400MB.

### Issue: Slow startup
**Solution**: 
- Embeddings are precomputed on startup (expected)
- Check if you have sufficient memory
- Disable semantic search if needed: `USE_SEMANTIC_SEARCH=False`

### Issue: Low accuracy
**Solution**:
- Adjust `ML_CONFIDENCE_THRESHOLD` (try 0.3 or 0.7)
- Check model loading logs
- Verify ML models downloaded correctly

### Issue: Out of memory
**Solution**:
- Use CPU instead of GPU
- Reduce batch size in embedding computation
- Disable features selectively

## 📝 Logging

### What Gets Logged
```
- Model load times
- ML prediction confidence scores
- Fallback usage frequency
- Search result quality metrics
- Query processing times
```

### Check Logs
```bash
# Start server with verbose logging
cd chatbot_backend
python3 -m uvicorn app.main:app --reload --log-level info
```

Look for:
- `✅ Semantic model initialized in X.XXs`
- `✅ Intent classifier initialized in X.XXs`
- `✅ NER model initialized in X.XXs`
- `ML intent: QUERY_FOOD (confidence: 0.87)`
- `⚠️ ML confidence too low, falling back to rules`

## 🚀 Running the Server

### Development Mode
```bash
cd chatbot_backend
uvicorn app.main:app --reload --port 8000
```

### Production Mode
```bash
cd chatbot_backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

## 📚 Additional Resources

- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Sentence Transformers](https://www.sbert.net/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)

## 💡 Tips

1. **Start with ML disabled** to verify basic functionality
2. **Enable features one by one** to isolate issues
3. **Monitor logs** for confidence scores and fallback usage
4. **Test with real user queries** from your dataset
5. **Adjust thresholds** based on your accuracy requirements

---

Last Updated: 2025-12-29
