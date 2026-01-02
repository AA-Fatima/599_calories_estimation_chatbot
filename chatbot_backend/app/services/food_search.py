from typing import List, Dict, Any, Tuple
from rapidfuzz import fuzz, process
import logging
from app.config import settings

logger = logging.getLogger(__name__)

# Constants
USDA_LARGE_DATABASE_THRESHOLD = 5000  # Threshold for using batch encoding

# Try to import sentence_transformers util at module level
try:
    from sentence_transformers import util as st_util
    ST_UTIL_AVAILABLE = True
except ImportError:
    ST_UTIL_AVAILABLE = False
    st_util = None


class FoodSearchService:
    """Food search using semantic embeddings + fuzzy matching for intelligent food discovery"""
    
    def __init__(
        self,
        usda_foundation:  Any,
        usda_sr_legacy:  Any,
        dishes: Any,
        nlp_engine: Any
    ):
        # Extract foods
        self.usda_foundation = self._get_foods_list(usda_foundation)
        self.usda_sr_legacy = self._get_foods_list(usda_sr_legacy)
        self.dishes = self._get_dishes_list(dishes)
        self.nlp_engine = nlp_engine
        
        logger.info(f"Loaded - Foundation: {len(self.usda_foundation)}, SR Legacy: {len(self.usda_sr_legacy)}, Dishes: {len(self.dishes)}")
        
        # Build search index
        self.search_index = self._build_search_index()
        
        # Separate indices
        self.dish_index = [item for item in self.search_index if item["source"] == "dishes"]
        self.usda_index = [item for item in self.search_index if item["source"] != "dishes"]
        
        # Name lists for fuzzy search
        self.dish_names = [item["name"] for item in self.dish_index]
        self.usda_names = [item["name"] for item in self.usda_index]
        self.all_names = [item["name"] for item in self.search_index]
        
        # Semantic embeddings cache
        self.dish_embeddings = None
        self.usda_embeddings = None
        
        logger.info(f"Search index:  {len(self.search_index)} total ({len(self.dish_index)} dishes, {len(self.usda_index)} USDA)")
    
    def _get_foods_list(self, data: Any) -> List[Dict]: 
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ['foods', 'FoundationFoods', 'SRLegacyFoods']: 
                if key in data and isinstance(data[key], list):
                    return data[key]
        return []
    
    def _get_dishes_list(self, data: Any) -> List[Dict]: 
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ['all_dishes', 'dishes']: 
                if key in data and isinstance(data[key], list):
                    return data[key]
        return []
    
    def _build_search_index(self) -> List[Dict]:
        index = []
        
        for dish in self.dishes:
            name = dish.get("dish_name", "")
            if name:
                index.append({
                    "name": name.lower(),
                    "original_name": name,
                    "data": dish,
                    "source": "dishes",
                    "country": dish.get("country", "").lower()
                })
        
        for food in self.usda_foundation:
            name = food.get("description", "")
            if name:
                index.append({
                    "name":  name.lower(),
                    "original_name": name,
                    "data":  food,
                    "source": "usda_foundation",
                    "country":  ""
                })
        
        for food in self.usda_sr_legacy: 
            name = food.get("description", "")
            if name:
                index.append({
                    "name": name.lower(),
                    "original_name": name,
                    "data": food,
                    "source": "usda_sr_legacy",
                    "country":  ""
                })
        
        return index
    
    def precompute_embeddings(self):
        """Pre-compute semantic embeddings for all dishes on startup"""
        if not settings.USE_SEMANTIC_SEARCH:
            logger.info("⚠️ Semantic search disabled in config")
            return
            
        if not self.nlp_engine.semantic_model:
            logger.warning("⚠️ Semantic model not available, skipping embedding precomputation")
            return
        
        try:
            import time
            start_time = time.time()
            
            logger.info("Computing semantic embeddings for dishes...")
            if self.dish_names:
                self.dish_embeddings = self.nlp_engine.semantic_model.encode(
                    self.dish_names,
                    convert_to_tensor=True,
                    show_progress_bar=False
                )
            
            logger.info("Computing semantic embeddings for USDA foods...")
            if self.usda_names:
                # USDA is large, so we might want to limit or batch this
                if len(self.usda_names) > USDA_LARGE_DATABASE_THRESHOLD:
                    logger.info(f"USDA database is large ({len(self.usda_names)}), using batch encoding")
                self.usda_embeddings = self.nlp_engine.semantic_model.encode(
                    self.usda_names,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    batch_size=32
                )
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Embeddings computed in {elapsed:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to precompute embeddings: {e}")
            self.dish_embeddings = None
            self.usda_embeddings = None
    
    def search(self, query: str, country: str = "", top_k: int = 5, search_type: str = 'auto') -> List[Tuple[Dict, str, float]]: 
        """
        Search for food with type awareness
        
        Args:
            query: Food name to search
            country: Country filter
            top_k: Number of results
            search_type: 'ingredient', 'dish', or 'auto' (detect automatically)
        """
        query_lower = query.lower().strip()
        
        if not query_lower: 
            return []
        
        # Validate search_type parameter
        valid_search_types = {'auto', 'ingredient', 'dish'}
        if search_type not in valid_search_types:
            logger.warning(f"Invalid search_type '{search_type}', defaulting to 'auto'")
            search_type = 'auto'
        
        logger.info(f"Searching for: '{query_lower}' (country: {country})")
        
        # Detect search type if auto
        if search_type == 'auto':
            search_type = self.nlp_engine.classify_food_type(query)
            logger.info(f"Auto-detected search type: {search_type.upper()}")
        
        # Route to appropriate search strategy
        if search_type == 'ingredient':
            logger.info(f"🍎 INGREDIENT SEARCH for: '{query_lower}'")
            return self._search_ingredient_priority(query_lower, top_k)
        else:
            logger.info(f"🍽️ DISH SEARCH for: '{query_lower}' (country: {country})")
            return self._search_dish_priority(query_lower, country, top_k)
    
    def _search_ingredient_priority(self, query: str, top_k: int = 5) -> List[Tuple[Dict, str, float]]:
        """Search USDA only (skip dishes entirely)"""
        
        logger.info("Searching USDA databases only (ingredient mode)...")
        
        # Exact match in USDA
        for item in self.usda_index:
            if query == item["name"]:
                logger.info(f"✅ Exact USDA match: {item['original_name']}")
                return [(item["data"], item["source"], 1.0)]
        
        results = []
        
        # Semantic search in USDA
        if settings.USE_SEMANTIC_SEARCH and self.usda_embeddings is not None:
            semantic_results = self._semantic_search_usda(query)
            if semantic_results:
                results.extend(semantic_results)
                logger.info(f"Semantic USDA: {len(semantic_results)} results")
        
        # Fuzzy search in USDA
        fuzzy_results = self._fuzzy_search_usda(query)
        if fuzzy_results:
            seen = {data.get("description", "").lower() for data, _, _ in results}
            for data, source, score in fuzzy_results:
                name = data.get("description", "").lower()
                if name not in seen:
                    results.append((data, source, score))
                    seen.add(name)
        
        if results:
            results.sort(key=lambda x: x[2], reverse=True)
            logger.info(f"✅ Top USDA result: {results[0][0].get('description')} ({results[0][2]:.2f})")
            return results[:top_k]
        
        logger.info(f"❌ No USDA results for: '{query}'")
        return []
    
    def _search_dish_priority(self, query: str, country: str, top_k: int) -> List[Tuple[Dict, str, float]]:
        """Search dishes first, then USDA (normal flow)"""
        
        country_lower = country.lower().strip() if country else ""
        
        # ========================================
        # STEP 1: EXACT MATCH (dishes and USDA)
        # ========================================
        for item in self.search_index:
            if query == item["name"]:
                if item["source"] == "dishes": 
                    if not country_lower or item["country"] == country_lower: 
                        logger.info(f"✅ Exact dish match: {item['original_name']}")
                        return [(item["data"], item["source"], 1.0)]
                else:
                    logger.info(f"✅ Exact USDA match: {item['original_name']}")
                    return [(item["data"], item["source"], 1.0)]
        
        results = []
        
        # ========================================
        # STEP 2: SEARCH DISHES (semantic + fuzzy)
        # ========================================
        logger.info("🔍 Searching in dishes...")
        
        # Try semantic search first if available
        if settings.USE_SEMANTIC_SEARCH and self.dish_embeddings is not None:
            semantic_results = self._semantic_search_dishes(query, country_lower)
            if semantic_results:
                results.extend(semantic_results)
                logger.info(f"Semantic search found {len(semantic_results)} dish results")
        
        # Combine with fuzzy search for robustness
        fuzzy_results = self._fuzzy_search_dishes(query, country_lower)
        if fuzzy_results:
            # Merge results, avoiding duplicates
            seen_names = {data.get("dish_name", "").lower() for data, _, _ in results}
            for data, source, score in fuzzy_results:
                name = data.get("dish_name", "").lower()
                if name not in seen_names:
                    results.append((data, source, score))
                    seen_names.add(name)
        
        # If we found good dish matches, return them
        if results: 
            results.sort(key=lambda x: x[2], reverse=True)
            
            if results[0][2] >= 0.70:  # Good confidence threshold
                logger.info(f"✅ Found dish:  {results[0][0].get('dish_name')} (confidence: {results[0][2]:.2f})")
                return results[:top_k]
            else:
                logger.info(f"⚠️ Dish matches too weak (best: {results[0][2]:.2f}), trying USDA...")
        else:
            logger.info("❌ No dishes found, trying USDA...")
        
        # ========================================
        # STEP 3: SEARCH USDA (semantic + fuzzy)
        # ========================================
        logger.info("🔍 Searching in USDA databases...")
        
        usda_results = []
        
        # Try semantic search first if available
        if settings.USE_SEMANTIC_SEARCH and self.usda_embeddings is not None:
            semantic_usda = self._semantic_search_usda(query)
            if semantic_usda:
                usda_results.extend(semantic_usda)
                logger.info(f"Semantic search found {len(semantic_usda)} USDA results")
        
        # Combine with word-level and fuzzy matching
        word_fuzzy_usda = self._fuzzy_search_usda(query)
        if word_fuzzy_usda:
            seen_names = {data.get("description", "").lower() for data, _, _ in usda_results}
            for data, source, score in word_fuzzy_usda:
                name = data.get("description", "").lower()
                if name not in seen_names:
                    usda_results.append((data, source, score))
                    seen_names.add(name)
        
        # If USDA results found, return them
        if usda_results: 
            usda_results.sort(key=lambda x: x[2], reverse=True)
            
            if usda_results: 
                logger.info(f"✅ Found USDA ingredient: {usda_results[0][0].get('description')} (confidence: {usda_results[0][2]:.2f})")
                return usda_results[:top_k]
        
        # ========================================
        # STEP 4: NOTHING FOUND
        # ========================================
        logger.info(f"❌ No results found for:  '{query}'")
        return []
    
    def _semantic_search_dishes(self, query: str, country: str = "") -> List[Tuple[Dict, str, float]]:
        """Perform semantic search on dishes"""
        if not self.nlp_engine.semantic_model or self.dish_embeddings is None:
            return []
        
        if not ST_UTIL_AVAILABLE:
            logger.warning("sentence_transformers.util not available")
            return []
        
        try:
            # Encode query
            query_embedding = self.nlp_engine.semantic_model.encode(query, convert_to_tensor=True)
            
            # Compute cosine similarity
            similarities = st_util.cos_sim(query_embedding, self.dish_embeddings)[0]
            
            # Get top results
            top_indices = similarities.argsort(descending=True)[:5]
            
            results = []
            for idx in top_indices:
                idx = idx.item()
                score = similarities[idx].item()
                
                # Only include if score is reasonable
                if score >= 0.4:  # Semantic similarity threshold
                    item = self.dish_index[idx]
                    
                    # Apply country filter/penalty
                    final_score = score
                    if country and item["country"] != country:
                        final_score *= 0.85
                    elif country and item["country"] == country:
                        final_score *= 1.1  # Boost country matches
                    
                    results.append((item["data"], item["source"], min(final_score, 1.0)))
            
            return results
            
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
            return []
    
    def _semantic_search_usda(self, query: str) -> List[Tuple[Dict, str, float]]:
        """Perform semantic search on USDA foods"""
        if not self.nlp_engine.semantic_model or self.usda_embeddings is None:
            return []
        
        if not ST_UTIL_AVAILABLE:
            logger.warning("sentence_transformers.util not available")
            return []
        
        try:
            # Encode query
            query_embedding = self.nlp_engine.semantic_model.encode(query, convert_to_tensor=True)
            
            # Compute cosine similarity
            similarities = st_util.cos_sim(query_embedding, self.usda_embeddings)[0]
            
            # Get top results
            top_indices = similarities.argsort(descending=True)[:5]
            
            results = []
            for idx in top_indices:
                idx = idx.item()
                score = similarities[idx].item()
                
                # Only include if score is reasonable
                if score >= 0.5:  # Higher threshold for USDA
                    item = self.usda_index[idx]
                    results.append((item["data"], item["source"], score))
            
            return results
            
        except Exception as e:
            logger.warning(f"Semantic USDA search failed: {e}")
            return []
    
    def _fuzzy_search_dishes(self, query: str, country: str = "") -> List[Tuple[Dict, str, float]]:
        """Perform fuzzy search on dishes"""
        results = []
        
        # Search dishes in selected country first
        if self.dish_names and country:
            country_dishes = [(item["name"], item) for item in self.dish_index if item["country"] == country]
            if country_dishes:
                country_dish_names = [d[0] for d in country_dishes]
                matches = process.extract(query, country_dish_names, scorer=fuzz.WRatio, limit=3)
                
                for name, score, _ in matches:
                    if score >= 70:
                        for dish_name, item in country_dishes:
                            if dish_name == name:
                                results.append((item["data"], item["source"], score / 100.0))
                                break
        
        # Search all dishes if no good country match
        if not results or (results and results[0][2] < 0.8):
            matches = process.extract(query, self.dish_names, scorer=fuzz.WRatio, limit=3)
            
            for name, score, _ in matches:
                if score >= 70:
                    for item in self.dish_index:
                        if item["name"] == name:
                            final_score = score / 100.0
                            # Penalize if not from selected country
                            if country and item["country"] != country:
                                final_score *= 0.9
                            results.append((item["data"], item["source"], final_score))
                            break
        
        return results
    
    def _fuzzy_search_usda(self, query: str) -> List[Tuple[Dict, str, float]]:
        """Perform fuzzy and word-level search on USDA foods"""
        results = []
        
        # Word-level matching (best for ingredients)
        word_matches = []
        for item in self.usda_index:
            item_words = item["name"].replace(',', ' ').replace('-', ' ').replace('(', ' ').replace(')', ' ').lower().split()
            item_words = [w for w in item_words if w]  # Remove empty
            
            # Check if query matches the FIRST word (most relevant)
            if item_words and item_words[0] == query:
                word_matches.append((item, 0.95))
            # Check if query is any complete word in the name
            elif query in item_words:
                word_matches.append((item, 0.85))
            # Check if query is at start of first word (prefix match)
            elif item_words and item_words[0].startswith(query) and len(query) >= 4:
                word_matches.append((item, 0.75))
        
        # Sort by score and add to results
        word_matches.sort(key=lambda x: x[1], reverse=True)
        for item, score in word_matches[:5]: 
            results.append((item["data"], item["source"], score))
        
        # If no word matches, try fuzzy matching
        if not word_matches: 
            usda_fuzzy = process.extract(query, self.usda_names, scorer=fuzz.WRatio, limit=3)
            for name, score, _ in usda_fuzzy: 
                if score >= 60:  # Lower threshold for USDA
                    for item in self.usda_index:
                        if item["name"] == name:
                            results.append((item["data"], item["source"], score / 100.0))
                            break
        
        return results

    def search_ingredient(self, query: str, top_k: int = 3) -> List[Tuple[Dict, str, float]]: 
        """Search USDA only"""
        query_lower = query.lower().strip()
        results = []
        
        # Contained matches
        for item in self.usda_index:
            if query_lower in item["name"]: 
                score = len(query_lower) / len(item["name"]) + 0.5
                results.append((item["data"], item["source"], min(score, 0.95)))
        
        if results:
            results.sort(key=lambda x: x[2], reverse=True)
            return results[:top_k]
        
        # Fuzzy match
        if self.usda_names:
            matches = process.extract(query_lower, self.usda_names, scorer=fuzz.WRatio, limit=top_k)
            for name, score, _ in matches:
                if score >= 50:
                    for item in self.usda_index: 
                        if item["name"] == name: 
                            results.append((item["data"], item["source"], score / 100.0))
                            break
        
        return results[:top_k]