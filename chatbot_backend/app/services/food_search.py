from typing import List, Dict, Any, Tuple
from rapidfuzz import fuzz, process
import logging

logger = logging.getLogger(__name__)


class FoodSearchService:
    """Food search using fuzzy matching - NO semantic search to avoid wrong results"""
    
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
    
    def search(self, query: str, country: str = "", top_k: int = 5) -> List[Tuple[Dict, str, float]]: 
        """
        Search for food with priority: 
        1. Search dishes first (exact + fuzzy)
        2. If no good dish match, search USDA
        3. If nothing found, return empty
        """
        query_lower = query.lower().strip()
        country_lower = country.lower().strip() if country else ""
        
        if not query_lower: 
            return []
        
        logger.info(f"Searching for: '{query_lower}' (country: {country_lower})")
        
        # ========================================
        # STEP 1: EXACT MATCH (dishes and USDA)
        # ========================================
        for item in self.search_index:
            if query_lower == item["name"]:
                if item["source"] == "dishes": 
                    if not country_lower or item["country"] == country_lower: 
                        logger.info(f"✅ Exact dish match: {item['original_name']}")
                        return [(item["data"], item["source"], 1.0)]
                else:
                    logger.info(f"✅ Exact USDA match: {item['original_name']}")
                    return [(item["data"], item["source"], 1.0)]
        
        results = []
        
        # ========================================
        # STEP 2: SEARCH DISHES FIRST (fuzzy)
        # ========================================
        logger.info("🔍 Searching in dishes...")
        
        # 2a. Search dishes in selected country
        if self.dish_names and country_lower:
            country_dishes = [(item["name"], item) for item in self.dish_index if item["country"] == country_lower]
            if country_dishes:
                country_dish_names = [d[0] for d in country_dishes]
                matches = process.extract(query_lower, country_dish_names, scorer=fuzz.WRatio, limit=3)
                
                for name, score, _ in matches:
                    if score >= 70:
                        for dish_name, item in country_dishes:
                            if dish_name == name:
                                results.append((item["data"], item["source"], score / 100.0))
                                break
        
        # 2b. If no good country match, search ALL dishes
        if not results or (results and results[0][2] < 0.8):
            matches = process.extract(query_lower, self.dish_names, scorer=fuzz.WRatio, limit=3)
            
            for name, score, _ in matches:
                if score >= 70:
                    for item in self.dish_index:
                        if item["name"] == name:
                            final_score = score / 100.0
                            # Penalize if not from selected country
                            if country_lower and item["country"] != country_lower:
                                final_score *= 0.9
                            results.append((item["data"], item["source"], final_score))
                            break
        
        # If we found good dish matches, return them
        if results: 
            results.sort(key=lambda x: x[2], reverse=True)
            # Remove duplicates
            seen = set()
            unique = []
            for data, source, conf in results:
                name = data.get("dish_name", "").lower()
                if name and name not in seen:
                    seen.add(name)
                    unique.append((data, source, conf))
            
            if unique and unique[0][2] >= 0.70:  # Good confidence threshold
                logger.info(f"✅ Found dish:  {unique[0][0].get('dish_name')} (confidence: {unique[0][2]:.2f})")
                return unique[: top_k]
            else:
                logger.info(f"⚠️ Dish matches too weak (best: {unique[0][2]:.2f}), trying USDA...")
        else:
            logger.info("❌ No dishes found, trying USDA...")
        
        # ========================================
        # STEP 3: SEARCH USDA (only if dishes failed)
        # ========================================
        logger.info("🔍 Searching in USDA databases...")
        
        usda_results = []
        
        # 3a. Word-level matching (best for ingredients)
        word_matches = []
        for item in self.usda_index:
            item_words = item["name"].replace(',', ' ').replace('-', ' ').replace('(', ' ').replace(')', ' ').lower().split()
            item_words = [w for w in item_words if w]  # Remove empty
            
            # Check if query matches the FIRST word (most relevant)
            if item_words and item_words[0] == query_lower:
                word_matches.append((item, 0.95))
            # Check if query is any complete word in the name
            elif query_lower in item_words:
                word_matches.append((item, 0.85))
            # Check if query is at start of first word (prefix match)
            elif item_words and item_words[0].startswith(query_lower) and len(query_lower) >= 4:
                word_matches.append((item, 0.75))
        
        # Sort by score and add to results
        word_matches.sort(key=lambda x: x[1], reverse=True)
        for item, score in word_matches[: 5]: 
            usda_results.append((item["data"], item["source"], score))
        
        # 3b. If no word matches, try fuzzy matching
        if not word_matches: 
            usda_fuzzy = process.extract(query_lower, self.usda_names, scorer=fuzz.WRatio, limit=3)
            for name, score, _ in usda_fuzzy: 
                if score >= 60:  # Lower threshold for USDA
                    for item in self.usda_index:
                        if item["name"] == name:
                            usda_results.append((item["data"], item["source"], score / 100.0))
                            break
        
        # If USDA results found, return them
        if usda_results: 
            usda_results.sort(key=lambda x: x[2], reverse=True)
            
            # Remove duplicates
            seen = set()
            unique_usda = []
            for data, source, conf in usda_results:
                name = data.get("description", "").lower()
                if name and name not in seen: 
                    seen.add(name)
                    unique_usda.append((data, source, conf))
            
            if unique_usda: 
                logger.info(f"✅ Found USDA ingredient: {unique_usda[0][0].get('description')} (confidence: {unique_usda[0][2]:.2f})")
                return unique_usda[:top_k]
        
        # ========================================
        # STEP 4: NOTHING FOUND
        # ========================================
        logger.info(f"❌ No results found for:  '{query_lower}'")
        return []

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