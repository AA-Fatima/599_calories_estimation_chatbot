from typing import Dict, Any, List
from datetime import datetime
import json
import csv
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class MissingDishLogger:
    """Log missing dishes for later review"""
    
    def __init__(self, log_file: str = "missing_dishes.csv", json_file: str = "missing_dishes.json"):
        self.log_file = Path(log_file)
        self.json_file = json_file
        self.missing_dishes = self._load_json_logs()
        self._ensure_csv_file_exists()
    
    def _ensure_csv_file_exists(self):
        """Create CSV log file if doesn't exist"""
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'dish_name', 'user_query', 'country', 'gpt_ingredients'])
    
    def _load_json_logs(self) -> List[Dict]: 
        """Load existing logs from JSON (backward compatibility)"""
        if os.path.exists(self.json_file):
            try:
                with open(self.json_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save_json_logs(self):
        """Save logs to JSON file (backward compatibility)"""
        try:
            with open(self.json_file, 'w') as f:
                json.dump(self.missing_dishes, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving missing dishes log: {e}")
    
    def log(self, query: str, country: str, fallback_response: Dict = None, user_ingredients: List[str] = None, gpt_result: Dict = None):
        """Log a missing dish to both CSV and JSON"""
        # Log to CSV (new format)
        try:
            with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    query,
                    query,  # user_query (same as dish_name for now)
                    country,
                    json.dumps(gpt_result.get('ingredients', []) if gpt_result else [])
                ])
            logger.info(f"📝 Logged missing dish to CSV: {query}")
        except Exception as e:
            logger.error(f"Failed to log missing dish to CSV: {e}")
        
        # Log to JSON (backward compatibility)
        entry = {
            "query": query,
            "country": country,
            "timestamp": datetime.utcnow().isoformat(),
            "fallback_response": fallback_response,
            "user_provided_ingredients": user_ingredients,
            "gpt_result": gpt_result,
            "resolved":  False
        }
        
        # Check if already logged
        for dish in self.missing_dishes: 
            if dish["query"].lower() == query.lower() and dish["country"] == country: 
                logger.info(f"Dish already logged in JSON: {query}")
                return
        
        self.missing_dishes.append(entry)
        self._save_json_logs()
        logger.info(f"Logged missing dish to JSON: {query} ({country})")
    
    def get_missing_dishes(self) -> List[Dict]:
        """Read all missing dishes from CSV"""
        dishes = []
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    dishes.append(row)
        except Exception as e:
            logger.error(f"Failed to read missing dishes: {e}")
        return dishes
    
    def get_unresolved(self) -> List[Dict]:
        """Get all unresolved missing dishes from JSON"""
        return [d for d in self.missing_dishes if not d["resolved"]]
    
    def mark_resolved(self, query: str, country:  str):
        """Mark a dish as resolved in JSON"""
        for dish in self.missing_dishes:
            if dish["query"].lower() == query.lower() and dish["country"] == country: 
                dish["resolved"] = True
        self._save_json_logs()