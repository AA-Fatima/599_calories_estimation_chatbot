export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp:  Date;
  calorieResult?: CalorieResult;
}

export interface CalorieResult {
  food_name: string;
  original_query: string;
  total_calories: number;
  weight_g: number;
  ingredients?:  Ingredient[];
  modifications?: string[];
  source:  string;
  confidence: number;
  is_approximate: boolean;
  country?:  string;
}

export interface Ingredient {
  usda_fdc_id?:  number;
  name:  string;
  weight_g: number;
  calories:  number;
}

export interface ChatRequest {
  message:  string;
  session_id:  string;
  country: string;
}

export interface ChatResponse {
  message: string;
  calorie_result?: CalorieResult;
  follow_up_questions?: string[];
  requires_clarification:  boolean;
  session_id: string;
}