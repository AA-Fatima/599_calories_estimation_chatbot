import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../../environments/environment';

export interface Dish {
  id?: number;
  dish_name: string;
  country: string;
  total_calories?: number;
  weight_g?: number;
  ingredients?: any[];
}

export interface MissingDish {
  timestamp: string;
  dish_name: string;
  user_query: string;
  country: string;
  gpt_ingredients: string;
}

@Injectable({
  providedIn: 'root'
})
export class AdminService {
  private apiUrl = environment.apiUrl;

  constructor(private http: HttpClient) { }

  getDishes(): Observable<Dish[]> {
    return this.http.get<Dish[]>(`${this.apiUrl}/admin/dishes`);
  }

  getMissingDishes(): Observable<MissingDish[]> {
    return this.http.get<MissingDish[]>(`${this.apiUrl}/admin/missing-dishes`);
  }

  addDish(dish: Dish): Observable<Dish> {
    return this.http.post<Dish>(`${this.apiUrl}/admin/dishes`, dish);
  }

  updateDish(id: number, dish: Dish): Observable<Dish> {
    return this.http.put<Dish>(`${this.apiUrl}/admin/dishes/${id}`, dish);
  }

  deleteDish(id: number): Observable<void> {
    return this.http.delete<void>(`${this.apiUrl}/admin/dishes/${id}`);
  }
}
