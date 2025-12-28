import { Injectable, signal } from '@angular/core';
import { ApiService } from './api.service';
import { Country } from '../models/country.model';
import { Observable } from 'rxjs';

@Injectable({
  providedIn:  'root'
})
export class CountryService {
  private selectedCountry = signal<Country | null>(null);

  constructor(private api: ApiService) {}

  getCountries(): Observable<Country[]> {
    return this.api.get<Country[]>('/countries/');
  }

  setSelectedCountry(country:  Country): void {
    this.selectedCountry.set(country);
    localStorage.setItem('selectedCountry', JSON.stringify(country));
  }

  getSelectedCountry(): Country | null {
    if (! this.selectedCountry()) {
      const stored = localStorage.getItem('selectedCountry');
      if (stored) {
        this.selectedCountry.set(JSON.parse(stored));
      }
    }
    return this.selectedCountry();
  }

  clearSelectedCountry(): void {
    this.selectedCountry.set(null);
    localStorage.removeItem('selectedCountry');
  }
}