import { Component, OnInit, signal } from '@angular/core';
import { Router } from '@angular/router';
import { CommonModule } from '@angular/common';
import { CountryService } from '../../core/services/country.service';
import { Country } from '../../core/models/country.model';

@Component({
  selector: 'app-country-select',
  standalone:  true,
  imports: [CommonModule],
  templateUrl:  './country-select.component.html',
  styleUrl: './country-select.component.scss'
})
export class CountrySelectComponent implements OnInit {
  countries = signal<Country[]>([]);
  selectedCountry = signal<Country | null>(null);
  isLoading = signal<boolean>(true);

  constructor(
    private countryService: CountryService,
    private router: Router
  ) {}

  ngOnInit(): void {
    this.loadCountries();
  }

  loadCountries(): void {
    this.countryService.getCountries().subscribe({
      next: (countries) => {
        this.countries.set(countries);
        this.isLoading.set(false);
      },
      error: (err) => {
        console.error('Error loading countries:', err);
        // Fallback to hardcoded countries
        this.countries.set([
          { code: 'lebanon', name_en: 'Lebanon', name_ar:  'لبنان', flag_emoji: '🇱🇧' },
          { code: 'egypt', name_en:  'Egypt', name_ar: 'مصر', flag_emoji: '🇪🇬' },
          { code: 'saudi', name_en: 'Saudi Arabia', name_ar: 'السعودية', flag_emoji: '🇸🇦' },
          { code:  'syria', name_en:  'Syria', name_ar: 'سوريا', flag_emoji: '🇸🇾' },
          { code: 'iraq', name_en:  'Iraq', name_ar: 'العراق', flag_emoji: '🇮🇶' },
          { code: 'jordan', name_en:  'Jordan', name_ar: 'الأردن', flag_emoji: '🇯🇴' },
          { code: 'palestine', name_en:  'Palestine', name_ar: 'فلسطين', flag_emoji: '🇵🇸' },
          { code: 'morocco', name_en:  'Morocco', name_ar: 'المغرب', flag_emoji: '🇲🇦' },
          { code:  'tunisia', name_en: 'Tunisia', name_ar: 'تونس', flag_emoji: '🇹🇳' },
          { code: 'algeria', name_en:  'Algeria', name_ar: 'الجزائر', flag_emoji:  '🇩🇿' },
        ]);
        this.isLoading.set(false);
      }
    });
  }

  selectCountry(country: Country): void {
    this.selectedCountry.set(country);
  }

  startChat(): void {
    const country = this.selectedCountry();
    if (country) {
      this.countryService.setSelectedCountry(country);
      this.router.navigate(['/chat']);
    }
  }

  goBack(): void {
    this.router.navigate(['/']);
  }
}