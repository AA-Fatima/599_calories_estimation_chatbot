import { Component } from '@angular/core';
import { Router } from '@angular/router';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-landing',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './landing.component.html',
  styleUrl: './landing.component.scss'
})
export class LandingComponent {
  features = [
    { icon: '🍽️', title: 'Arabic Cuisine', description: 'Specialized in dishes from Lebanon, Egypt, Saudi Arabia, and more' }, 
   // { icon: '🤖', title: 'Smart AI', description: 'Understands English, Arabic, and Franco-Arabic (Arabizi)' },
    
    { icon: '✏️', title: 'Customizable', description: 'Modify dishes - add or remove ingredients easily' },
    { icon: '📊', title: 'Accurate Data', description: 'Based on USDA database and verified nutritional information' }
  ];

  constructor(private router: Router) {}

  getStarted(): void {
    this.router.navigate(['/select-country']);
  }
}