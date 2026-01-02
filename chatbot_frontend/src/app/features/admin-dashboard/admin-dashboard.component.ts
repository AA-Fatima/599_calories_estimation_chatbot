import { Component, OnInit, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { AdminService, Dish, MissingDish } from '../../core/services/admin.service';

@Component({
  selector: 'app-admin-dashboard',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './admin-dashboard.component.html',
  styleUrl: './admin-dashboard.component.scss'
})
export class AdminDashboardComponent implements OnInit {
  dishes = signal<Dish[]>([]);
  missingDishes = signal<MissingDish[]>([]);
  showAddModal = signal<boolean>(false);
  showEditModal = signal<boolean>(false);
  currentDish: Dish = this.getEmptyDish();
  isLoading = signal<boolean>(false);
  activeTab = signal<'dishes' | 'missing'>('dishes');

  constructor(private adminService: AdminService) {}

  ngOnInit() {
    this.loadDishes();
    this.loadMissingDishes();
  }

  loadDishes() {
    this.isLoading.set(true);
    this.adminService.getDishes().subscribe({
      next: (dishes) => {
        this.dishes.set(dishes);
        this.isLoading.set(false);
      },
      error: (error) => {
        console.error('Error loading dishes:', error);
        this.isLoading.set(false);
      }
    });
  }

  loadMissingDishes() {
    this.adminService.getMissingDishes().subscribe({
      next: (missing) => {
        this.missingDishes.set(missing);
      },
      error: (error) => {
        console.error('Error loading missing dishes:', error);
      }
    });
  }

  openAddModal() {
    this.currentDish = this.getEmptyDish();
    this.showAddModal.set(true);
  }

  openEditModal(dish: Dish) {
    this.currentDish = { ...dish };
    this.showEditModal.set(true);
  }

  closeModal() {
    this.showAddModal.set(false);
    this.showEditModal.set(false);
    this.currentDish = this.getEmptyDish();
  }

  saveDish() {
    if (!this.currentDish.dish_name || !this.currentDish.country) {
      alert('Please fill in required fields');
      return;
    }

    if (this.showEditModal() && this.currentDish.id) {
      // Update existing dish
      this.adminService.updateDish(this.currentDish.id, this.currentDish).subscribe({
        next: () => {
          this.loadDishes();
          this.closeModal();
        },
        error: (error) => {
          console.error('Error updating dish:', error);
          alert('Failed to update dish');
        }
      });
    } else {
      // Add new dish
      this.adminService.addDish(this.currentDish).subscribe({
        next: () => {
          this.loadDishes();
          this.closeModal();
        },
        error: (error) => {
          console.error('Error adding dish:', error);
          alert('Failed to add dish');
        }
      });
    }
  }

  deleteDish(dish: Dish) {
    if (!dish.id) return;
    
    if (confirm(`Are you sure you want to delete "${dish.dish_name}"?`)) {
      this.adminService.deleteDish(dish.id).subscribe({
        next: () => {
          this.loadDishes();
        },
        error: (error) => {
          console.error('Error deleting dish:', error);
          alert('Failed to delete dish');
        }
      });
    }
  }

  setActiveTab(tab: 'dishes' | 'missing') {
    this.activeTab.set(tab);
  }

  private getEmptyDish(): Dish {
    return {
      dish_name: '',
      country: 'lebanon',
      total_calories: 0,
      weight_g: 0,
      ingredients: []
    };
  }
}
