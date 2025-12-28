import { Component, OnInit, signal, ViewChild, ElementRef, AfterViewChecked } from '@angular/core';
import { Router } from '@angular/router';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ChatService } from '../../core/services/chat.service';
import { CountryService } from '../../core/services/country.service';
import { ChatMessage } from '../../core/models/message.model';
import { Country } from '../../core/models/country.model';

@Component({
  selector:  'app-chat',
  standalone:  true,
  imports: [CommonModule, FormsModule],
  templateUrl: './chat.component.html',
  styleUrl: './chat.component.scss'
})
export class ChatComponent implements OnInit, AfterViewChecked {
  @ViewChild('messagesContainer') private messagesContainer!:  ElementRef;
  @ViewChild('messageInput') private messageInput!: ElementRef;

  country:  Country | null = null;
  messageText = '';
  messages = signal<ChatMessage[]>([]);
  isLoading = signal<boolean>(false);

  quickQueries = [
    'Shawarma calories',
    'Falafel',
    'Kushari',
    'fajita',
    'Kabsa'
  ];

  constructor(
    private chatService: ChatService,
    private countryService: CountryService,
    private router: Router
  ) {}

  ngOnInit(): void {
    this.country = this.countryService.getSelectedCountry();
    if (! this.country) {
      this.router.navigate(['/select-country']);
      return;
    }

    // Add welcome message
    this.messages.set([{
      role: 'assistant',
      content:  this.getWelcomeMessage(),
      timestamp: new Date()
    }]);
  }

  ngAfterViewChecked(): void {
    this.scrollToBottom();
  }

  getWelcomeMessage(): string {
    const countryName = this.country?.name_en || 'your country';
    return `🍽️ Welcome!  I'm your Arabic Food Calorie Calculator for ${countryName} ${this.country?.flag_emoji || ''}\n\nAsk me about any dish or ingredient, and I'll tell you the calories!\n\nExamples:\n• "How many calories in shawarma? "\n• "Falafel without tahini"\n• "200g grilled chicken"`;
  }

  sendMessage(): void {
    if (! this.messageText.trim() || this.isLoading()) return;

    const message = this.messageText.trim();
    this.messageText = '';
    this.isLoading.set(true);

    // Add user message
    this.messages.update(msgs => [...msgs, {
      role: 'user' as const,
      content: message,
      timestamp: new Date()
    }]);

    // Send to backend
    this.chatService.sendMessage(message, this.country?.code || 'lebanon').subscribe({
      next: (response) => {
        this.messages.update(msgs => [...msgs, {
          role: 'assistant' as const,
          content: response.message,
          timestamp:  new Date(),
          calorieResult:  response.calorie_result || undefined
        }]);
        this.isLoading.set(false);
      },
      error: (err) => {
        console.error('Chat error:', err);
        this.messages.update(msgs => [...msgs, {
          role: 'assistant' as const,
          content: '❌ Sorry, I encountered an error. Please try again.',
          timestamp: new Date()
        }]);
        this.isLoading.set(false);
      }
    });
  }

  sendQuickQuery(query: string): void {
    this.messageText = query;
    this.sendMessage();
  }

  handleKeyPress(event:  KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.sendMessage();
    }
  }

  scrollToBottom(): void {
    try {
      if (this.messagesContainer) {
        this.messagesContainer.nativeElement.scrollTop = 
          this.messagesContainer.nativeElement.scrollHeight;
      }
    } catch(err) {}
  }

  changeCountry(): void {
    this.countryService.clearSelectedCountry();
    this.router.navigate(['/select-country']);
  }

  clearChat(): void {
    this.chatService.clearChat();
    this.messages.set([{
      role: 'assistant',
      content: this.getWelcomeMessage(),
      timestamp: new Date()
    }]);
  }

  formatMessage(content: string): string {
    // Convert markdown-like formatting to HTML
    return content
      .replace(/\*\*(.*? )\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/\n/g, '<br>');
  }
}