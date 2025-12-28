import { Injectable, signal } from '@angular/core';
import { ApiService } from './api.service';
import { ChatMessage, ChatRequest, ChatResponse } from '../models/message.model';
import { Observable, tap } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ChatService {
  private sessionId = signal<string>('');
  private messages = signal<ChatMessage[]>([]);
  private isLoading = signal<boolean>(false);

  constructor(private api: ApiService) {}

  get sessionId$() { return this.sessionId; }
  get messages$() { return this.messages; }
  get isLoading$() { return this.isLoading; }

  createSession(country: string): Observable<{session_id: string, country: string}> {
    return this.api.post<{session_id: string, country:  string}>('/chat/session', null, ).pipe(
      tap(response => {
        this.sessionId.set(response.session_id);
        this.messages.set([]);
      })
    );
  }

  sendMessage(message: string, country:  string): Observable<ChatResponse> {
    this.isLoading.set(true);
    
    // Add user message immediately
    const userMessage: ChatMessage = {
      role: 'user',
      content: message,
      timestamp: new Date()
    };
    this.messages.update(msgs => [...msgs, userMessage]);

    const request: ChatRequest = {
      message,
      session_id: this.sessionId() || 'new',
      country
    };

    return this.api.post<ChatResponse>('/chat/message', request).pipe(
      tap(response => {
        // Update session ID if new
        if (response.session_id) {
          this.sessionId.set(response.session_id);
        }

        // Add assistant response
        const assistantMessage: ChatMessage = {
          role: 'assistant',
          content:  response.message,
          timestamp:  new Date(),
          calorieResult: response.calorie_result || undefined
        };
        this.messages.update(msgs => [...msgs, assistantMessage]);
        this.isLoading.set(false);
      })
    );
  }

  clearChat(): void {
    this.messages.set([]);
    this.sessionId.set('');
  }
}