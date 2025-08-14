const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface Session {
  session_id: string;
  created_at: string;
  updated_at: string;
  metadata?: Record<string, any>;
}

export interface Message {
  id: string;
  session_id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  created_at: string;
  model?: string;
  function_calls?: any[];
}

export interface Model {
  id: string;
  name: string;
  provider: string;
  max_tokens: number;
  supports_functions: boolean;
  supports_vision: boolean;
}

export interface ChatRequest {
  message: string;
  model?: string;
  temperature?: number;
  max_tokens?: number;
  stream?: boolean;
}

class ApiClient {
  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${API_URL}${endpoint}`;
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(error.detail || `HTTP error! status: ${response.status}`);
    }

    return response.json();
  }

  async createSession(metadata?: Record<string, any>): Promise<Session> {
    return this.request<Session>('/api/v1/chat/sessions', {
      method: 'POST',
      body: JSON.stringify({ metadata: metadata || {} }),
    });
  }

  async sendMessage(
    sessionId: string,
    request: ChatRequest
  ): Promise<Message> {
    return this.request<Message>('/api/v1/chat/messages', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        ...request,
      }),
    });
  }

  async getModels(): Promise<Model[]> {
    return this.request<Model[]>('/api/v1/models');
  }

  async getSessionHistory(sessionId: string): Promise<Message[]> {
    return this.request<Message[]>(`/api/v1/chat/sessions/${sessionId}/messages`);
  }
}

export const api = new ApiClient();