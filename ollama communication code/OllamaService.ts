import { AIProviderService, AIMessage, AIResponse } from './AIProviderService';

export class OllamaService extends AIProviderService {
  private baseUrl: string;
  private model: string;

  constructor(baseUrl: string = 'http://localhost:11434', model: string = 'huihui_ai/magistral-abliterated:24b') {
    super();
    this.baseUrl = baseUrl;
    this.model = model;
  }

  async generateResponse(messages: AIMessage[], systemPrompt: string): Promise<AIResponse> {
    try {
      const formattedMessages = [
        { role: 'system', content: systemPrompt },
        ...messages
      ];

      const response = await fetch(`${this.baseUrl}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: this.model,
          messages: formattedMessages,
          stream: false,
          options: {
            temperature: 0.95,
            top_p: 0.94,
            num_predict: 150
          }
        }),
      });

      if (!response.ok) {
        throw new Error(`Ollama request failed: ${response.statusText}`);
      }

      const data = await response.json();
      return {
        content: data.message?.content || "Sorry, I couldn't generate a response."
      };
    } catch (error) {
      console.error('Ollama error:', error);
      return {
        content: "I'm having trouble connecting to the local AI. Is Ollama running?",
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  async isAvailable(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/api/tags`, {
        method: 'GET',
        signal: AbortSignal.timeout(3000)
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  getName(): string {
    return 'Ollama';
  }

  async getAvailableModels(): Promise<string[]> {
    try {
      const response = await fetch(`${this.baseUrl}/api/tags`);
      if (!response.ok) return [];
      
      const data = await response.json();
      return data.models?.map((model: { name: string }) => model.name) || [];
    } catch {
      return [];
    }
  }

  setModel(model: string) {
    this.model = model;
  }

  setBaseUrl(url: string) {
    this.baseUrl = url;
  }
}