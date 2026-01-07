
// This service handles communication with the Ollama API

import { UniqueOption } from "recharts/types/util/payload/getUniqPayload";

export interface OllamaResponse {
  model: string;
  created_at: string;
  response: string;
  done: boolean;
}

export interface OllamaGenerationOptions {
  model: string;
  prompt: string;
  system?: string;
  temperature?: number;
  max_tokens?: number;
}

const defaultOptions = {
  temperature: 0.7,
  max_tokens: 500,
};

export class OllamaService {
  private baseUrl: string;
  
  constructor(baseUrl = 'http://127.0.0.1:11434') {
    this.baseUrl = baseUrl;
  }

  async generateStory(options: OllamaGenerationOptions): Promise<string> {
    try {
      const response = await fetch(`${this.baseUrl}/api/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: options.model,
          prompt: options.prompt,
          system: options.system || "You are an interactive storyteller. Provide engaging, rich narratives based on the user's input.",
          options: {
            temperature: options.temperature || defaultOptions.temperature,
            max_tokens: options.max_tokens || defaultOptions.max_tokens,
          },
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Ollama API error: ${response.status} - ${errorText}`);
      }

      // Ollama API returns line-by-line JSON objects
      const reader = response.body?.getReader();
      if (!reader) throw new Error("Response body is null");

      let result = '';
      let done = false;

      while (!done) {
        const { done: readerDone, value } = await reader.read();
        
        if (readerDone) {
          done = true;
          continue;
        }

        // Convert the Uint8Array to text
        const chunk = new TextDecoder().decode(value);
        
        // Each line is a JSON object
        const lines = chunk.split('\n').filter(line => line.trim() !== '');
        
        for (const line of lines) {
          try {
            const parsedLine = JSON.parse(line) as OllamaResponse;
            result += parsedLine.response;
            
            if (parsedLine.done) {
              done = true;
              break;
            }
          } catch (e) {
            console.error('Error parsing JSON from Ollama:', e);
          }
        }
      }

      return result;
    } catch (error) {
      console.error('Error generating story with Ollama:', error);
      return 'Unable to connect to the Ollama service. Please make sure Ollama is running on your device.';
    }
  }

  async getAvailableModels(): Promise<string[]> {
    try {
      const response = await fetch(`${this.baseUrl}/api/tags`, {
        method: 'GET',
      });

      if (!response.ok) {
        throw new Error(`Failed to get models: ${response.statusText}`);
      }

      const data = await response.json();
      return data.models?.map((model: UniqueOption<{ name: string }>) => (model as { name: string }).name) || [];
    } catch (error) {
      console.error('Error fetching available models:', error);
      return [];
    }
  }
}

export const ollamaService = new OllamaService();
