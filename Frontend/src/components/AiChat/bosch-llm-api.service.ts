/**
 * Service that connects to the Bosch LLM Farm API. Can and should be extended.
 */

import { create, setBaseUrl } from "../../apiClient/apiClient";
import { environment } from "../../environments/environment";


export interface Message {
  role: string;
  content: string;
}

export interface ModelOption {
  label: string;
  value: string;
}

export class BoschLlmApiService {
  private readonly API_KEY = environment.API_KEY;
  private model = '';
  
  constructor() {
    setBaseUrl('/api');
  }
  
  private modelUrlPaths: { [key: string]: string } = {
    'gemini-1.5-flash': 'openai/deployments/google-gemini-1-5-flash/chat/completions',
    'gemini-2.0-flash-lite': 'openai/deployments/google-gemini-2-0-flash-lite/chat/completions',
    'gpt-4o-mini': 'openai/deployments/askbosch-prod-farm-openai-gpt-4o-mini-2024-07-18/chat/completions?api-version=2024-08-01-preview',
  };

  // Available models
  availableModels: ModelOption[] = [
    { label: 'Gemini 1.5 Flash', value: 'gemini-1.5-flash' },
    { label: 'Gemini 2.0 Flash Lite', value: 'gemini-2.0-flash-lite' },
    { label: 'GPT-4o Mini', value: 'gpt-4o-mini' },
  ];

  /**
   * Main method to fire a call to the LLM Farm API via HTTP POST method
   *
   * @param messages Array of user messages and AI answers
   * @returns A Promise that resolves to the API response
   */
  public async get(messages: Message[]): Promise<any> {
    if (!this.model) {
      throw new Error('Model not selected');
    }

    const url = this.modelUrlPaths[this.model];
    const body = {
      messages: messages,
      model: this.model
    };

    const headers: Record<string, string> = {
      'genaiplatform-farm-subscription-key': this.API_KEY,
      'Content-Type': 'application/json',
      'model_name': this.model
    };

    try {
      const response = await create<any>(url, body, { headers });
      return response.data;
    } catch (error) {
      const mapped = this.mapHttpError(error);
      console.error('API Error Details (mapped):', mapped);
      throw mapped;
    }
  }

  /**
   * The Bosch LLM farm offers a lot of different models so you first need to select one ;-)
   * @param modelName The model name must match one of the keys in this.modelUrlPaths array.
   * @returns True if model path is available otherwise false
   */
  isModelSupported(modelName: string): boolean {
    return !!this.modelUrlPaths[modelName];
  }

  selectModel(modelName: string): boolean {
    const supported = this.isModelSupported(modelName);
    if (supported) {
      this.model = modelName;
      return true;
    }
    return false;
  }

  getCurrentModel(): string {
    return this.model;
  }

  private mapHttpError(error: any): { code: string; status: number; message: string; raw?: any } {
    const status = error?.response?.status ?? 0;
    const base = {
      raw: error,
    } as { raw?: any };

    switch (status) {
      case 400:
        return { code: 'BAD_REQUEST', status, message: 'Bad Request: Please check your input and try again.', ...base };
      case 401:
        return { code: 'UNAUTHORIZED', status, message: 'Unauthorized: Please check your API key.', ...base };
      case 403:
        return { code: 'FORBIDDEN', status, message: 'Forbidden: Access denied.', ...base };
      case 429:
        return { code: 'RATE_LIMITED', status, message: 'Rate limit exceeded: Please wait and try again.', ...base };
      case 500:
        return { code: 'SERVER_ERROR', status, message: 'Server error: Please try again later.', ...base };
      case 504:
        return { code: 'GATEWAY_TIMEOUT', status, message: 'Gateway timeout: The request took too long.', ...base };
      default:
        return { code: 'UNKNOWN', status, message: error?.message || 'Unknown error', ...base };
    }
  }
}

// Export a singleton instance
export const boschLlmApiService = new BoschLlmApiService();
