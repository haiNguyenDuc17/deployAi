/**
 * Simple HTTP client for making API requests
 */

export interface ApiResponse<T = any> {
  data: T;
  status: number;
  statusText: string;
}

export interface RequestConfig {
  headers?: Record<string, string>;
  timeout?: number;
}

let baseUrl = '';

export function setBaseUrl(url: string): void {
  baseUrl = url;
}

export function getBaseUrl(): string {
  return baseUrl;
}

export async function create<T = any>(
  url: string,
  data?: any,
  config?: RequestConfig
): Promise<ApiResponse<T>> {
  const fullUrl = baseUrl ? `${baseUrl}/${url}` : url;
  
  const defaultHeaders: Record<string, string> = {
    'Content-Type': 'application/json',
  };

  const headers = {
    ...defaultHeaders,
    ...config?.headers,
  };

  try {
    const response = await fetch(fullUrl, {
      method: 'POST',
      headers,
      body: data ? JSON.stringify(data) : undefined,
    });

    const responseData = await response.json();

    return {
      data: responseData,
      status: response.status,
      statusText: response.statusText,
    };
  } catch (error) {
    throw {
      response: {
        status: 0,
        statusText: 'Network Error',
      },
      message: error instanceof Error ? error.message : 'Unknown error',
    };
  }
}

export async function get<T = any>(
  url: string,
  config?: RequestConfig
): Promise<ApiResponse<T>> {
  const fullUrl = baseUrl ? `${baseUrl}/${url}` : url;
  
  const defaultHeaders: Record<string, string> = {};

  const headers = {
    ...defaultHeaders,
    ...config?.headers,
  };

  try {
    const response = await fetch(fullUrl, {
      method: 'GET',
      headers,
    });

    const responseData = await response.json();

    return {
      data: responseData,
      status: response.status,
      statusText: response.statusText,
    };
  } catch (error) {
    throw {
      response: {
        status: 0,
        statusText: 'Network Error',
      },
      message: error instanceof Error ? error.message : 'Unknown error',
    };
  }
}
