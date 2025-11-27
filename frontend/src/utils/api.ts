/**
 * API client utilities with centralized error handling and configuration.
 */
import axios, { AxiosError } from 'axios';
import type { AxiosInstance, AxiosRequestConfig } from 'axios';
import { API_BASE_URL } from '../config/constants';

/**
 * Custom error class for API errors.
 */
export class ApiError extends Error {
  constructor(
    message: string,
    public statusCode?: number,
    public details?: any
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

/**
 * Create a configured axios instance.
 */
const createApiClient = (): AxiosInstance => {
  const client = axios.create({
    baseURL: API_BASE_URL,
    timeout: 120000, // 2 minutes for long-running operations
    headers: {
      'Content-Type': 'application/json',
    },
  });

  // Request interceptor
  client.interceptors.request.use(
    (config) => {
      // Add any auth tokens or custom headers here if needed
      return config;
    },
    (error) => {
      return Promise.reject(error);
    }
  );

  // Response interceptor
  client.interceptors.response.use(
    (response) => response,
    (error: AxiosError) => {
      // Handle errors globally
      if (error.response) {
        // Server responded with error status
        const message = (error.response.data as any)?.detail || error.message;
        throw new ApiError(
          message,
          error.response.status,
          error.response.data
        );
      } else if (error.request) {
        // Request made but no response
        throw new ApiError(
          'サーバーに接続できません。ネットワーク接続を確認してください。',
          0
        );
      } else {
        // Error in request setup
        throw new ApiError(error.message);
      }
    }
  );

  return client;
};

/**
 * Singleton API client instance.
 */
export const apiClient = createApiClient();

/**
 * Type-safe API request wrapper.
 */
export async function apiRequest<T>(
  config: AxiosRequestConfig
): Promise<T> {
  try {
    const response = await apiClient.request<T>(config);
    return response.data;
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    throw new ApiError('予期しないエラーが発生しました。');
  }
}

/**
 * Convenience methods for common HTTP verbs.
 */
export const api = {
  get: <T>(url: string, config?: AxiosRequestConfig) =>
    apiRequest<T>({ ...config, method: 'GET', url }),

  post: <T>(url: string, data?: any, config?: AxiosRequestConfig) =>
    apiRequest<T>({ ...config, method: 'POST', url, data }),

  put: <T>(url: string, data?: any, config?: AxiosRequestConfig) =>
    apiRequest<T>({ ...config, method: 'PUT', url, data }),

  delete: <T>(url: string, config?: AxiosRequestConfig) =>
    apiRequest<T>({ ...config, method: 'DELETE', url }),

  patch: <T>(url: string, data?: any, config?: AxiosRequestConfig) =>
    apiRequest<T>({ ...config, method: 'PATCH', url, data }),
};

/**
 * Format error message for display to user.
 */
export function formatErrorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    return error.message;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return 'エラーが発生しました';
}

/**
 * Check if error is a network error.
 */
export function isNetworkError(error: unknown): boolean {
  return error instanceof ApiError && error.statusCode === 0;
}

/**
 * Check if error is a not found error.
 */
export function isNotFoundError(error: unknown): boolean {
  return error instanceof ApiError && error.statusCode === 404;
}

/**
 * Check if error is a validation error.
 */
export function isValidationError(error: unknown): boolean {
  return error instanceof ApiError && error.statusCode === 400;
}
