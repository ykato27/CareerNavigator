import { handleBrowserApiRequest } from '../cloudflare/browserApi';

export interface AxiosRequestConfig<T = any> {
  url?: string;
  method?: string;
  data?: T;
  params?: Record<string, any>;
  headers?: Record<string, string>;
  timeout?: number;
  baseURL?: string;
}

export interface AxiosResponse<T = any> {
  data: T;
  status: number;
  statusText: string;
  config: AxiosRequestConfig;
  headers: Record<string, string>;
}

export class AxiosError<T = any> extends Error {
  response?: AxiosResponse<T>;
  request?: AxiosRequestConfig;

  constructor(message: string, response?: AxiosResponse<T>, request?: AxiosRequestConfig) {
    super(message);
    this.name = 'AxiosError';
    this.response = response;
    this.request = request;
  }
}

type Fulfilled<T> = (value: T) => T | Promise<T>;
type Rejected = (reason: any) => any;

class InterceptorManager<T> {
  private handlers: Array<{ fulfilled: Fulfilled<T>; rejected?: Rejected }> = [];

  use(fulfilled: Fulfilled<T>, rejected?: Rejected): number {
    this.handlers.push({ fulfilled, rejected });
    return this.handlers.length - 1;
  }

  async run(value: T): Promise<T> {
    let current = value;
    for (const handler of this.handlers) {
      current = await handler.fulfilled(current);
    }
    return current;
  }
}

export interface AxiosInstance {
  interceptors: {
    request: InterceptorManager<AxiosRequestConfig>;
    response: InterceptorManager<AxiosResponse>;
  };
  request<T = any>(config: AxiosRequestConfig): Promise<AxiosResponse<T>>;
  get<T = any>(url: string, config?: AxiosRequestConfig): Promise<AxiosResponse<T>>;
  post<T = any>(url: string, data?: any, config?: AxiosRequestConfig): Promise<AxiosResponse<T>>;
  put<T = any>(url: string, data?: any, config?: AxiosRequestConfig): Promise<AxiosResponse<T>>;
  patch<T = any>(url: string, data?: any, config?: AxiosRequestConfig): Promise<AxiosResponse<T>>;
  delete<T = any>(url: string, config?: AxiosRequestConfig): Promise<AxiosResponse<T>>;
}

function normalizeUrl(config: AxiosRequestConfig): string {
  const base = config.baseURL ?? '';
  const path = config.url ?? '';
  if (path.startsWith('http://') || path.startsWith('https://')) {
    return path;
  }
  return `${base}${path}`;
}

function isLocalhost(): boolean {
  if (typeof window === 'undefined') {
    return false;
  }
  return ['localhost', '127.0.0.1'].includes(window.location.hostname);
}

async function performHttpRequest(config: AxiosRequestConfig): Promise<AxiosResponse> {
  const url = new URL(normalizeUrl(config), typeof window !== 'undefined' ? window.location.origin : 'https://career-navigator.local');
  if (config.params) {
    Object.entries(config.params).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        url.searchParams.set(key, String(value));
      }
    });
  }

  const headers = new Headers(config.headers ?? {});
  const method = (config.method ?? 'GET').toUpperCase();
  let body: BodyInit | undefined;
  if (config.data instanceof FormData) {
    body = config.data;
  } else if (config.data !== undefined && method !== 'GET') {
    if (!headers.has('Content-Type')) {
      headers.set('Content-Type', 'application/json');
    }
    body = JSON.stringify(config.data);
  }

  const response = await fetch(url.toString(), {
    method,
    headers,
    body,
  });

  const contentType = response.headers.get('content-type') ?? '';
  const data = contentType.includes('application/json')
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    throw new AxiosError(
      typeof data === 'object' && data?.detail ? data.detail : `Request failed with status ${response.status}`,
      {
        data,
        status: response.status,
        statusText: response.statusText,
        config,
        headers: Object.fromEntries(response.headers.entries()),
      },
      config
    );
  }

  return {
    data,
    status: response.status,
    statusText: response.statusText,
    config,
    headers: Object.fromEntries(response.headers.entries()),
  };
}

function createAxiosInstance(defaults: AxiosRequestConfig = {}): AxiosInstance {
  const requestInterceptors = new InterceptorManager<AxiosRequestConfig>();
  const responseInterceptors = new InterceptorManager<AxiosResponse>();

  const instance: AxiosInstance = {
    interceptors: {
      request: requestInterceptors,
      response: responseInterceptors,
    },
    async request<T = any>(config: AxiosRequestConfig): Promise<AxiosResponse<T>> {
      const mergedConfig = await requestInterceptors.run({
        ...defaults,
        ...config,
        headers: {
          ...(defaults.headers ?? {}),
          ...(config.headers ?? {}),
        },
      });

      try {
        const response = await performHttpRequest(mergedConfig);
        return (await responseInterceptors.run(response)) as AxiosResponse<T>;
      } catch (error: any) {
        const shouldFallback =
          isLocalhost() &&
          (normalizeUrl(mergedConfig).startsWith('/api/') || normalizeUrl(mergedConfig).includes('/api/'));

        if (shouldFallback && !(error instanceof AxiosError && error.response && error.response.status !== 404)) {
          try {
            const data = await handleBrowserApiRequest({
              ...mergedConfig,
              url: normalizeUrl(mergedConfig),
            });
            const fallbackResponse = await responseInterceptors.run({
              data,
              status: 200,
              statusText: 'OK',
              config: mergedConfig,
              headers: {},
            });
            return fallbackResponse as AxiosResponse<T>;
          } catch (fallbackError: any) {
            error = fallbackError;
          }
        }

        const status = error?.response?.status ?? error?.statusCode ?? 500;
        const response: AxiosResponse = {
          data: error?.response?.data ?? { detail: error?.message ?? 'Unknown error' },
          status,
          statusText: status >= 400 ? 'Error' : 'OK',
          config: mergedConfig,
          headers: {},
        };
        throw new AxiosError(error?.message ?? 'Request failed', response, mergedConfig);
      }
    },
    get<T = any>(url: string, config?: AxiosRequestConfig) {
      return instance.request<T>({ ...config, method: 'GET', url });
    },
    post<T = any>(url: string, data?: any, config?: AxiosRequestConfig) {
      return instance.request<T>({ ...config, method: 'POST', url, data });
    },
    put<T = any>(url: string, data?: any, config?: AxiosRequestConfig) {
      return instance.request<T>({ ...config, method: 'PUT', url, data });
    },
    patch<T = any>(url: string, data?: any, config?: AxiosRequestConfig) {
      return instance.request<T>({ ...config, method: 'PATCH', url, data });
    },
    delete<T = any>(url: string, config?: AxiosRequestConfig) {
      return instance.request<T>({ ...config, method: 'DELETE', url });
    },
  };

  return instance;
}

const axios = Object.assign(createAxiosInstance(), {
  create: createAxiosInstance,
  AxiosError,
});

export default axios;
