/**
 * Application Configuration
 *
 * This file centralizes all configuration values and environment variables.
 * Uses Vite's import.meta.env for environment variable access.
 */

interface AppConfig {
    // API Configuration
    apiBaseUrl: string
    apiEndpoints: {
        health: string
        chatMessages: string
        chatSend: string
        chatClear: string
        documents: string
        deleteDocument: (id: string) => string
        upload: string
        uploadStatus: (filename: string) => string
    }

    // Timeout Configuration
    backendCheckTimeoutMs: number
    backendCheckIntervalMs: number
    statusPollIntervalMs: number
    uploadTimeoutMs: number
    maxPollingTimeMs: number
    maxConsecutiveFailures: number

    // UI Configuration
    appTitle: string
    appSubtitle: string
    successMessageDisplayMs: number
}

// Helper function to get environment variable with fallback
const getEnvVar = (key: string, defaultValue: string): string => {
    return ((import.meta as any).env as Record<string, string>)[key] || defaultValue
}

const getEnvNumber = (key: string, defaultValue: number): number => {
    const value = ((import.meta as any).env as Record<string, string>)[key]
    return value ? parseInt(value, 10) : defaultValue
}

// Build configuration object
const config: AppConfig = {
    // API Configuration
    apiBaseUrl: getEnvVar('VITE_API_BASE_URL', 'http://127.0.0.1:8000'),

    apiEndpoints: {
        health: '/health',
        chatMessages: '/chat/messages',
        chatSend: '/chat/send',
        chatClear: '/chat/clear',
        documents: '/documents',
        deleteDocument: (id: string) => `/documents/${id}`,
        upload: '/upload',
        uploadStatus: (filename: string) => `/${encodeURIComponent(filename)}/status`,
    },

    // Timeout Configuration
    backendCheckTimeoutMs: getEnvNumber('VITE_BACKEND_CHECK_TIMEOUT_MS', 30000),
    backendCheckIntervalMs: getEnvNumber('VITE_BACKEND_CHECK_INTERVAL_MS', 1000),
    statusPollIntervalMs: getEnvNumber('VITE_STATUS_POLL_INTERVAL_MS', 2000),
    uploadTimeoutMs: getEnvNumber('VITE_UPLOAD_TIMEOUT_MS', 172800000), // 48 hours
    maxPollingTimeMs: getEnvNumber('VITE_MAX_POLLING_TIME_MS', 172800000), // 48 hours
    maxConsecutiveFailures: getEnvNumber('VITE_MAX_CONSECUTIVE_FAILURES', 10),

    // UI Configuration
    appTitle: getEnvVar('VITE_APP_TITLE', 'RAG Chat Aplikace'),
    appSubtitle: getEnvVar('VITE_APP_SUBTITLE', 'Lokální chat s dokumenty'),
    successMessageDisplayMs: getEnvNumber('VITE_SUCCESS_MESSAGE_DISPLAY_MS', 5000),
}

// Helper functions to build full URLs
export const buildApiUrl = (endpoint: string): string => {
    return `${config.apiBaseUrl}${endpoint}`
}

export const buildApiEndpointUrl = (endpointKey: keyof typeof config.apiEndpoints): string => {
    const endpoint = config.apiEndpoints[endpointKey]
    return typeof endpoint === 'string' ? buildApiUrl(endpoint) : ''
}

// Helper function for authenticated fetch requests
export const fetchWithAuth = async (url: string, options: RequestInit = {}, token: string | null): Promise<Response> => {
    const headers = new Headers(options.headers)

    if (token) {
        headers.set('Authorization', `Bearer ${token}`)
    }

    const response = await fetch(url, {
        ...options,
        headers,
    })

    // Handle 401 Unauthorized - token expired or invalid
    if (response.status === 401) {
        // Clear auth data
        localStorage.removeItem('rag_auth_token')
        localStorage.removeItem('rag_auth_user')
        // Reload page to trigger login
        window.location.reload()
    }

    return response
}

export default config
