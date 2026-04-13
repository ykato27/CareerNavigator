import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import './index.css'
import App from './App.tsx'
import { API_BASE_URL, SESSION_KEYS } from './config/constants'

const API_SCOPE_KEY = 'career_api_base_url'

function clearSessionWhenApiScopeChanges() {
  const currentApiScope = API_BASE_URL || 'browser-fallback'
  const previousApiScope = sessionStorage.getItem(API_SCOPE_KEY)
  const hasExistingSession = Boolean(
    sessionStorage.getItem(SESSION_KEYS.SESSION_ID) ||
    sessionStorage.getItem(SESSION_KEYS.MODEL_ID) ||
    sessionStorage.getItem(SESSION_KEYS.DATA_UPLOADED)
  )

  if (previousApiScope !== currentApiScope && (previousApiScope !== null || hasExistingSession)) {
    const keysToRemove: string[] = []
    for (let index = 0; index < sessionStorage.length; index += 1) {
      const key = sessionStorage.key(index)
      if (key?.startsWith('career_')) {
        keysToRemove.push(key)
      }
    }
    keysToRemove.forEach((key) => sessionStorage.removeItem(key))
  }

  sessionStorage.setItem(API_SCOPE_KEY, currentApiScope)
}

clearSessionWhenApiScopeChanges()

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </StrictMode>,
)
