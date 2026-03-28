/**
 * Custom hook for managing session state.
 */
import { useState, useEffect } from 'react';
import { SESSION_KEYS } from '../config/constants';

interface SessionState {
  sessionId: string | null;
  modelId: string | null;
  dataUploaded: boolean;
}

export function useSession() {
  const [session, setSession] = useState<SessionState>({
    sessionId: null,
    modelId: null,
    dataUploaded: false,
  });

  useEffect(() => {
    // Load session state from sessionStorage
    const sessionId = sessionStorage.getItem(SESSION_KEYS.SESSION_ID);
    const modelId = sessionStorage.getItem(SESSION_KEYS.MODEL_ID);
    const dataUploaded = sessionStorage.getItem(SESSION_KEYS.DATA_UPLOADED) === 'true';

    setSession({
      sessionId,
      modelId,
      dataUploaded,
    });
  }, []);

  const setSessionId = (id: string | null) => {
    if (id) {
      sessionStorage.setItem(SESSION_KEYS.SESSION_ID, id);
    } else {
      sessionStorage.removeItem(SESSION_KEYS.SESSION_ID);
    }
    setSession((prev) => ({ ...prev, sessionId: id }));
  };

  const setModelId = (id: string | null) => {
    if (id) {
      sessionStorage.setItem(SESSION_KEYS.MODEL_ID, id);
    } else {
      sessionStorage.removeItem(SESSION_KEYS.MODEL_ID);
    }
    setSession((prev) => ({ ...prev, modelId: id }));
  };

  const setDataUploaded = (uploaded: boolean) => {
    sessionStorage.setItem(SESSION_KEYS.DATA_UPLOADED, uploaded.toString());
    setSession((prev) => ({ ...prev, dataUploaded: uploaded }));
  };

  const clearSession = () => {
    sessionStorage.removeItem(SESSION_KEYS.SESSION_ID);
    sessionStorage.removeItem(SESSION_KEYS.MODEL_ID);
    sessionStorage.removeItem(SESSION_KEYS.DATA_UPLOADED);
    setSession({
      sessionId: null,
      modelId: null,
      dataUploaded: false,
    });
  };

  return {
    ...session,
    setSessionId,
    setModelId,
    setDataUploaded,
    clearSession,
  };
}
