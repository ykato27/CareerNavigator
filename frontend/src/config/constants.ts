/**
 * Application-wide constants and configuration.
 */

// API Configuration
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
export const API_PREFIX = '/api';

// Session Storage Keys
export const SESSION_KEYS = {
  SESSION_ID: 'career_session_id',
  MODEL_ID: 'career_model_id',
  DATA_UPLOADED: 'career_data_uploaded',
} as const;

// API Endpoints
export const API_ENDPOINTS = {
  // Upload
  UPLOAD: `${API_PREFIX}/upload`,

  // Training
  TRAIN: `${API_PREFIX}/train`,
  MODEL_SUMMARY: (modelId: string) => `${API_PREFIX}/model/${modelId}/summary`,

  // Recommendations
  RECOMMEND: `${API_PREFIX}/recommend`,

  // Graphs
  GRAPH_EGO: `${API_PREFIX}/graph/ego`,
  GRAPH_FULL: `${API_PREFIX}/graph/full`,

  // Weights
  WEIGHTS: (modelId: string) => `${API_PREFIX}/weights/${modelId}`,
  WEIGHTS_UPDATE: `${API_PREFIX}/weights/update`,
  WEIGHTS_OPTIMIZE: `${API_PREFIX}/weights/optimize`,

  // Session
  SESSION_MEMBERS: (sessionId: string) => `${API_PREFIX}/session/${sessionId}/members`,

  // Career
  CAREER_MEMBERS: `${API_PREFIX}/career/members`,
  CAREER_ROLES: `${API_PREFIX}/career/roles`,
  CAREER_MEMBER_SKILLS: `${API_PREFIX}/career/member-skills`,
  CAREER_GAP_ANALYSIS: `${API_PREFIX}/career/gap-analysis`,
  CAREER_ROLE_GAP_ANALYSIS: `${API_PREFIX}/career/role/gap-analysis`,
  CAREER_PATH: `${API_PREFIX}/career/career-path`,
  CAREER_ROLE_PATH: `${API_PREFIX}/career/role/career-path`,
  CAREER_ROADMAP: `${API_PREFIX}/career/career-roadmap`,
  CAREER_ROLE_ROADMAP: `${API_PREFIX}/career/role/career-roadmap`,
  CAREER_ROLE_SKILLS: `${API_PREFIX}/career/role/role-skills`,

  // Organizational
  ORG_METRICS: `${API_PREFIX}/organizational/metrics`,
  ORG_SKILL_GAP: `${API_PREFIX}/organizational/skill-gap`,
  ORG_SUCCESSION: `${API_PREFIX}/organizational/succession`,
  ORG_SIMULATE: `${API_PREFIX}/organizational/simulate`,
} as const;

// UI Constants
export const UI_CONSTANTS = {
  MAX_RECOMMENDATIONS: 50,
  DEFAULT_RECOMMENDATIONS: 10,
  MIN_PERCENTILE: 5,
  MAX_PERCENTILE: 50,
  DEFAULT_PERCENTILE: 20,
  GRAPH_HEIGHT: '600px',
} as const;

// Default Model Parameters
export const DEFAULT_MODEL_PARAMS = {
  MIN_MEMBERS_PER_SKILL: 5,
  CORRELATION_THRESHOLD: 0.2,
  WEIGHTS: {
    READINESS: 0.6,
    BAYESIAN: 0.3,
    UTILITY: 0.1,
  },
} as const;

// File Upload Configuration
export const UPLOAD_CONFIG = {
  REQUIRED_FILES: [
    'members',
    'skills',
    'education',
    'license',
    'categories',
    'acquired',
  ] as const,
  MAX_FILE_SIZE: 50 * 1024 * 1024, // 50MB
  ACCEPTED_TYPES: ['.csv'],
} as const;

// Color Scheme (Skillnote theme)
export const COLORS = {
  PRIMARY: '#00A968',
  PRIMARY_DARK: '#008F58',
  SECONDARY: '#4A90E2',
  SUCCESS: '#10B981',
  WARNING: '#F59E0B',
  ERROR: '#EF4444',
  INFO: '#3B82F6',
} as const;
