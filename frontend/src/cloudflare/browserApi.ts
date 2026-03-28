const DB_NAME = 'career-navigator-cloudflare';
const DB_VERSION = 2;
const REQUIRED_FILE_KEYS = ['members', 'skills', 'education', 'license', 'categories', 'acquired'] as const;
const HEADER_CLEANUP_PATTERN = /\s*###\[.*?\]###/g;
const SESSION_TTL_DAYS = 30;
const TRAINING_MODE = 'cloudflare-approx' as const;
const ARTIFACT_VERSION = '2026-03-28' as const;

type FileKey = (typeof REQUIRED_FILE_KEYS)[number];
type HttpMethod = 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';
type Row = Record<string, string>;

interface AxiosLikeRequestConfig {
  url: string;
  method?: string;
  data?: any;
  params?: Record<string, any>;
}

interface RequestContext {
  path: string;
  method: HttpMethod;
  data: any;
  params: URLSearchParams;
}

interface SessionData {
  id: string;
  createdAt: string;
  expires_at: string;
  files: Record<FileKey, Row[]>;
  members: MemberRecord[];
  skills: SkillRecord[];
  acquired: AcquisitionRecord[];
}

interface MemberRecord {
  member_code: string;
  member_name: string;
  role: string;
  grade: string;
  occupation: string;
  display_name: string;
}

interface SkillRecord {
  skill_code: string;
  skill_name: string;
  category: string;
  type: string;
}

interface AcquisitionRecord {
  member_code: string;
  skill_code: string;
  skill_name: string;
  category: string;
  level: number;
}

interface ConstraintRecord {
  id: string;
  from_skill: string;
  to_skill: string;
  constraint_type: 'required' | 'forbidden';
  value?: number;
  created_at: string;
}

interface ModelArtifact {
  model_id: string;
  session_id: string;
  created_at: string;
  artifact_version: string;
  training_mode: typeof TRAINING_MODE;
  source_storage: 'browser';
  weights: Record<string, number>;
  min_members_per_skill: number;
  correlation_threshold: number;
  members: MemberRecord[];
  skills: SkillRecord[];
  skill_popularity: Record<string, number>;
  adjacency: Record<string, Record<string, number>>;
  skill_degree: Record<string, number>;
  member_skill_codes: Record<string, string[]>;
  member_skill_names: Record<string, string[]>;
  member_skill_levels: Record<string, Record<string, number>>;
  constraints: ConstraintRecord[];
}

interface StoredRecord<T> {
  id: string;
  value: T;
}

const dbPromise = openDatabase();
const sessionCache = new Map<string, SessionData>();
const modelCache = new Map<string, ModelArtifact>();
const constraintCache = new Map<string, ConstraintRecord[]>();

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains('sessions')) {
        db.createObjectStore('sessions', { keyPath: 'id' });
      }
      if (!db.objectStoreNames.contains('models')) {
        db.createObjectStore('models', { keyPath: 'id' });
      }
      if (!db.objectStoreNames.contains('constraints')) {
        db.createObjectStore('constraints', { keyPath: 'id' });
      }
      if (!db.objectStoreNames.contains('job_logs')) {
        db.createObjectStore('job_logs', { keyPath: 'id' });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function idbPut<T>(storeName: string, id: string, value: T): Promise<void> {
  const db = await dbPromise;
  await new Promise<void>((resolve, reject) => {
    const transaction = db.transaction(storeName, 'readwrite');
    transaction.objectStore(storeName).put({ id, value } satisfies StoredRecord<T>);
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error);
  });
}

async function idbGet<T>(storeName: string, id: string): Promise<T | null> {
  const db = await dbPromise;
  return new Promise<T | null>((resolve, reject) => {
    const transaction = db.transaction(storeName, 'readonly');
    const request = transaction.objectStore(storeName).get(id);
    request.onsuccess = () => {
      const result = request.result as StoredRecord<T> | undefined;
      resolve(result?.value ?? null);
    };
    request.onerror = () => reject(request.error);
  });
}

async function idbDelete(storeName: string, id: string): Promise<void> {
  const db = await dbPromise;
  await new Promise<void>((resolve, reject) => {
    const transaction = db.transaction(storeName, 'readwrite');
    transaction.objectStore(storeName).delete(id);
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error);
  });
}

function cleanHeader(header: string): string {
  return header.replace(HEADER_CLEANUP_PATTERN, '').trim();
}

function parseCsv(text: string): Row[] {
  const normalized = text.replace(/^\uFEFF/, '');
  const rows: string[][] = [];
  let currentRow: string[] = [];
  let currentCell = '';
  let inQuotes = false;

  for (let i = 0; i < normalized.length; i += 1) {
    const char = normalized[i];
    const next = normalized[i + 1];

    if (char === '"') {
      if (inQuotes && next === '"') {
        currentCell += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (char === ',' && !inQuotes) {
      currentRow.push(currentCell);
      currentCell = '';
      continue;
    }

    if ((char === '\n' || char === '\r') && !inQuotes) {
      if (char === '\r' && next === '\n') {
        i += 1;
      }
      currentRow.push(currentCell);
      rows.push(currentRow);
      currentRow = [];
      currentCell = '';
      continue;
    }

    currentCell += char;
  }

  if (currentCell.length > 0 || currentRow.length > 0) {
    currentRow.push(currentCell);
    rows.push(currentRow);
  }

  if (rows.length === 0) {
    return [];
  }

  const headers = rows[0].map((header) => cleanHeader(header));
  return rows
    .slice(1)
    .filter((row) => row.some((cell) => cell.trim() !== ''))
    .map((row) => {
      const record: Row = {};
      headers.forEach((header, index) => {
        record[header] = (row[index] ?? '').trim();
      });
      return record;
    });
}

function rowSignature(row: Row): string {
  return Object.keys(row)
    .sort()
    .map((key) => `${key}:${row[key]}`)
    .join('|');
}

function mergeRows(rowsList: Row[][]): Row[] {
  const merged: Row[] = [];
  const seen = new Set<string>();
  const duplicates: string[] = [];

  rowsList.forEach((rows, fileIndex) => {
    rows.forEach((row, rowIndex) => {
      const signature = rowSignature(row);
      if (seen.has(signature)) {
        duplicates.push(`file_${fileIndex + 1}:${rowIndex + 2}`);
        return;
      }
      seen.add(signature);
      merged.push(row);
    });
  });

  if (duplicates.length > 0) {
    throw createHttpError(400, `重複行が検出されました: ${duplicates.join(', ')}`);
  }

  return merged;
}

function pickFirst(row: Row, keys: string[], fallback = ''): string {
  for (const key of keys) {
    if (row[key]) {
      return row[key];
    }
  }
  return fallback;
}

function toNumber(value: string | number | undefined, fallback = 0): number {
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : fallback;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function normalizeWeights(weights?: Record<string, number>): Record<string, number> {
  const base = {
    readiness: 0.6,
    bayesian: 0.3,
    utility: 0.1,
    ...weights,
  };
  const total = Object.values(base).reduce((sum, value) => sum + value, 0);
  if (total <= 0) {
    return { readiness: 0.6, bayesian: 0.3, utility: 0.1 };
  }
  return {
    readiness: base.readiness / total,
    bayesian: base.bayesian / total,
    utility: base.utility / total,
  };
}

function generateSessionId(): string {
  return `session_${Date.now()}`;
}

function generateModelId(sessionId: string): string {
  return `model_${sessionId}_${Date.now()}`;
}

function computeExpiryIso(createdAt: string, ttlDays = SESSION_TTL_DAYS): string {
  const base = new Date(createdAt);
  if (Number.isNaN(base.getTime())) {
    return new Date(Date.now() + ttlDays * 24 * 60 * 60 * 1000).toISOString();
  }
  base.setUTCDate(base.getUTCDate() + ttlDays);
  return base.toISOString();
}

function isExpiredAt(expiresAt: string | null | undefined, now = new Date()): boolean {
  if (!expiresAt) {
    return false;
  }
  const expiresAtTime = new Date(expiresAt).getTime();
  if (Number.isNaN(expiresAtTime)) {
    return false;
  }
  return expiresAtTime <= now.getTime();
}

function buildSessionData(files: Record<FileKey, Row[]>): SessionData {
  const createdAt = new Date().toISOString();
  const members = files.members
    .map((row) => {
      const memberCode = pickFirst(row, ['メンバーコード', 'member_code']);
      const memberName = pickFirst(row, ['メンバー名', 'name']);
      const role = pickFirst(row, ['役職', 'job_position'], '未設定');
      const grade = pickFirst(row, ['職能・等級', 'job_grade'], '未設定');
      const occupation = pickFirst(row, ['職種', 'occupations'], '未設定');
      return {
        member_code: memberCode,
        member_name: memberName,
        role,
        grade,
        occupation,
        display_name: `${memberCode} - ${memberName}`,
      };
    })
    .filter((member) => member.member_code && member.member_name);

  const skillsByCode = new Map<string, SkillRecord>();
  const masterSources: Array<[Row[], string]> = [
    [files.skills, 'SKILL'],
    [files.education, 'EDUCATION'],
    [files.license, 'LICENSE'],
  ];

  for (const [rows, type] of masterSources) {
    rows.forEach((row) => {
      const skillCode = pickFirst(row, ['力量コード', 'skill_code']);
      const skillName = pickFirst(row, ['力量名', 'skill_name']);
      if (!skillCode || !skillName) {
        return;
      }
      skillsByCode.set(skillCode, {
        skill_code: skillCode,
        skill_name: skillName,
        category: pickFirst(row, ['力量カテゴリー', '力量カテゴリー名', 'competence_category_name_1'], '未分類'),
        type,
      });
    });
  }

  const acquired = files.acquired
    .map((row) => {
      const skillCode = pickFirst(row, ['力量コード', 'skill_code']);
      const existingSkill = skillsByCode.get(skillCode);
      const skillName = pickFirst(row, ['力量名', 'skill_name'], existingSkill?.skill_name ?? skillCode);
      const category = existingSkill?.category ?? pickFirst(row, ['力量カテゴリー', '力量カテゴリー名'], '未分類');
      const type = pickFirst(row, ['力量タイプ', 'competence_type'], existingSkill?.type ?? 'SKILL');
      if (skillCode && !skillsByCode.has(skillCode)) {
        skillsByCode.set(skillCode, {
          skill_code: skillCode,
          skill_name: skillName,
          category,
          type,
        });
      }
      return {
        member_code: pickFirst(row, ['メンバーコード', 'member_code']),
        skill_code: skillCode,
        skill_name: skillName,
        category,
        level: toNumber(pickFirst(row, ['レベル', 'level']), 0),
      };
    })
    .filter((record) => record.member_code && record.skill_code);

  return {
    id: generateSessionId(),
    createdAt,
    expires_at: computeExpiryIso(createdAt),
    files,
    members,
    skills: Array.from(skillsByCode.values()),
    acquired,
  };
}

function buildArtifact(
  session: SessionData,
  options: {
    modelId: string;
    weights?: Record<string, number>;
    minMembersPerSkill?: number;
    correlationThreshold?: number;
    constraints?: ConstraintRecord[];
  }
): ModelArtifact {
  const weights = normalizeWeights(options.weights);
  const minMembersPerSkill = options.minMembersPerSkill ?? 5;
  const correlationThreshold = options.correlationThreshold ?? 0.2;
  const constraints = options.constraints ?? [];

  const memberSkillCodes: Record<string, string[]> = {};
  const memberSkillNames: Record<string, string[]> = {};
  const memberSkillLevels: Record<string, Record<string, number>> = {};
  const skillOwners = new Map<string, Set<string>>();

  session.acquired.forEach((record) => {
    if (!memberSkillCodes[record.member_code]) {
      memberSkillCodes[record.member_code] = [];
      memberSkillNames[record.member_code] = [];
      memberSkillLevels[record.member_code] = {};
    }

    if (!memberSkillCodes[record.member_code].includes(record.skill_code)) {
      memberSkillCodes[record.member_code].push(record.skill_code);
      memberSkillNames[record.member_code].push(record.skill_name);
    }
    memberSkillLevels[record.member_code][record.skill_code] = record.level;

    if (!skillOwners.has(record.skill_code)) {
      skillOwners.set(record.skill_code, new Set<string>());
    }
    skillOwners.get(record.skill_code)?.add(record.member_code);
  });

  const eligibleSkillCodes = new Set<string>(
    session.skills
      .filter((skill) => (skillOwners.get(skill.skill_code)?.size ?? 0) >= minMembersPerSkill)
      .map((skill) => skill.skill_code)
  );

  const adjacency: Record<string, Record<string, number>> = {};
  const pairCounts = new Map<string, number>();
  const skillPopularity: Record<string, number> = {};
  const totalMembers = Math.max(session.members.length, 1);

  eligibleSkillCodes.forEach((skillCode) => {
    const owners = skillOwners.get(skillCode)?.size ?? 0;
    skillPopularity[skillCode] = owners / totalMembers;
  });

  Object.values(memberSkillCodes).forEach((skillCodes) => {
    const filtered = skillCodes.filter((skillCode) => eligibleSkillCodes.has(skillCode));
    for (let i = 0; i < filtered.length; i += 1) {
      for (let j = 0; j < filtered.length; j += 1) {
        if (i === j) {
          continue;
        }
        const key = `${filtered[i]}::${filtered[j]}`;
        pairCounts.set(key, (pairCounts.get(key) ?? 0) + 1);
      }
    }
  });

  pairCounts.forEach((count, key) => {
    const [source, target] = key.split('::');
    const owners = skillOwners.get(source)?.size ?? 1;
    const weight = count / owners;
    if (weight < correlationThreshold) {
      return;
    }
    if (!adjacency[source]) {
      adjacency[source] = {};
    }
    adjacency[source][target] = weight;
  });

  const skillByName = new Map(session.skills.map((skill) => [skill.skill_name, skill.skill_code]));
  constraints.forEach((constraint) => {
    const fromCode = skillByName.get(constraint.from_skill);
    const toCode = skillByName.get(constraint.to_skill);
    if (!fromCode || !toCode) {
      return;
    }
    if (!adjacency[fromCode]) {
      adjacency[fromCode] = {};
    }
    if (constraint.constraint_type === 'required') {
      adjacency[fromCode][toCode] = Math.max(adjacency[fromCode][toCode] ?? 0, constraint.value ?? 0.5);
    } else {
      delete adjacency[fromCode][toCode];
    }
  });

  const skillDegree: Record<string, number> = {};
  Object.entries(adjacency).forEach(([skillCode, edges]) => {
    skillDegree[skillCode] = Object.values(edges).reduce((sum, value) => sum + value, 0);
  });

  return {
    model_id: options.modelId,
    session_id: session.id,
    created_at: new Date().toISOString(),
    artifact_version: ARTIFACT_VERSION,
    training_mode: TRAINING_MODE,
    source_storage: 'browser',
    weights,
    min_members_per_skill: minMembersPerSkill,
    correlation_threshold: correlationThreshold,
    members: session.members,
    skills: session.skills.filter((skill) => eligibleSkillCodes.has(skill.skill_code)),
    skill_popularity: skillPopularity,
    adjacency,
    skill_degree: skillDegree,
    member_skill_codes: memberSkillCodes,
    member_skill_names: memberSkillNames,
    member_skill_levels: memberSkillLevels,
    constraints,
  };
}

function createHttpError(status: number, detail: string) {
  const error = new Error(detail) as Error & {
    response: { status: number; data: { detail: string } };
    statusCode: number;
  };
  error.statusCode = status;
  error.response = {
    status,
    data: { detail },
  };
  return error;
}

function escapeHtml(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function polarLayout(labels: string[], radius: number, centerX: number, centerY: number) {
  return labels.map((label, index) => {
    const angle = (Math.PI * 2 * index) / Math.max(labels.length, 1);
    return {
      id: label,
      x: centerX + Math.cos(angle) * radius,
      y: centerY + Math.sin(angle) * radius,
    };
  });
}

function buildGraphHtml(
  title: string,
  nodes: Array<{ id: string; label: string; color: string; x: number; y: number }>,
  edges: Array<{ source: string; target: string; weight: number }>
): string {
  const nodeById = new Map(nodes.map((node) => [node.id, node]));
  const svgEdges = edges
    .map((edge) => {
      const source = nodeById.get(edge.source);
      const target = nodeById.get(edge.target);
      if (!source || !target) {
        return '';
      }
      return `<line x1="${source.x}" y1="${source.y}" x2="${target.x}" y2="${target.y}" stroke="#94a3b8" stroke-width="${Math.max(1, edge.weight * 6)}" marker-end="url(#arrow)" />`;
    })
    .join('');
  const svgNodes = nodes
    .map(
      (node) => `
        <g>
          <circle cx="${node.x}" cy="${node.y}" r="28" fill="${node.color}" stroke="#0f172a" stroke-width="1.5"></circle>
          <text x="${node.x}" y="${node.y + 48}" text-anchor="middle" font-size="12" fill="#0f172a">${escapeHtml(node.label)}</text>
        </g>`
    )
    .join('');

  return `<!doctype html>
  <html lang="ja">
    <head>
      <meta charset="utf-8" />
      <title>${escapeHtml(title)}</title>
      <style>
        body { margin: 0; font-family: Arial, sans-serif; background: #f8fafc; color: #0f172a; }
        .wrap { padding: 16px; }
        .meta { margin-bottom: 12px; font-size: 14px; color: #334155; }
        svg { width: 100%; height: 100%; min-height: 560px; background: white; border: 1px solid #e2e8f0; border-radius: 12px; }
      </style>
    </head>
    <body>
      <div class="wrap">
        <div class="meta">${escapeHtml(title)}</div>
        <svg viewBox="0 0 960 600" preserveAspectRatio="xMidYMid meet">
          <defs>
            <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
              <path d="M0,0 L0,6 L9,3 z" fill="#94a3b8"></path>
            </marker>
          </defs>
          ${svgEdges}
          ${svgNodes}
        </svg>
      </div>
    </body>
  </html>`;
}

async function saveSession(session: SessionData): Promise<void> {
  sessionCache.set(session.id, session);
  await idbPut('sessions', session.id, session);
}

async function writeJobLog(
  payload: {
    sessionId?: string;
    modelId?: string;
    jobType: string;
    status: 'started' | 'completed' | 'failed';
    detail?: string;
  }
): Promise<void> {
  await idbPut('job_logs', `${payload.jobType}_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`, {
    session_id: payload.sessionId ?? null,
    model_id: payload.modelId ?? null,
    job_type: payload.jobType,
    status: payload.status,
    detail: payload.detail ?? null,
    created_at: new Date().toISOString(),
  });
}

async function loadSession(sessionId: string): Promise<SessionData> {
  const cached = sessionCache.get(sessionId);
  if (cached) {
    if (isExpiredAt(cached.expires_at)) {
      await deleteSession(sessionId);
      throw createHttpError(410, 'Session expired');
    }
    return cached;
  }
  const stored = await idbGet<SessionData>('sessions', sessionId);
  if (!stored) {
    throw createHttpError(404, 'Session not found');
  }
  if (!stored.expires_at) {
    stored.expires_at = computeExpiryIso(stored.createdAt);
  }
  if (isExpiredAt(stored.expires_at)) {
    await deleteSession(sessionId);
    throw createHttpError(410, 'Session expired');
  }
  sessionCache.set(sessionId, stored);
  return stored;
}

async function deleteSession(sessionId: string): Promise<void> {
  const modelIds = Array.from(modelCache.values())
    .filter((model) => model.session_id === sessionId)
    .map((model) => model.model_id);
  const storedModels = await idbGetAll<ModelArtifact>('models');
  storedModels
    .filter((model) => model.session_id === sessionId)
    .forEach((model) => {
      if (!modelIds.includes(model.model_id)) {
        modelIds.push(model.model_id);
      }
    });

  for (const modelId of modelIds) {
    modelCache.delete(modelId);
    await idbDelete('models', modelId);
  }
  sessionCache.delete(sessionId);
  constraintCache.delete(sessionId);
  await idbDelete('sessions', sessionId);
  await idbDelete('constraints', sessionId);
}

async function saveModel(model: ModelArtifact): Promise<void> {
  modelCache.set(model.model_id, model);
  await idbPut('models', model.model_id, model);
}

async function loadModel(modelId: string): Promise<ModelArtifact> {
  const cached = modelCache.get(modelId);
  if (cached) {
    return cached;
  }
  const stored = await idbGet<ModelArtifact>('models', modelId);
  if (!stored) {
    throw createHttpError(404, `Model '${modelId}' not found`);
  }
  stored.artifact_version = stored.artifact_version ?? ARTIFACT_VERSION;
  stored.training_mode = stored.training_mode ?? TRAINING_MODE;
  stored.source_storage = stored.source_storage ?? 'browser';
  modelCache.set(modelId, stored);
  return stored;
}

async function loadConstraints(sessionId: string): Promise<ConstraintRecord[]> {
  const cached = constraintCache.get(sessionId);
  if (cached) {
    return cached;
  }
  const stored = await idbGet<ConstraintRecord[]>('constraints', sessionId);
  const result = stored ?? [];
  constraintCache.set(sessionId, result);
  return result;
}

async function saveConstraints(sessionId: string, constraints: ConstraintRecord[]): Promise<void> {
  constraintCache.set(sessionId, constraints);
  await idbPut('constraints', sessionId, constraints);
}

async function idbGetAll<T>(storeName: string): Promise<T[]> {
  const db = await dbPromise;
  return new Promise<T[]>((resolve, reject) => {
    const transaction = db.transaction(storeName, 'readonly');
    const request = transaction.objectStore(storeName).getAll();
    request.onsuccess = () => {
      const results = (request.result as Array<StoredRecord<T>> | undefined) ?? [];
      resolve(results.map((entry) => entry.value));
    };
    request.onerror = () => reject(request.error);
  });
}

function pickMember(session: SessionData, memberCode: string): MemberRecord {
  const member = session.members.find((item) => item.member_code === memberCode);
  if (!member) {
    throw createHttpError(404, `Member '${memberCode}' not found`);
  }
  return member;
}

function scoreRecommendations(artifact: ModelArtifact, memberId: string, topN: number) {
  const ownedCodes = new Set(artifact.member_skill_codes[memberId] ?? []);
  if (!artifact.members.some((member) => member.member_code === memberId)) {
    throw createHttpError(404, `Member '${memberId}' not found`);
  }

  const maxUtility = Math.max(1, ...Object.values(artifact.skill_degree));
  const skillNameByCode = new Map(artifact.skills.map((skill) => [skill.skill_code, skill.skill_name]));

  return artifact.skills
    .filter((skill) => !ownedCodes.has(skill.skill_code))
    .map((skill) => {
      const readinessReasons = Array.from(ownedCodes)
        .map((ownedSkill) => ({
          skillName: skillNameByCode.get(ownedSkill) ?? ownedSkill,
          effect: artifact.adjacency[ownedSkill]?.[skill.skill_code] ?? 0,
        }))
        .filter((reason) => reason.effect > 0)
        .sort((a, b) => b.effect - a.effect)
        .slice(0, 5);

      const readiness = readinessReasons[0]?.effect ?? 0;
      const bayesian = artifact.skill_popularity[skill.skill_code] ?? 0;
      const utilityReasons = Object.entries(artifact.adjacency[skill.skill_code] ?? {})
        .map(([targetSkillCode, effect]) => ({
          skillName: skillNameByCode.get(targetSkillCode) ?? targetSkillCode,
          effect,
        }))
        .sort((a, b) => b.effect - a.effect)
        .slice(0, 5);
      const utility = (artifact.skill_degree[skill.skill_code] ?? 0) / maxUtility;
      const finalScore =
        artifact.weights.readiness * readiness +
        artifact.weights.bayesian * bayesian +
        artifact.weights.utility * utility;

      return {
        rank: 0,
        competence_code: skill.skill_code,
        competence_name: skill.skill_name,
        skill_code: skill.skill_code,
        skill_name: skill.skill_name,
        category: skill.category,
        score: finalScore,
        total_score: finalScore,
        final_score: finalScore,
        readiness_score: readiness,
        readiness_score_normalized: readiness,
        probability_score: bayesian,
        bayesian_score: bayesian,
        bayesian_score_normalized: bayesian,
        utility_score: utility,
        utility_score_normalized: utility,
        reason:
          readinessReasons.length > 0
            ? `${readinessReasons[0].skillName} との関連が強いため`
            : '既存スキル構成との整合性が高いため',
        explanation:
          readinessReasons.length > 0
            ? `${readinessReasons[0].skillName} を保有しているため、${skill.skill_name} の習得準備度が高いと推定しました。`
            : `${skill.skill_name} は現在のスキル構成に対して有望です。`,
        dependencies: readinessReasons.map((reason) => reason.skillName),
        readiness_reasons: readinessReasons.map((reason) => [reason.skillName, reason.effect] as [string, number]),
        utility_reasons: utilityReasons.map((reason) => [reason.skillName, reason.effect] as [string, number]),
        prerequisites: readinessReasons.map((reason) => ({ skill_name: reason.skillName, effect: reason.effect })),
        enables: utilityReasons.map((reason) => ({ skill_name: reason.skillName, effect: reason.effect })),
        details: {
          readiness_score_normalized: readiness,
          bayesian_score_normalized: bayesian,
          utility_score_normalized: utility,
          readiness_reasons: readinessReasons.map((reason) => [reason.skillName, reason.effect] as [string, number]),
          utility_reasons: utilityReasons.map((reason) => [reason.skillName, reason.effect] as [string, number]),
          bayesian_score: bayesian,
        },
      };
    })
    .sort((a, b) => b.final_score - a.final_score)
    .map((recommendation, index) => ({ ...recommendation, rank: index + 1 }))
    .slice(0, topN);
}

function memberCurrentSkills(artifact: ModelArtifact, memberCode: string) {
  const skillByCode = new Map(artifact.skills.map((skill) => [skill.skill_code, skill]));
  const levels = artifact.member_skill_levels[memberCode] ?? {};
  return (artifact.member_skill_codes[memberCode] ?? [])
    .map((skillCode) => {
      const skill = skillByCode.get(skillCode);
      if (!skill) {
        return null;
      }
      return {
        skill_code: skill.skill_code,
        skill_name: skill.skill_name,
        category: skill.category,
        level: levels[skill.skill_code] ?? 0,
      };
    })
    .filter(Boolean);
}

function getRoleSkills(session: SessionData, roleName: string, minFrequency: number) {
  const roleMembers = session.members.filter((member) => member.role === roleName);
  const memberCodes = new Set(roleMembers.map((member) => member.member_code));
  const counts = new Map<string, number>();
  const skillLookup = new Map(session.skills.map((skill) => [skill.skill_code, skill]));

  session.acquired.forEach((record) => {
    if (!memberCodes.has(record.member_code)) {
      return;
    }
    counts.set(record.skill_code, (counts.get(record.skill_code) ?? 0) + 1);
  });

  const totalMembers = Math.max(roleMembers.length, 1);
  return Array.from(counts.entries())
    .map(([skillCode, count]) => {
      const skill = skillLookup.get(skillCode);
      if (!skill) {
        return null;
      }
      const frequency = count / totalMembers;
      if (frequency < minFrequency) {
        return null;
      }
      return {
        skill_code: skillCode,
        skill_name: skill.skill_name,
        category: skill.category,
        frequency,
        member_count: count,
        priority: frequency >= 0.5 ? 'high' : frequency >= 0.3 ? 'medium' : 'low',
      };
    })
    .filter(Boolean)
    .sort((a, b) => (b?.frequency ?? 0) - (a?.frequency ?? 0));
}

function buildGapAnalysis(session: SessionData, artifact: ModelArtifact, sourceMemberCode: string, targetSkillCodes: string[]) {
  const sourceMember = pickMember(session, sourceMemberCode);
  const sourceSkills = new Set(artifact.member_skill_codes[sourceMemberCode] ?? []);
  const missingSkillCodes = targetSkillCodes.filter((skillCode) => !sourceSkills.has(skillCode));
  const skillByCode = new Map(artifact.skills.map((skill) => [skill.skill_code, skill]));
  const gapSkills = missingSkillCodes
    .map((skillCode) => {
      const skill = skillByCode.get(skillCode);
      if (!skill) {
        return null;
      }
      return { skill_code: skill.skill_code, skill_name: skill.skill_name, category: skill.category };
    })
    .filter(Boolean);

  return {
    source_member: sourceMember,
    gap_skills: gapSkills,
    gap_count: gapSkills.length,
    source_skill_count: sourceSkills.size,
    completion_rate: targetSkillCodes.length === 0 ? 1 : (targetSkillCodes.length - gapSkills.length) / targetSkillCodes.length,
  };
}

function buildCareerPathResponse(
  artifact: ModelArtifact,
  sourceMemberCode: string,
  targetSkillCodes: string[],
  minTotalScore: number,
  minReadinessScore: number
) {
  const targetSet = new Set(targetSkillCodes);
  const recommendations = scoreRecommendations(artifact, sourceMemberCode, artifact.skills.length)
    .filter((recommendation) => targetSet.has(recommendation.skill_code))
    .filter((recommendation) => recommendation.total_score >= minTotalScore && recommendation.readiness_score >= minReadinessScore)
    .map((recommendation) => ({
      competence_code: recommendation.skill_code,
      competence_name: recommendation.skill_name,
      category: recommendation.category,
      total_score: recommendation.total_score,
      readiness_score: recommendation.readiness_score,
      bayesian_score: recommendation.bayesian_score,
      utility_score: recommendation.utility_score,
      readiness_reasons: recommendation.readiness_reasons,
      utility_reasons: recommendation.utility_reasons,
      prerequisites: recommendation.prerequisites,
      enables: recommendation.enables,
      explanation: recommendation.explanation,
    }));

  const avgScore =
    recommendations.length === 0
      ? 0
      : recommendations.reduce((sum, recommendation) => sum + recommendation.total_score, 0) / recommendations.length;
  const totalDependencies = recommendations.reduce((sum, recommendation) => sum + recommendation.prerequisites.length, 0);

  return {
    success: true,
    recommended_skills: recommendations,
    skill_count: recommendations.length,
    avg_score: avgScore,
    total_dependencies: totalDependencies,
    estimated_months: Math.max(1, Math.ceil(recommendations.length / 2)),
    message: recommendations.length > 0 ? `${recommendations.length}件の優先スキルを抽出しました` : '条件に一致する推薦がありませんでした',
  };
}

function buildRoadmapChart(recommendedSkills: Array<{ competence_name: string; total_score: number }>) {
  return {
    data: [
      {
        type: 'bar',
        orientation: 'h',
        x: recommendedSkills.map(() => 7),
        y: recommendedSkills.map((skill) => skill.competence_name),
        base: recommendedSkills.map((_, index) => index * 7),
        marker: {
          color: recommendedSkills.map((skill) => (skill.total_score >= 0.7 ? '#2ecc71' : skill.total_score >= 0.4 ? '#3498db' : '#95a5a6')),
        },
        customdata: recommendedSkills.map((_, index) => [index * 7, index * 7 + 6]),
        hovertemplate: '%{y}<br>開始週: %{base}<br>終了週: %{customdata[1]}<extra></extra>',
      },
    ],
    layout: {
      title: '学習ロードマップ',
      barmode: 'stack',
      xaxis: { title: '週' },
      yaxis: { automargin: true },
      margin: { l: 160, r: 20, t: 60, b: 40 },
      height: Math.max(360, recommendedSkills.length * 56),
    },
  };
}

async function routeGet(path: string, ctx: RequestContext) {
  const sessionMatch = path.match(/^\/api\/session\/([^/]+)$/);
  if (sessionMatch) {
    const session = await loadSession(sessionMatch[1]);
    return {
      success: true,
      session_id: session.id,
      created_at: session.createdAt,
      expires_at: session.expires_at,
      source_storage: 'browser',
    };
  }

  const sessionMembersMatch = path.match(/^\/api\/session\/([^/]+)\/members$/);
  if (sessionMembersMatch) {
    const session = await loadSession(sessionMembersMatch[1]);
    return { success: true, members: session.members, total_count: session.members.length, expires_at: session.expires_at };
  }

  const constraintsMatch = path.match(/^\/api\/constraints\/([^/]+)$/);
  if (constraintsMatch) {
    return { constraints: await loadConstraints(constraintsMatch[1]) };
  }

  const weightsMatch = path.match(/^\/api\/weights\/([^/]+)$/);
  if (weightsMatch) {
    return { success: true, weights: (await loadModel(weightsMatch[1])).weights };
  }

  const skillsMatch = path.match(/^\/api\/skills\/([^/]+)$/);
  if (skillsMatch) {
    const model = await loadModel(skillsMatch[1]);
    return { success: true, skills: model.skills.map((skill) => skill.skill_name) };
  }

  const modelSummaryMatch = path.match(/^\/api\/model\/([^/]+)\/summary$/);
  if (modelSummaryMatch) {
    const model = await loadModel(modelSummaryMatch[1]);
    return {
      success: true,
      summary: {
        model_id: model.model_id,
        session_id: model.session_id,
        created_at: model.created_at,
        artifact_version: model.artifact_version,
        training_mode: model.training_mode,
        source_storage: model.source_storage,
        num_members: model.members.length,
        num_skills: model.skills.length,
        weights: model.weights,
        min_members_per_skill: model.min_members_per_skill,
        correlation_threshold: model.correlation_threshold,
      },
    };
  }

  if (path === '/api/career/members') {
    const sessionId = ctx.params.get('session_id');
    if (!sessionId) {
      throw createHttpError(400, 'session_id is required');
    }
    const session = await loadSession(sessionId);
    return {
      success: true,
      members: session.members.map((member) => ({
        ...member,
        skill_count: session.acquired.filter((record) => record.member_code === member.member_code).length,
      })),
    };
  }

  if (path === '/api/career/roles') {
    const sessionId = ctx.params.get('session_id');
    if (!sessionId) {
      throw createHttpError(400, 'session_id is required');
    }
    const session = await loadSession(sessionId);
    const roleCounts = new Map<string, number>();
    session.members.forEach((member) => {
      roleCounts.set(member.role, (roleCounts.get(member.role) ?? 0) + 1);
    });
    return {
      success: true,
      roles: Array.from(roleCounts.entries())
        .map(([role_name, member_count]) => ({ role_name, member_count }))
        .sort((a, b) => b.member_count - a.member_count),
    };
  }

  throw createHttpError(404, `Unknown endpoint: ${path}`);
}

async function routePost(path: string, ctx: RequestContext) {
  if (path === '/api/upload') {
    const formData = ctx.data as FormData;
    const parsedFiles: Partial<Record<FileKey, Row[]>> = {};

    for (const fileKey of REQUIRED_FILE_KEYS) {
      const fileEntries = Array.from(formData.entries())
        .filter(([key]) => key.startsWith(`${fileKey}[`))
        .map(([, value]) => value)
        .filter((value): value is File => value instanceof File);

      if (fileEntries.length === 0) {
        throw createHttpError(400, `以下のカテゴリにファイルがありません: ${fileKey}`);
      }

      const rowsPerFile = await Promise.all(fileEntries.map(async (file) => parseCsv(await file.text())));
      parsedFiles[fileKey] = mergeRows(rowsPerFile);
    }

    const session = buildSessionData(parsedFiles as Record<FileKey, Row[]>);
    await saveSession(session);
    return {
      session_id: session.id,
      message: 'Files uploaded and parsed successfully',
      files: REQUIRED_FILE_KEYS,
      expires_at: session.expires_at,
      source_storage: 'browser',
      training_mode: TRAINING_MODE,
    };
  }

  if (path === '/api/admin/cleanup') {
    const sessions = await idbGetAll<SessionData>('sessions');
    const expiredSessions = sessions.filter((session) => isExpiredAt(session.expires_at)).map((session) => session.id);
    for (const sessionId of expiredSessions) {
      await deleteSession(sessionId);
    }
    return {
      success: true,
      deleted_session_ids: expiredSessions,
      deleted_count: expiredSessions.length,
      executed_at: new Date().toISOString(),
    };
  }

  if (path === '/api/train') {
    const sessionId = ctx.data.session_id as string;
    if (!sessionId) {
      throw createHttpError(400, 'session_id is required');
    }
    const session = await loadSession(sessionId);
    const modelId = generateModelId(sessionId);
    const startedAt = Date.now();
    await writeJobLog({
      sessionId,
      modelId,
      jobType: 'train',
      status: 'started',
      detail: 'Browser fallback approximate training started',
    });
    try {
      const constraints = await loadConstraints(sessionId);
      const artifact = buildArtifact(session, {
        modelId,
        weights: ctx.data.weights,
        minMembersPerSkill: ctx.data.min_members_per_skill ?? ctx.data.min_members ?? 5,
        correlationThreshold: ctx.data.correlation_threshold ?? 0.2,
        constraints,
      });
      await saveModel(artifact);
      const learningTime = Number(((Date.now() - startedAt) / 1000).toFixed(3));
      await writeJobLog({
        sessionId,
        modelId,
        jobType: 'train',
        status: 'completed',
        detail: `Browser fallback approximate training completed in ${learningTime}s`,
      });
      return {
        success: true,
        model_id: modelId,
        message: 'ブラウザ内エンジンで近似学習が完了しました',
        training_mode: artifact.training_mode,
        artifact_version: artifact.artifact_version,
        source_storage: artifact.source_storage,
        expires_at: session.expires_at,
        summary: {
          model_id: modelId,
          session_id: sessionId,
          num_members: artifact.members.length,
          num_skills: artifact.skills.length,
          learning_time: learningTime,
          total_time: learningTime,
          weights: artifact.weights,
          min_members_per_skill: artifact.min_members_per_skill,
          correlation_threshold: artifact.correlation_threshold,
          training_mode: artifact.training_mode,
          artifact_version: artifact.artifact_version,
          source_storage: artifact.source_storage,
          expires_at: session.expires_at,
        },
      };
    } catch (error: any) {
      await writeJobLog({
        sessionId,
        modelId,
        jobType: 'train',
        status: 'failed',
        detail: error?.message ?? 'Training failed',
      });
      throw error;
    }
  }

  if (path === '/api/recommend') {
    const model = await loadModel(ctx.data.model_id);
    const session = await loadSession(model.session_id);
    const recommendations = scoreRecommendations(model, ctx.data.member_id, ctx.data.top_n ?? 10);
    return {
      success: true,
      model_id: model.model_id,
      member_id: ctx.data.member_id,
      member_name: pickMember(session, ctx.data.member_id).member_name,
      recommendations,
      metadata: { weights: model.weights, total_candidates: recommendations.length },
      message: recommendations.length > 0 ? '推薦を生成しました' : '利用可能な推薦がありません',
    };
  }

  if (path === '/api/scatter-plot') {
    const model = await loadModel(ctx.data.model_id);
    const recommendations = scoreRecommendations(model, ctx.data.member_id, model.skills.length);
    return {
      success: true,
      model_id: model.model_id,
      member_id: ctx.data.member_id,
      skills: recommendations.map((recommendation) => ({
        skill_code: recommendation.skill_code,
        skill_name: recommendation.skill_name,
        category: recommendation.category,
        readiness_score: recommendation.readiness_score,
        utility_score: recommendation.utility_score,
        final_score: recommendation.final_score,
      })),
      metadata: { weights: model.weights, total_skills: recommendations.length },
    };
  }

  if (path === '/api/graph/ego') {
    const model = await loadModel(ctx.data.model_id);
    const centerCode =
      ctx.data.skill_code ??
      ctx.data.competence_code ??
      model.skills.find((skill) => skill.skill_name === ctx.data.center_node)?.skill_code ??
      ctx.data.center_node;
    if (!centerCode) {
      throw createHttpError(400, 'center_node is required');
    }
    const centerSkill = model.skills.find((skill) => skill.skill_code === centerCode);
    if (!centerSkill) {
      throw createHttpError(404, 'Skill not found in model');
    }

    const radius = ctx.data.radius ?? 1;
    const threshold = ctx.data.threshold ?? 0.05;
    const memberSkills = new Set<string>(ctx.data.member_skills ?? []);
    const outgoing = Object.entries(model.adjacency[centerCode] ?? {})
      .filter(([, weight]) => weight >= threshold)
      .sort((a, b) => b[1] - a[1])
      .slice(0, Math.max(4, radius * 6));
    const nodes = [
      { id: centerCode, label: centerSkill.skill_name, color: '#97C2FC', x: 480, y: 280 },
      ...polarLayout(outgoing.map(([skillCode]) => skillCode), 180, 480, 280).map((position) => {
        const skill = model.skills.find((item) => item.skill_code === position.id);
        return {
          id: position.id,
          label: skill?.skill_name ?? position.id,
          color: memberSkills.has(skill?.skill_name ?? '') ? '#90EE90' : '#DDDDDD',
          x: position.x,
          y: position.y,
        };
      }),
    ];
    const edges = outgoing.map(([target, weight]) => ({ source: centerCode, target, weight }));
    return {
      success: true,
      html: buildGraphHtml(`${centerSkill.skill_name} の関連スキル`, nodes, edges),
      graph_data: { nodes, edges },
    };
  }

  if (path === '/api/graph/full') {
    const model = await loadModel(ctx.data.model_id);
    const threshold = ctx.data.threshold ?? 0.3;
    const topN = Math.min(ctx.data.top_n ?? 20, model.skills.length);
    const topSkills = [...model.skills]
      .sort((a, b) => (model.skill_popularity[b.skill_code] ?? 0) - (model.skill_popularity[a.skill_code] ?? 0))
      .slice(0, topN);
    const nodes = polarLayout(topSkills.map((skill) => skill.skill_code), 220, 480, 300).map((position) => {
      const skill = topSkills.find((item) => item.skill_code === position.id)!;
      return { id: skill.skill_code, label: skill.skill_name, color: '#DDDDDD', x: position.x, y: position.y };
    });
    const allowedSkills = new Set(topSkills.map((skill) => skill.skill_code));
    const edges = Object.entries(model.adjacency)
      .flatMap(([source, targets]) => Object.entries(targets).map(([target, weight]) => ({ source, target, weight })))
      .filter((edge) => edge.weight >= threshold)
      .filter((edge) => allowedSkills.has(edge.source) && allowedSkills.has(edge.target))
      .sort((a, b) => b.weight - a.weight)
      .slice(0, topN * 2);
    return {
      success: true,
      html: buildGraphHtml('因果グラフ全体', nodes, edges),
      node_count: nodes.length,
      graph_data: { nodes, edges },
    };
  }

  if (path === '/api/weights/update') {
    const model = await loadModel(ctx.data.model_id);
    const updated = { ...model, weights: normalizeWeights(ctx.data.weights) };
    await saveModel(updated);
    return { success: true, model_id: updated.model_id, weights: updated.weights };
  }

  if (path === '/api/weights/optimize') {
    const model = await loadModel(ctx.data.model_id);
    const utilityBias = model.skills.length > 200 ? 0.25 : 0.15;
    return {
      success: true,
      optimized_weights: normalizeWeights({ readiness: 0.6, bayesian: 0.25 - utilityBias / 2, utility: 0.15 + utilityBias }),
    };
  }

  if (path === '/api/career/member-skills') {
    const session = await loadSession(ctx.data.session_id);
    const artifact = Array.from(modelCache.values()).find((item) => item.session_id === session.id)
      ?? buildArtifact(session, { modelId: 'ephemeral', minMembersPerSkill: 1, correlationThreshold: 0 });
    return { success: true, current_skills: memberCurrentSkills(artifact, ctx.data.member_code) };
  }

  if (path === '/api/career/gap-analysis' || path === '/api/career/role/gap-analysis') {
    const session = await loadSession(ctx.data.session_id);
    const model = await loadModel(ctx.data.model_id);
    let targetSkillCodes: string[];
    let targetMember: MemberRecord | undefined;
    if (path.endsWith('/role/gap-analysis')) {
      targetSkillCodes = getRoleSkills(session, ctx.data.target_role, ctx.data.min_frequency ?? 0.1).map((skill) => skill!.skill_code);
    } else {
      targetMember = pickMember(session, ctx.data.target_member_code);
      targetSkillCodes = model.member_skill_codes[ctx.data.target_member_code] ?? [];
    }
    const base = buildGapAnalysis(session, model, ctx.data.source_member_code, targetSkillCodes);
    return {
      success: true,
      ...base,
      target_member: targetMember ?? {
        member_code: '',
        member_name: path.endsWith('/role/gap-analysis') ? ctx.data.target_role : '',
        role: ctx.data.target_role ?? '',
        grade: '',
        occupation: '',
        display_name: ctx.data.target_role ?? '',
      },
      target_skill_count: targetSkillCodes.length,
    };
  }

  if (path === '/api/career/career-path' || path === '/api/career/role/career-path') {
    const session = await loadSession(ctx.data.session_id);
    const model = await loadModel(ctx.data.model_id);
    const targetSkillCodes = path.endsWith('/role/career-path')
      ? getRoleSkills(session, ctx.data.target_role, ctx.data.min_frequency ?? 0.1).map((skill) => skill!.skill_code)
      : model.member_skill_codes[ctx.data.target_member_code] ?? [];
    return buildCareerPathResponse(model, ctx.data.source_member_code, targetSkillCodes, ctx.data.min_total_score ?? 0, ctx.data.min_readiness_score ?? 0);
  }

  if (path === '/api/career/career-roadmap' || path === '/api/career/role/career-roadmap') {
    const session = await loadSession(ctx.data.session_id);
    const model = await loadModel(ctx.data.model_id);
    const targetSkillCodes = path.endsWith('/role/career-roadmap')
      ? getRoleSkills(session, ctx.data.target_role, ctx.data.min_frequency ?? 0.1).map((skill) => skill!.skill_code)
      : model.member_skill_codes[ctx.data.target_member_code] ?? [];
    const pathResponse = buildCareerPathResponse(model, ctx.data.source_member_code, targetSkillCodes, ctx.data.min_total_score ?? 0, ctx.data.min_readiness_score ?? 0);
    return { success: true, gantt_chart: buildRoadmapChart(pathResponse.recommended_skills) };
  }

  if (path === '/api/career/role/role-skills') {
    const session = await loadSession(ctx.data.session_id);
    const skills = getRoleSkills(session, ctx.data.role_name, ctx.data.min_frequency ?? 0.1);
    return {
      success: true,
      role_name: ctx.data.role_name,
      total_members: session.members.filter((member) => member.role === ctx.data.role_name).length,
      skills,
      skill_count: skills.length,
    };
  }

  if (path.startsWith('/api/constraints/') && path.endsWith('/apply')) {
    const sessionId = path.split('/')[3];
    const session = await loadSession(sessionId);
    const currentModel = await loadModel(ctx.data.model_id);
    const constraints = await loadConstraints(sessionId);
    const updated = buildArtifact(session, {
      modelId: currentModel.model_id,
      weights: currentModel.weights,
      minMembersPerSkill: currentModel.min_members_per_skill,
      correlationThreshold: currentModel.correlation_threshold,
      constraints,
    });
    await saveModel(updated);
    return { success: true, applied_count: constraints.length, skipped_count: 0 };
  }

  if (path.startsWith('/api/constraints/')) {
    const sessionId = path.split('/')[3];
    const constraints = await loadConstraints(sessionId);
    const nextConstraint: ConstraintRecord = {
      id: `constraint_${Date.now()}`,
      from_skill: ctx.data.from_skill,
      to_skill: ctx.data.to_skill,
      constraint_type: ctx.data.constraint_type,
      value: ctx.data.value,
      created_at: new Date().toISOString(),
    };
    await saveConstraints(sessionId, [...constraints, nextConstraint]);
    return { success: true, constraint: nextConstraint };
  }

  if (path === '/api/organizational/metrics') {
    const session = await loadSession(ctx.data.session_id);
    const uniqueSkills = new Set(session.acquired.map((record) => record.skill_code));
    const memberCounts = new Map<string, number>();
    session.acquired.forEach((record) => {
      memberCounts.set(record.member_code, (memberCounts.get(record.member_code) ?? 0) + 1);
    });
    const avgSkills = session.members.length === 0 ? 0 : Array.from(memberCounts.values()).reduce((sum, value) => sum + value, 0) / session.members.length;
    const topSkills = Array.from(uniqueSkills)
      .map((skillCode) => {
        const skill = session.skills.find((item) => item.skill_code === skillCode);
        return {
          skill_code: skillCode,
          skill_name: skill?.skill_name ?? skillCode,
          member_count: session.acquired.filter((record) => record.skill_code === skillCode).length,
        };
      })
      .sort((a, b) => b.member_count - a.member_count)
      .slice(0, 10);
    return {
      success: true,
      metrics: {
        total_members: session.members.length,
        total_skills: uniqueSkills.size,
        avg_skills_per_member: Number(avgSkills.toFixed(1)),
        coverage_rate: uniqueSkills.size / Math.max(session.skills.length, 1),
        diversity_index: Number((uniqueSkills.size / Math.max(session.members.length, 1)).toFixed(2)),
        high_concentration_skills: topSkills.filter((skill) => skill.member_count >= session.members.length * 0.5).length,
        low_concentration_skills: topSkills.filter((skill) => skill.member_count <= session.members.length * 0.1).length,
      },
      top_skills: topSkills,
    };
  }

  if (path === '/api/organizational/skill-gap') {
    const session = await loadSession(ctx.data.session_id);
    const percentile = Number(ctx.data.percentile ?? 0.2);
    const skillCountsByMember = session.members.map((member) => ({
      member_code: member.member_code,
      skill_count: session.acquired.filter((record) => record.member_code === member.member_code).length,
    }));
    const sortedMembers = [...skillCountsByMember].sort((a, b) => b.skill_count - a.skill_count);
    const targetMembers = new Set(sortedMembers.slice(0, Math.max(1, Math.ceil(sortedMembers.length * percentile))).map((member) => member.member_code));
    const currentCounts = new Map<string, number>();
    const targetCounts = new Map<string, number>();
    session.acquired.forEach((record) => {
      currentCounts.set(record.skill_code, (currentCounts.get(record.skill_code) ?? 0) + 1);
      if (targetMembers.has(record.member_code)) {
        targetCounts.set(record.skill_code, (targetCounts.get(record.skill_code) ?? 0) + 1);
      }
    });
    const gaps = session.skills
      .map((skill) => {
        const currentRate = (currentCounts.get(skill.skill_code) ?? 0) / Math.max(session.members.length, 1);
        const targetRate = (targetCounts.get(skill.skill_code) ?? 0) / Math.max(targetMembers.size, 1);
        const gapRate = Math.max(0, targetRate - currentRate);
        return {
          skill_code: skill.skill_code,
          skill_name: skill.skill_name,
          current_rate: currentRate,
          target_rate: targetRate,
          gap_rate: gapRate,
          gap_percentage: targetRate === 0 ? 0 : gapRate / targetRate,
        };
      })
      .filter((gap) => gap.gap_rate > 0)
      .sort((a, b) => b.gap_rate - a.gap_rate)
      .slice(0, 20);
    return { success: true, gap_analysis: gaps, critical_skills: gaps.filter((gap) => gap.gap_percentage >= 0.3).slice(0, 5) };
  }

  if (path === '/api/organizational/succession') {
    const session = await loadSession(ctx.data.session_id);
    const targetPosition = String(ctx.data.target_position ?? '');
    const roleMembers = session.members.filter((member) => member.role.includes(targetPosition));
    const targetSkillCodes = roleMembers.length > 0 ? getRoleSkills(session, roleMembers[0].role, 0.2).map((skill) => skill!.skill_code) : [];
    const targetSet = new Set(targetSkillCodes);
    const candidates = session.members
      .filter((member) => !member.role.includes(targetPosition))
      .map((member) => {
        const owned = new Set(session.acquired.filter((record) => record.member_code === member.member_code).map((record) => record.skill_code));
        const matchCount = targetSkillCodes.filter((skillCode) => owned.has(skillCode)).length;
        const readiness = targetSkillCodes.length === 0 ? 0 : matchCount / targetSkillCodes.length;
        return {
          member_code: member.member_code,
          member_name: member.member_name,
          current_position: member.role,
          current_grade: member.grade,
          readiness_score: readiness,
          skill_match_rate: readiness,
          owned_skills_count: owned.size,
          missing_skills_count: Math.max(0, targetSet.size - matchCount),
          estimated_timeline: readiness >= 0.8 ? '0-3ヶ月' : readiness >= 0.6 ? '3-6ヶ月' : '6-12ヶ月',
        };
      })
      .sort((a, b) => b.readiness_score - a.readiness_score)
      .slice(0, Number(ctx.data.max_candidates ?? 20));
    return { success: true, candidates };
  }

  throw createHttpError(404, `Unknown endpoint: ${path}`);
}

async function routeDelete(path: string) {
  const sessionMatch = path.match(/^\/api\/session\/([^/]+)$/);
  if (sessionMatch) {
    await deleteSession(sessionMatch[1]);
    return { success: true };
  }

  const constraintMatch = path.match(/^\/api\/constraints\/([^/]+)\/([^/]+)$/);
  if (constraintMatch) {
    const sessionId = constraintMatch[1];
    const constraintId = constraintMatch[2];
    const constraints = await loadConstraints(sessionId);
    await saveConstraints(sessionId, constraints.filter((constraint) => constraint.id !== constraintId));
    return { success: true };
  }

  throw createHttpError(404, `Unknown endpoint: ${path}`);
}

export async function handleBrowserApiRequest(config: AxiosLikeRequestConfig) {
  const method = (config.method?.toUpperCase() ?? 'GET') as HttpMethod;
  const url = new URL(config.url, 'https://career-navigator.local');
  if (config.params) {
    Object.entries(config.params).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        url.searchParams.set(key, String(value));
      }
    });
  }

  const ctx: RequestContext = { path: url.pathname, method, data: config.data, params: url.searchParams };

  switch (method) {
    case 'GET':
      return routeGet(ctx.path, ctx);
    case 'POST':
      return routePost(ctx.path, ctx);
    case 'DELETE':
      return routeDelete(ctx.path);
    default:
      throw createHttpError(405, `Method ${method} is not supported`);
  }
}
