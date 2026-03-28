import {
  ARTIFACT_VERSION,
  REQUIRED_FILE_KEYS,
  SESSION_TTL_DAYS,
  TRAINING_MODE,
  type ConstraintRecord,
  type FileKey,
  type ModelArtifact,
  type SessionData,
  type Row,
  buildArtifact,
  buildCareerPathResponse,
  buildGapAnalysis,
  buildGraphHtml,
  buildRoadmapChart,
  buildSessionData,
  computeExpiryIso,
  createHttpError,
  generateModelId,
  getRoleSkills,
  isExpiredAt,
  memberCurrentSkills,
  mergeRows,
  normalizeWeights,
  parseCsv,
  pickMember,
  polarLayout,
  scoreRecommendations,
} from '../../shared/cloudflare-engine';

interface Env {
  DB: D1Database;
  STORAGE?: R2Bucket;
  CLEANUP_TOKEN?: string;
  SESSION_TTL_DAYS?: string;
}

const SESSION_OBJECT_KEY = (sessionId: string) => `sessions/${sessionId}.json`;
const MODEL_OBJECT_KEY = (modelId: string) => `models/${modelId}.json`;
const DEFAULT_CLEANUP_BATCH = 100;

interface SessionRow {
  id: string;
  created_at: string;
  expires_at: string | null;
  r2_key: string;
  source_storage: 'r2' | 'd1' | null;
}

interface ModelRow {
  id: string;
  session_id: string;
  created_at: string;
  r2_key: string;
  artifact_version: string | null;
  training_mode: string | null;
  source_storage: 'r2' | 'd1' | null;
  weights_json: string;
}

function getStorageMode(env: Env): 'r2' | 'd1' {
  return env.STORAGE ? 'r2' : 'd1';
}

function getSessionTtlDays(env: Env): number {
  const parsed = Number(env.SESSION_TTL_DAYS ?? SESSION_TTL_DAYS);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : SESSION_TTL_DAYS;
}

function getSessionExpiry(createdAt: string, expiresAt: string | null | undefined, env: Env): string {
  return expiresAt ?? computeExpiryIso(createdAt, getSessionTtlDays(env));
}

function jsonResponse(data: unknown, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      'content-type': 'application/json; charset=utf-8',
    },
  });
}

async function writeJobLog(
  env: Env,
  payload: {
    sessionId?: string;
    modelId?: string;
    jobType: string;
    status: 'started' | 'completed' | 'failed';
    detail?: string;
  }
): Promise<void> {
  await env.DB.prepare(
    'INSERT INTO job_logs (id, session_id, model_id, job_type, status, detail, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)'
  )
    .bind(
      `${payload.jobType}_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
      payload.sessionId ?? null,
      payload.modelId ?? null,
      payload.jobType,
      payload.status,
      payload.detail ?? null,
      new Date().toISOString()
    )
    .run();
}

async function putJson(env: Env, key: string, value: unknown): Promise<void> {
  const json = JSON.stringify(value);
  if (env.STORAGE) {
    await env.STORAGE.put(key, json, {
      httpMetadata: { contentType: 'application/json; charset=utf-8' },
    });
    return;
  }

  await env.DB.prepare(
    'INSERT OR REPLACE INTO storage_objects (object_key, json_value, updated_at) VALUES (?1, ?2, ?3)'
  )
    .bind(key, json, new Date().toISOString())
    .run();
}

async function getJson<T>(env: Env, key: string): Promise<T> {
  if (env.STORAGE) {
    const object = await env.STORAGE.get(key);
    if (!object) {
      throw createHttpError(404, `Object '${key}' not found`);
    }
    return (await object.json()) as T;
  }

  const record = await env.DB.prepare('SELECT json_value FROM storage_objects WHERE object_key = ?1')
    .bind(key)
    .first<{ json_value: string }>();
  if (!record?.json_value) {
    throw createHttpError(404, `Object '${key}' not found`);
  }
  return JSON.parse(record.json_value) as T;
}

async function deleteJson(env: Env, key: string): Promise<void> {
  if (env.STORAGE) {
    await env.STORAGE.delete(key);
    return;
  }

  await env.DB.prepare('DELETE FROM storage_objects WHERE object_key = ?1').bind(key).run();
}

async function saveSession(env: Env, session: SessionData): Promise<void> {
  const objectKey = SESSION_OBJECT_KEY(session.id);
  const sourceStorage = getStorageMode(env);
  await putJson(env, objectKey, session);
  await env.DB.prepare(
    'INSERT OR REPLACE INTO sessions (id, created_at, expires_at, r2_key, file_keys, source_storage) VALUES (?1, ?2, ?3, ?4, ?5, ?6)'
  )
    .bind(session.id, session.createdAt, session.expires_at, objectKey, JSON.stringify(REQUIRED_FILE_KEYS), sourceStorage)
    .run();
}

async function loadSessionRow(env: Env, sessionId: string): Promise<SessionRow> {
  const result = await env.DB.prepare(
    'SELECT id, created_at, expires_at, r2_key, source_storage FROM sessions WHERE id = ?1'
  )
    .bind(sessionId)
    .first<SessionRow>();
  if (!result?.r2_key) {
    throw createHttpError(404, 'Session not found');
  }
  return result;
}

async function loadSession(env: Env, sessionId: string): Promise<SessionData> {
  const result = await loadSessionRow(env, sessionId);
  const expiresAt = getSessionExpiry(result.created_at, result.expires_at, env);
  if (isExpiredAt(expiresAt)) {
    await deleteSession(env, sessionId);
    throw createHttpError(410, 'Session expired');
  }
  const session = await getJson<SessionData>(env, result.r2_key);
  if (!session.expires_at) {
    session.expires_at = expiresAt;
  }
  return session;
}

async function deleteSession(env: Env, sessionId: string): Promise<void> {
  const result = await env.DB.prepare('SELECT r2_key FROM sessions WHERE id = ?1').bind(sessionId).first<{ r2_key: string }>();
  if (result?.r2_key) {
    await deleteJson(env, result.r2_key);
  }
  const modelRows = await env.DB.prepare('SELECT id, r2_key FROM models WHERE session_id = ?1').bind(sessionId).all<{ id: string; r2_key: string }>();
  for (const model of modelRows.results ?? []) {
    await deleteJson(env, model.r2_key);
  }
  await env.DB.batch([
    env.DB.prepare('DELETE FROM storage_objects WHERE object_key LIKE ?1').bind(`sessions/${sessionId}.json`),
    env.DB.prepare('DELETE FROM constraints WHERE session_id = ?1').bind(sessionId),
    env.DB.prepare('DELETE FROM job_logs WHERE session_id = ?1').bind(sessionId),
    env.DB.prepare('DELETE FROM models WHERE session_id = ?1').bind(sessionId),
    env.DB.prepare('DELETE FROM sessions WHERE id = ?1').bind(sessionId),
  ]);
}

async function saveModel(env: Env, model: ModelArtifact): Promise<void> {
  const objectKey = MODEL_OBJECT_KEY(model.model_id);
  await putJson(env, objectKey, model);
  await env.DB.prepare(
    'INSERT OR REPLACE INTO models (id, session_id, created_at, r2_key, weights_json, artifact_version, training_mode, source_storage) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)'
  )
    .bind(
      model.model_id,
      model.session_id,
      model.created_at,
      objectKey,
      JSON.stringify(model.weights),
      model.artifact_version,
      model.training_mode,
      model.source_storage
    )
    .run();
}

async function loadModel(env: Env, modelId: string): Promise<ModelArtifact> {
  const result = await env.DB.prepare(
    'SELECT id, session_id, created_at, r2_key, artifact_version, training_mode, source_storage, weights_json FROM models WHERE id = ?1'
  )
    .bind(modelId)
    .first<ModelRow>();
  if (!result?.r2_key) {
    throw createHttpError(404, `Model '${modelId}' not found`);
  }
  const model = await getJson<ModelArtifact>(env, result.r2_key);
  model.artifact_version = model.artifact_version ?? result.artifact_version ?? ARTIFACT_VERSION;
  model.training_mode = model.training_mode ?? (result.training_mode as ModelArtifact['training_mode'] | null) ?? TRAINING_MODE;
  model.source_storage = model.source_storage ?? result.source_storage ?? getStorageMode(env);
  return model;
}

async function getConstraints(env: Env, sessionId: string): Promise<ConstraintRecord[]> {
  const result = await env.DB.prepare(
    'SELECT id, session_id, from_skill, to_skill, constraint_type, value, created_at FROM constraints WHERE session_id = ?1 ORDER BY created_at ASC'
  )
    .bind(sessionId)
    .all<ConstraintRecord>();
  return result.results ?? [];
}

async function addConstraint(env: Env, sessionId: string, data: any): Promise<ConstraintRecord> {
  const constraint: ConstraintRecord = {
    id: `constraint_${Date.now()}`,
    session_id: sessionId,
    from_skill: data.from_skill,
    to_skill: data.to_skill,
    constraint_type: data.constraint_type,
    value: data.value,
    created_at: new Date().toISOString(),
  };
  await env.DB.prepare(
    'INSERT INTO constraints (id, session_id, from_skill, to_skill, constraint_type, value, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)'
  )
    .bind(
      constraint.id,
      constraint.session_id,
      constraint.from_skill,
      constraint.to_skill,
      constraint.constraint_type,
      constraint.value ?? null,
      constraint.created_at
    )
    .run();
  return constraint;
}

async function removeConstraint(env: Env, sessionId: string, constraintId: string): Promise<void> {
  await env.DB.prepare('DELETE FROM constraints WHERE session_id = ?1 AND id = ?2').bind(sessionId, constraintId).run();
}

function assertCleanupAuthorized(request: Request, env: Env) {
  if (!env.CLEANUP_TOKEN) {
    throw createHttpError(501, 'CLEANUP_TOKEN is not configured');
  }
  const token = request.headers.get('x-admin-token');
  if (!token || token !== env.CLEANUP_TOKEN) {
    throw createHttpError(401, 'Invalid admin token');
  }
}

async function cleanupExpiredSessions(env: Env): Promise<{ deleted_session_ids: string[]; deleted_count: number }> {
  const now = new Date().toISOString();
  const result = await env.DB.prepare(
    'SELECT id FROM sessions WHERE expires_at IS NOT NULL AND expires_at <= ?1 ORDER BY expires_at ASC LIMIT ?2'
  )
    .bind(now, DEFAULT_CLEANUP_BATCH)
    .all<{ id: string }>();
  const deletedSessionIds: string[] = [];
  for (const row of result.results ?? []) {
    await deleteSession(env, row.id);
    deletedSessionIds.push(row.id);
  }
  return {
    deleted_session_ids: deletedSessionIds,
    deleted_count: deletedSessionIds.length,
  };
}

function parsePath(request: Request) {
  return new URL(request.url).pathname;
}

async function parseUploadForm(request: Request) {
  const formData = await request.formData();
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

  return parsedFiles as Record<FileKey, Row[]>;
}

async function handleGet(request: Request, env: Env) {
  const path = parsePath(request);
  const url = new URL(request.url);

  const sessionMatch = path.match(/^\/api\/session\/([^/]+)$/);
  if (sessionMatch) {
    const session = await loadSession(env, sessionMatch[1]);
    return jsonResponse({
      success: true,
      session_id: session.id,
      created_at: session.createdAt,
      expires_at: session.expires_at,
      source_storage: getStorageMode(env),
    });
  }

  const sessionMembersMatch = path.match(/^\/api\/session\/([^/]+)\/members$/);
  if (sessionMembersMatch) {
    const session = await loadSession(env, sessionMembersMatch[1]);
    return jsonResponse({
      success: true,
      members: session.members,
      total_count: session.members.length,
      expires_at: session.expires_at,
    });
  }

  const constraintsMatch = path.match(/^\/api\/constraints\/([^/]+)$/);
  if (constraintsMatch) {
    return jsonResponse({ constraints: await getConstraints(env, constraintsMatch[1]) });
  }

  const weightsMatch = path.match(/^\/api\/weights\/([^/]+)$/);
  if (weightsMatch) {
    const model = await loadModel(env, weightsMatch[1]);
    return jsonResponse({ success: true, weights: model.weights });
  }

  const skillsMatch = path.match(/^\/api\/skills\/([^/]+)$/);
  if (skillsMatch) {
    const model = await loadModel(env, skillsMatch[1]);
    return jsonResponse({ success: true, skills: model.skills.map((skill) => skill.skill_name) });
  }

  const modelSummaryMatch = path.match(/^\/api\/model\/([^/]+)\/summary$/);
  if (modelSummaryMatch) {
    const model = await loadModel(env, modelSummaryMatch[1]);
    return jsonResponse({
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
    });
  }

  if (path === '/api/career/members') {
    const sessionId = url.searchParams.get('session_id');
    if (!sessionId) {
      throw createHttpError(400, 'session_id is required');
    }
    const session = await loadSession(env, sessionId);
    return jsonResponse({
      success: true,
      members: session.members.map((member) => ({
        ...member,
        skill_count: session.acquired.filter((record) => record.member_code === member.member_code).length,
      })),
    });
  }

  if (path === '/api/career/roles') {
    const sessionId = url.searchParams.get('session_id');
    if (!sessionId) {
      throw createHttpError(400, 'session_id is required');
    }
    const session = await loadSession(env, sessionId);
    const roleCounts = new Map<string, number>();
    session.members.forEach((member) => {
      roleCounts.set(member.role, (roleCounts.get(member.role) ?? 0) + 1);
    });
    return jsonResponse({
      success: true,
      roles: Array.from(roleCounts.entries())
        .map(([role_name, member_count]) => ({ role_name, member_count }))
        .sort((a, b) => b.member_count - a.member_count),
    });
  }

  throw createHttpError(404, `Unknown endpoint: ${path}`);
}

async function handlePost(request: Request, env: Env) {
  const path = parsePath(request);

  if (path === '/api/upload') {
    const files = await parseUploadForm(request);
    const session = buildSessionData(files, getSessionTtlDays(env));
    await saveSession(env, session);
    return jsonResponse({
      session_id: session.id,
      message: 'Files uploaded successfully',
      files: REQUIRED_FILE_KEYS,
      expires_at: session.expires_at,
      source_storage: getStorageMode(env),
      training_mode: TRAINING_MODE,
    });
  }

  const body = await request.json<any>().catch(() => ({}));

  if (path === '/api/admin/cleanup') {
    assertCleanupAuthorized(request, env);
    const cleanupResult = await cleanupExpiredSessions(env);
    return jsonResponse({ success: true, ...cleanupResult, executed_at: new Date().toISOString() });
  }

  if (path === '/api/train') {
    if (!body.session_id) {
      throw createHttpError(400, 'session_id is required');
    }
    const session = await loadSession(env, body.session_id);
    const startedAt = Date.now();
    const modelId = generateModelId(body.session_id);
    await writeJobLog(env, {
      sessionId: body.session_id,
      modelId,
      jobType: 'train',
      status: 'started',
      detail: 'Cloudflare approximate training started',
    });
    try {
      const constraints = await getConstraints(env, body.session_id);
      const model = buildArtifact(session, {
        modelId,
        weights: body.weights,
        minMembersPerSkill: body.min_members_per_skill ?? body.min_members ?? 5,
        correlationThreshold: body.correlation_threshold ?? 0.2,
        constraints,
        sourceStorage: getStorageMode(env),
      });
      await saveModel(env, model);
      const learningTime = Number(((Date.now() - startedAt) / 1000).toFixed(3));
      await writeJobLog(env, {
        sessionId: body.session_id,
        modelId: model.model_id,
        jobType: 'train',
        status: 'completed',
        detail: `Cloudflare approximate training completed in ${learningTime}s`,
      });
      return jsonResponse({
        success: true,
        model_id: model.model_id,
        message: 'Cloudflare D1/R2 上で近似学習モデルを保存しました',
        training_mode: model.training_mode,
        artifact_version: model.artifact_version,
        source_storage: model.source_storage,
        expires_at: session.expires_at,
        summary: {
          model_id: model.model_id,
          session_id: model.session_id,
          num_members: model.members.length,
          num_skills: model.skills.length,
          learning_time: learningTime,
          total_time: learningTime,
          weights: model.weights,
          min_members_per_skill: model.min_members_per_skill,
          correlation_threshold: model.correlation_threshold,
          training_mode: model.training_mode,
          artifact_version: model.artifact_version,
          source_storage: model.source_storage,
          expires_at: session.expires_at,
        },
      });
    } catch (error: any) {
      await writeJobLog(env, {
        sessionId: body.session_id,
        modelId,
        jobType: 'train',
        status: 'failed',
        detail: error?.message ?? 'Training failed',
      });
      throw error;
    }
  }

  if (path === '/api/recommend') {
    const model = await loadModel(env, body.model_id);
    const session = await loadSession(env, model.session_id);
    const recommendations = scoreRecommendations(model, body.member_id, body.top_n ?? 10);
    return jsonResponse({
      success: true,
      model_id: model.model_id,
      member_id: body.member_id,
      member_name: pickMember(session, body.member_id).member_name,
      recommendations,
      metadata: { weights: model.weights, total_candidates: recommendations.length },
      message: recommendations.length > 0 ? '推薦を生成しました' : '利用可能な推薦がありません',
    });
  }

  if (path === '/api/scatter-plot') {
    const model = await loadModel(env, body.model_id);
    const recommendations = scoreRecommendations(model, body.member_id, model.skills.length);
    return jsonResponse({
      success: true,
      model_id: model.model_id,
      member_id: body.member_id,
      skills: recommendations.map((recommendation) => ({
        skill_code: recommendation.skill_code,
        skill_name: recommendation.skill_name,
        category: recommendation.category,
        readiness_score: recommendation.readiness_score,
        utility_score: recommendation.utility_score,
        final_score: recommendation.final_score,
      })),
      metadata: { weights: model.weights, total_skills: recommendations.length },
    });
  }

  if (path === '/api/graph/ego') {
    const model = await loadModel(env, body.model_id);
    const centerCode =
      body.skill_code ??
      body.competence_code ??
      model.skills.find((skill) => skill.skill_name === body.center_node)?.skill_code ??
      body.center_node;
    if (!centerCode) {
      throw createHttpError(400, 'center_node is required');
    }
    const centerSkill = model.skills.find((skill) => skill.skill_code === centerCode);
    if (!centerSkill) {
      throw createHttpError(404, 'Skill not found in model');
    }
    const radius = body.radius ?? 1;
    const threshold = body.threshold ?? 0.05;
    const memberSkills = new Set<string>(body.member_skills ?? []);
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
    return jsonResponse({ success: true, html: buildGraphHtml(`${centerSkill.skill_name} の関連スキル`, nodes, edges), graph_data: { nodes, edges } });
  }

  if (path === '/api/graph/full') {
    const model = await loadModel(env, body.model_id);
    const threshold = body.threshold ?? 0.3;
    const topN = Math.min(body.top_n ?? 20, model.skills.length);
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
    return jsonResponse({ success: true, html: buildGraphHtml('因果グラフ全体', nodes, edges), node_count: nodes.length, graph_data: { nodes, edges } });
  }

  if (path === '/api/weights/update') {
    const model = await loadModel(env, body.model_id);
    const updated = { ...model, weights: normalizeWeights(body.weights) };
    await saveModel(env, updated);
    return jsonResponse({ success: true, model_id: updated.model_id, weights: updated.weights });
  }

  if (path === '/api/weights/optimize') {
    const model = await loadModel(env, body.model_id);
    const utilityBias = model.skills.length > 200 ? 0.25 : 0.15;
    return jsonResponse({
      success: true,
      optimized_weights: normalizeWeights({ readiness: 0.6, bayesian: 0.25 - utilityBias / 2, utility: 0.15 + utilityBias }),
    });
  }

  if (path === '/api/career/member-skills') {
    const session = await loadSession(env, body.session_id);
    const artifact = body.model_id ? await loadModel(env, body.model_id).catch(() => null) : null;
    const model = artifact ?? buildArtifact(session, { modelId: 'ephemeral', minMembersPerSkill: 1, correlationThreshold: 0 });
    return jsonResponse({ success: true, current_skills: memberCurrentSkills(model, body.member_code) });
  }

  if (path === '/api/career/gap-analysis' || path === '/api/career/role/gap-analysis') {
    const session = await loadSession(env, body.session_id);
    const model = await loadModel(env, body.model_id);
    let targetSkillCodes: string[];
    let targetMember;
    if (path.endsWith('/role/gap-analysis')) {
      targetSkillCodes = getRoleSkills(session, body.target_role, body.min_frequency ?? 0.1).map((skill) => skill!.skill_code);
    } else {
      targetMember = pickMember(session, body.target_member_code);
      targetSkillCodes = model.member_skill_codes[body.target_member_code] ?? [];
    }
    const base = buildGapAnalysis(session, model, body.source_member_code, targetSkillCodes);
    return jsonResponse({
      success: true,
      ...base,
      target_member:
        targetMember ??
        {
          member_code: '',
          member_name: path.endsWith('/role/gap-analysis') ? body.target_role : '',
          role: body.target_role ?? '',
          grade: '',
          occupation: '',
          display_name: body.target_role ?? '',
        },
      target_skill_count: targetSkillCodes.length,
    });
  }

  if (path === '/api/career/career-path' || path === '/api/career/role/career-path') {
    const session = await loadSession(env, body.session_id);
    const model = await loadModel(env, body.model_id);
    const targetSkillCodes = path.endsWith('/role/career-path')
      ? getRoleSkills(session, body.target_role, body.min_frequency ?? 0.1).map((skill) => skill!.skill_code)
      : model.member_skill_codes[body.target_member_code] ?? [];
    return jsonResponse(buildCareerPathResponse(model, body.source_member_code, targetSkillCodes, body.min_total_score ?? 0, body.min_readiness_score ?? 0));
  }

  if (path === '/api/career/career-roadmap' || path === '/api/career/role/career-roadmap') {
    const session = await loadSession(env, body.session_id);
    const model = await loadModel(env, body.model_id);
    const targetSkillCodes = path.endsWith('/role/career-roadmap')
      ? getRoleSkills(session, body.target_role, body.min_frequency ?? 0.1).map((skill) => skill!.skill_code)
      : model.member_skill_codes[body.target_member_code] ?? [];
    const pathResponse = buildCareerPathResponse(model, body.source_member_code, targetSkillCodes, body.min_total_score ?? 0, body.min_readiness_score ?? 0);
    return jsonResponse({ success: true, gantt_chart: buildRoadmapChart(pathResponse.recommended_skills) });
  }

  if (path === '/api/career/role/role-skills') {
    const session = await loadSession(env, body.session_id);
    const skills = getRoleSkills(session, body.role_name, body.min_frequency ?? 0.1);
    return jsonResponse({
      success: true,
      role_name: body.role_name,
      total_members: session.members.filter((member) => member.role === body.role_name).length,
      skills,
      skill_count: skills.length,
    });
  }

  if (path.startsWith('/api/constraints/') && path.endsWith('/apply')) {
    const sessionId = path.split('/')[3];
    const session = await loadSession(env, sessionId);
    const currentModel = await loadModel(env, body.model_id);
    const constraints = await getConstraints(env, sessionId);
    const updated = buildArtifact(session, {
      modelId: currentModel.model_id,
      weights: currentModel.weights,
      minMembersPerSkill: currentModel.min_members_per_skill,
      correlationThreshold: currentModel.correlation_threshold,
      constraints,
    });
    await saveModel(env, updated);
    return jsonResponse({ success: true, applied_count: constraints.length, skipped_count: 0 });
  }

  if (path.startsWith('/api/constraints/')) {
    const sessionId = path.split('/')[3];
    const constraint = await addConstraint(env, sessionId, body);
    return jsonResponse({ success: true, constraint });
  }

  if (path === '/api/organizational/metrics') {
    const session = await loadSession(env, body.session_id);
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
    return jsonResponse({
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
    });
  }

  if (path === '/api/organizational/skill-gap') {
    const session = await loadSession(env, body.session_id);
    const percentile = Number(body.percentile ?? 0.2);
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
    return jsonResponse({ success: true, gap_analysis: gaps, critical_skills: gaps.filter((gap) => gap.gap_percentage >= 0.3).slice(0, 5) });
  }

  if (path === '/api/organizational/succession') {
    const session = await loadSession(env, body.session_id);
    const targetPosition = String(body.target_position ?? '');
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
      .slice(0, Number(body.max_candidates ?? 20));
    return jsonResponse({ success: true, candidates });
  }

  throw createHttpError(404, `Unknown endpoint: ${path}`);
}

async function handleDelete(request: Request, env: Env) {
  const path = parsePath(request);
  const sessionMatch = path.match(/^\/api\/session\/([^/]+)$/);
  if (sessionMatch) {
    await deleteSession(env, sessionMatch[1]);
    return jsonResponse({ success: true });
  }

  const constraintMatch = path.match(/^\/api\/constraints\/([^/]+)\/([^/]+)$/);
  if (constraintMatch) {
    await removeConstraint(env, constraintMatch[1], constraintMatch[2]);
    return jsonResponse({ success: true });
  }

  throw createHttpError(404, `Unknown endpoint: ${path}`);
}

export async function onRequest(context: any) {
  const { request, env } = context as { request: Request; env: Env };
  try {
    switch (request.method.toUpperCase()) {
      case 'GET':
        return await handleGet(request, env);
      case 'POST':
        return await handlePost(request, env);
      case 'DELETE':
        return await handleDelete(request, env);
      default:
        return jsonResponse({ detail: `Method ${request.method} is not supported` }, 405);
    }
  } catch (error: any) {
    const status = error?.response?.status ?? error?.statusCode ?? 500;
    const detail = error?.response?.data?.detail ?? error?.message ?? 'Unexpected error';
    return jsonResponse({ detail }, status);
  }
}
