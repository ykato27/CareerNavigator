import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  User,
  Briefcase,
  Target,
  Award,
  TrendingUp,
  Loader2,
  AlertCircle,
  ChevronDown,
  ChevronUp,
  Download,
  Info,
  Network,
  BarChart3,
  Settings
} from 'lucide-react';

// =========================================================
// Type Definitions
// =========================================================

interface MemberInfo {
  member_code: string;
  member_name: string;
  role: string;
  skill_count?: number;
  display_name?: string;
}

interface Skill {
  skill_code: string;
  skill_name: string;
  category: string;
  level: number;
}

interface RecommendedSkill {
  competence_code: string;
  competence_name: string;
  category: string;
  total_score: number;
  readiness_score: number;
  bayesian_score: number;
  utility_score: number;
  readiness_reasons: [string, number][];
  utility_reasons: [string, number][];
  prerequisites: { skill_name: string; effect: number }[];
  enables: { skill_name: string; effect: number }[];
  explanation?: string;
}

interface GapAnalysisResult {
  success: boolean;
  source_member: MemberInfo;
  target_member: MemberInfo;
  gap_skills: { skill_code: string; skill_name: string; category: string }[];
  gap_count: number;
  source_skill_count: number;
  target_skill_count: number;
  completion_rate: number;
}

interface CareerPathResult {
  success: boolean;
  recommended_skills: RecommendedSkill[];
  skill_count: number;
  avg_score: number;
  total_dependencies: number;
  estimated_months: number;
  message: string;
}

interface RoleSkillStats {
  success: boolean;
  role_name: string;
  total_members: number;
  skills: {
    skill_code: string;
    skill_name: string;
    category: string;
    frequency: number;
    member_count: number;
    priority: string;
  }[];
  skill_count: number;
}

// =========================================================
// Main Component
// =========================================================

export const EmployeeCareerDashboard = () => {
  // Session state
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [dataUploaded, setDataUploaded] = useState(false);
  const [modelId, setModelId] = useState('');

  // Member selection
  const [availableMembers, setAvailableMembers] = useState<MemberInfo[]>([]);
  const [selectedMember, setSelectedMember] = useState('');
  const [loadingMembers, setLoadingMembers] = useState(false);

  // Target selection
  const [targetSelectionMode, setTargetSelectionMode] = useState<'role_model' | 'role'>('role_model');
  const [targetMember, setTargetMember] = useState('');
  const [availableRoles, setAvailableRoles] = useState<{ role_name: string; member_count: number }[]>([]);
  const [selectedRole, setSelectedRole] = useState('');

  // Role based settings
  const [minRoleFrequency, setMinRoleFrequency] = useState(0.1);
  const [priorityHighThreshold, setPriorityHighThreshold] = useState(0.5);
  const [priorityMediumThreshold, setPriorityMediumThreshold] = useState(0.3);
  const [roleStats, setRoleStats] = useState<RoleSkillStats | null>(null);
  const [loadingRoleStats, setLoadingRoleStats] = useState(false);

  // Current skills
  const [currentSkills, setCurrentSkills] = useState<Skill[]>([]);
  const [loadingSkills, setLoadingSkills] = useState(false);

  // Gap analysis
  const [gapAnalysis, setGapAnalysis] = useState<GapAnalysisResult | null>(null);
  const [loadingGap, setLoadingGap] = useState(false);

  // Career path
  const [careerPath, setCareerPath] = useState<CareerPathResult | null>(null);
  const [loadingPath, setLoadingPath] = useState(false);

  // Filtering settings
  const [minTotalScore, setMinTotalScore] = useState(0.3);
  const [minReadinessScore, setMinReadinessScore] = useState(0.0);
  const [minEffectThreshold, setMinEffectThreshold] = useState(0.03);
  const [showFilterSettings, setShowFilterSettings] = useState(false);

  // UI state
  const [error, setError] = useState('');
  const [expandedSkillIndex, setExpandedSkillIndex] = useState<number | null>(null);

  // Graph visualization state
  const [graphHtml, setGraphHtml] = useState<string | null>(null);
  const [loadingGraph, setLoadingGraph] = useState(false);
  const [graphRadius, setGraphRadius] = useState(1);
  const [graphThreshold, setGraphThreshold] = useState(0.05);
  const [showGraphSettings, setShowGraphSettings] = useState(false);
  const [memberSkills, setMemberSkills] = useState<string[]>([]);

  // =========================================================
  // Initialize session from sessionStorage
  // =========================================================
  useEffect(() => {
    const sid = sessionStorage.getItem('career_session_id');
    const uploaded = sessionStorage.getItem('career_data_uploaded');
    const savedModelId = sessionStorage.getItem('career_model_id');

    setSessionId(sid);
    setDataUploaded(uploaded === 'true');
    if (savedModelId) {
      setModelId(savedModelId);
    }

    if (sid && uploaded === 'true') {
      loadAvailableMembers(sid);
      loadAvailableRoles(sid);
    }
  }, []);

  // =========================================================
  // Load available members
  // =========================================================
  const loadAvailableMembers = async (sid: string) => {
    setLoadingMembers(true);
    try {
      const response = await axios.get(`http://localhost:8000/api/career/members`, {
        params: { session_id: sid }
      });
      setAvailableMembers(response.data.members);
    } catch (err: any) {
      console.error('Failed to load members:', err);
      setError('メンバーリストの読み込みに失敗しました');
    } finally {
      setLoadingMembers(false);
    }
  };

  // =========================================================
  // Load available roles
  // =========================================================
  const loadAvailableRoles = async (sid: string) => {
    try {
      const response = await axios.get(`http://localhost:8000/api/career/roles`, {
        params: { session_id: sid }
      });
      if (response.data.success) {
        setAvailableRoles(response.data.roles);
      }
    } catch (err: any) {
      console.error('Failed to load roles:', err);
    }
  };

  // =========================================================
  // Load member's current skills
  // =========================================================
  const loadMemberSkills = async (memberCode: string) => {
    if (!sessionId) return;

    setLoadingSkills(true);
    setError('');

    try {
      const response = await axios.post(`http://localhost:8000/api/career/member-skills`, {
        session_id: sessionId,
        member_code: memberCode
      });

      if (response.data.success) {
        setCurrentSkills(response.data.current_skills);
      }
    } catch (err: any) {
      setError('スキル情報の読み込みに失敗しました');
      console.error('Failed to load member skills:', err);
    } finally {
      setLoadingSkills(false);
    }
  };

  // =========================================================
  // Load member skills for graph highlighting
  // =========================================================
  const loadMemberSkillsForGraph = async (memberCode: string) => {
    if (!sessionId) return;

    try {
      const response = await axios.post('http://localhost:8000/api/career/member-skills', {
        session_id: sessionId,
        member_code: memberCode
      });

      // Extract skill names from the response
      const skillNames = response.data.current_skills.map((skill: any) => skill.skill_name);
      setMemberSkills(skillNames);
    } catch (err: any) {
      console.error('Failed to load member skills for graph:', err);
      setMemberSkills([]);
    }
  };

  // =========================================================
  // Load ego graph for a specific skill
  // =========================================================
  const loadEgoGraph = async (skillName: string) => {
    if (!modelId) return;

    setLoadingGraph(true);

    try {
      const response = await axios.post('http://localhost:8000/api/graph/ego', {
        model_id: modelId,
        center_node: skillName,
        radius: graphRadius,
        threshold: graphThreshold,
        show_negative: false,
        member_skills: memberSkills
      });

      setGraphHtml(response.data.html);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'グラフの生成に失敗しました');
    } finally {
      setLoadingGraph(false);
    }
  };

  // =========================================================
  // Handle member selection
  // =========================================================
  useEffect(() => {
    if (selectedMember) {
      loadMemberSkills(selectedMember);
      loadMemberSkillsForGraph(selectedMember);
    }
  }, [selectedMember]);

  // =========================================================
  // Perform gap analysis
  // =========================================================
  const performGapAnalysis = async () => {
    if (!sessionId || !modelId || !selectedMember) {
      setError('必要な情報が入力されていません');
      return;
    }

    if (targetSelectionMode === 'role_model' && !targetMember) {
      setError('目標メンバーを選択してください');
      return;
    }

    if (targetSelectionMode === 'role' && !selectedRole) {
      setError('目標役職を選択してください');
      return;
    }

    setLoadingGap(true);
    setError('');
    setGapAnalysis(null);
    setCareerPath(null);

    try {
      let endpoint = `http://localhost:8000/api/career/gap-analysis`;
      let payload: any = {
        session_id: sessionId,
        model_id: modelId,
        source_member_code: selectedMember,
        min_total_score: minTotalScore,
        min_readiness_score: minReadinessScore
      };

      if (targetSelectionMode === 'role') {
        endpoint = `http://localhost:8000/api/career/role/gap-analysis`;
        payload.target_role = selectedRole;
        payload.min_frequency = minRoleFrequency;
      } else {
        payload.target_member_code = targetMember;
      }

      const response = await axios.post(endpoint, payload);

      setGapAnalysis(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'ギャップ分析に失敗しました');
    } finally {
      setLoadingGap(false);
    }
  };
  // =========================================================
  // Generate career path
  // =========================================================
  const generateCareerPath = async () => {
    if (!sessionId || !modelId || !selectedMember) return;
    if (targetSelectionMode === 'role_model' && !targetMember) return;
    if (targetSelectionMode === 'role' && !selectedRole) return;

    setLoadingPath(true);
    setError('');
    setCareerPath(null);

    try {
      let endpoint = `http://localhost:8000/api/career/career-path`;
      let payload: any = {
        session_id: sessionId,
        model_id: modelId,
        source_member_code: selectedMember,
        min_total_score: minTotalScore,
        min_readiness_score: minReadinessScore,
        min_effect_threshold: minEffectThreshold
      };

      if (targetSelectionMode === 'role') {
        endpoint = `http://localhost:8000/api/career/role/career-path`;
        payload.target_role = selectedRole;
        payload.min_frequency = minRoleFrequency;
      } else {
        payload.target_member_code = targetMember;
      }

      const response = await axios.post(endpoint, payload);

      setCareerPath(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'キャリアパス生成に失敗しました');
    } finally {
      setLoadingPath(false);
    }
  };

  // =========================================================
  // Handle analyze button click
  // =========================================================
  const handleAnalyze = async () => {
    await performGapAnalysis();
    await generateCareerPath();
  };

  // =========================================================
  // Fetch role stats when role is selected
  // =========================================================
  useEffect(() => {
    const fetchRoleStats = async () => {
      if (targetSelectionMode === 'role' && selectedRole && sessionId) {
        setLoadingRoleStats(true);
        try {
          const response = await axios.post('http://localhost:8000/api/career/role/role-skills', {
            session_id: sessionId,
            role_name: selectedRole,
            min_frequency: minRoleFrequency
          });
          setRoleStats(response.data);
        } catch (err) {
          console.error('Error fetching role stats:', err);
        } finally {
          setLoadingRoleStats(false);
        }
      } else {
        setRoleStats(null);
      }
    };

    fetchRoleStats();
  }, [selectedRole, targetSelectionMode, sessionId, minRoleFrequency]);

  // =========================================================
  // Download CSV
  // =========================================================
  const downloadCSV = () => {
    if (!careerPath || !careerPath.recommended_skills) return;

    const headers = ['力量名', 'カテゴリー', '総合スコア', '準備完了度', '確率', '有用性', '前提数', '次へ'];
    const rows = careerPath.recommended_skills.map(skill => [
      skill.competence_name,
      skill.category,
      skill.total_score.toFixed(3),
      skill.readiness_score.toFixed(3),
      skill.bayesian_score.toFixed(3),
      skill.utility_score.toFixed(3),
      skill.prerequisites.length,
      skill.enables.length
    ]);

    const csvContent = [
      headers.join(','),
      ...rows.map(row => row.join(','))
    ].join('\n');

    const blob = new Blob(['\uFEFF' + csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `career_path_${selectedMember}.csv`;
    link.click();
  };

  // =========================================================
  // Render: Data not loaded
  // =========================================================
  if (!dataUploaded) {
    return (
      <div className="flex-1 px-8 py-12">
        <div className="max-w-4xl mx-auto">
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 flex items-start gap-3">
            <AlertCircle size={24} className="text-yellow-600 flex-shrink-0 mt-0.5" />
            <div>
              <h3 className="font-semibold text-yellow-800 mb-2">データが読み込まれていません</h3>
              <p className="text-yellow-700 mb-3">
                まず「データ管理」ページで6種類のCSVファイルをアップロードし、
                「モデル学習」ページでモデルを学習してください。
              </p>
              <button
                onClick={() => (window.location.href = '/data-upload')}
                className="bg-yellow-600 text-white px-4 py-2 rounded-md hover:bg-yellow-700 transition-colors"
              >
                データ管理ページへ
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // =========================================================
  // Main Render
  // =========================================================
  return (
    <div className="flex-1 px-8 py-12">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <User size={32} className="text-[#00A968]" />
            <h1 className="text-3xl font-bold text-gray-800">従業員キャリアダッシュボード</h1>
          </div>
          <p className="text-gray-600">
            個人のキャリアパスを可視化し、目標達成に向けた最適な学習計画を提案します
          </p>
        </div>

        {/* Info Box */}
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-6 mb-6">
          <div className="flex items-start gap-3">
            <Info size={20} className="text-blue-600 flex-shrink-0 mt-0.5" />
            <div className="text-sm text-blue-800">
              <p className="font-semibold mb-2">このダッシュボードでできること</p>
              <ul className="space-y-1 list-disc list-inside">
                <li><strong>現状分析</strong>: あなたの保有スキルを可視化</li>
                <li><strong>目標設定</strong>: ロールモデルまたは職種を選択</li>
                <li><strong>ギャップ分析</strong>: 目標との差分を自動抽出</li>
                <li><strong>Causal学習ロードマップ</strong>: 因果関係ベースの効率的な学習順序を提案</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Member Selection */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
          <h2 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
            <User size={20} className="text-[#00A968]" />
            メンバー選択
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">モデルID</label>
              <input
                type="text"
                value={modelId}
                onChange={(e) => setModelId(e.target.value)}
                placeholder="例: model_001"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-[#00A968]"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">分析対象メンバー（あなた）</label>
              <select
                value={selectedMember}
                onChange={(e) => setSelectedMember(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-[#00A968]"
                disabled={loadingMembers}
              >
                <option value="">メンバーを選択してください</option>
                {availableMembers.map((member) => (
                  <option key={member.member_code} value={member.member_code}>
                    {member.display_name}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">現在の保有スキル</label>
              <div className="w-full px-3 py-2 border border-gray-200 rounded-md bg-gray-50">
                {loadingSkills ? (
                  <span className="text-gray-500">読み込み中...</span>
                ) : (
                  <span className="font-bold text-[#00A968]">{currentSkills.length}件</span>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Target Selection */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
          <h2 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
            <Target size={20} className="text-[#00A968]" />
            キャリア目標の設定
          </h2>

          {/* Selection Mode */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">目標設定方法</label>
            <div className="flex gap-4">
              <label className="flex items-center">
                <input
                  type="radio"
                  value="role_model"
                  checked={targetSelectionMode === 'role_model'}
                  onChange={(e) => setTargetSelectionMode(e.target.value as 'role_model' | 'role')}
                  className="mr-2"
                />
                ロールモデルから選ぶ
              </label>
              <label className="flex items-center">
                <input
                  type="radio"
                  value="role"
                  checked={targetSelectionMode === 'role'}
                  onChange={(e) => setTargetSelectionMode(e.target.value as 'role_model' | 'role')}
                  className="mr-2"
                />
                職種・役職から選ぶ
              </label>
            </div>
          </div>

          {/* Target Member or Role Selection */}
          {targetSelectionMode === 'role_model' ? (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">ロールモデル（目標メンバー）</label>
              <select
                value={targetMember}
                onChange={(e) => setTargetMember(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-[#00A968]"
                disabled={loadingMembers}
              >
                <option value="">目標メンバーを選択してください</option>
                {availableMembers
                  .filter((m) => m.member_code !== selectedMember)
                  .map((member) => (
                    <option key={member.member_code} value={member.member_code}>
                      {member.display_name}
                    </option>
                  ))}
              </select>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">目標役職</label>
                  <select
                    value={selectedRole}
                    onChange={(e) => setSelectedRole(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-[#00A968]"
                  >
                    <option value="">役職を選択してください</option>
                    {availableRoles.map((role) => (
                      <option key={role.role_name} value={role.role_name}>
                        {role.role_name} ({role.member_count}人)
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">役職分析情報</label>
                  <div className="w-full px-3 py-2 border border-gray-200 rounded-md bg-gray-50 min-h-[42px]">
                    {loadingRoleStats ? (
                      <div className="flex items-center gap-2 text-gray-500">
                        <Loader2 size={16} className="animate-spin" />
                        <span>分析中...</span>
                      </div>
                    ) : roleStats ? (
                      <div className="text-sm">
                        <span className="font-bold text-[#00A968]">{roleStats.total_members}人</span>のデータを分析
                        <span className="mx-2 text-gray-300">|</span>
                        目標スキル: <span className="font-bold text-[#00A968]">{roleStats.skill_count}件</span>
                      </div>
                    ) : (
                      <span className="text-gray-500">役職を選択してください</span>
                    )}
                  </div>
                </div>
              </div>

              {/* Role Settings Sliders */}
              {
                selectedRole && (
                  <div className="bg-gray-50 p-4 rounded-md border border-gray-200">
                    <h3 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
                      <Settings size={16} />
                      役職分析設定
                    </h3>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <label className="block text-xs font-medium text-gray-600 mb-1">
                          最小スキル保有率: {(minRoleFrequency * 100).toFixed(0)}%
                        </label>
                        <input
                          type="range"
                          min="0"
                          max="1"
                          step="0.05"
                          value={minRoleFrequency}
                          onChange={(e) => setMinRoleFrequency(parseFloat(e.target.value))}
                          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-[#00A968]"
                        />
                        <p className="text-xs text-gray-500 mt-1">これ以上の保有率のスキルを目標に含めます</p>
                      </div>

                      <div className="space-y-4">
                        <div>
                          <label className="block text-xs font-medium text-gray-600 mb-1">
                            🔴 必須スキル閾値: {(priorityHighThreshold * 100).toFixed(0)}%
                          </label>
                          <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.05"
                            value={priorityHighThreshold}
                            onChange={(e) => setPriorityHighThreshold(parseFloat(e.target.value))}
                            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-red-500"
                          />
                        </div>
                        <div>
                          <label className="block text-xs font-medium text-gray-600 mb-1">
                            🟡 推奨スキル閾値: {(priorityMediumThreshold * 100).toFixed(0)}%
                          </label>
                          <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.05"
                            value={priorityMediumThreshold}
                            onChange={(e) => setPriorityMediumThreshold(parseFloat(e.target.value))}
                            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-yellow-500"
                          />
                        </div>
                      </div>
                    </div>

                    {/* Priority Distribution */}
                    {roleStats && (
                      <div className="mt-4 grid grid-cols-3 gap-2 text-center text-xs">
                        <div className="bg-red-50 p-2 rounded border border-red-100">
                          <div className="font-bold text-red-700">必須</div>
                          <div>{roleStats.skills.filter(s => s.frequency >= priorityHighThreshold).length}件</div>
                        </div>
                        <div className="bg-yellow-50 p-2 rounded border border-yellow-100">
                          <div className="font-bold text-yellow-700">推奨</div>
                          <div>{roleStats.skills.filter(s => s.frequency >= priorityMediumThreshold && s.frequency < priorityHighThreshold).length}件</div>
                        </div>
                        <div className="bg-green-50 p-2 rounded border border-green-100">
                          <div className="font-bold text-green-700">任意</div>
                          <div>{roleStats.skills.filter(s => s.frequency < priorityMediumThreshold).length}件</div>
                        </div>
                      </div>
                    )}
                  </div>
                )
              }
            </div>
          )}
        </div>

        {/* Filter Settings */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
          <button
            onClick={() => setShowFilterSettings(!showFilterSettings)}
            className="w-full flex items-center justify-between text-left"
            type="button"
          >
            <div className="flex items-center gap-2">
              <Settings size={20} className="text-[#00A968]" />
              <h2 className="text-lg font-semibold text-gray-800">フィルタリング設定（オプション）</h2>
            </div>
            {showFilterSettings ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
          </button>

          {
            showFilterSettings && (
              <div className="mt-4 pt-4 border-t border-gray-200">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      総合スコア最小値 ({minTotalScore.toFixed(2)})
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={minTotalScore}
                      onChange={(e) => setMinTotalScore(parseFloat(e.target.value))}
                      className="w-full"
                    />
                    <p className="text-xs text-gray-500 mt-1">推薦スキルの最小スコア</p>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      準備完了度最小値 ({minReadinessScore.toFixed(2)})
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={minReadinessScore}
                      onChange={(e) => setMinReadinessScore(parseFloat(e.target.value))}
                      className="w-full"
                    />
                    <p className="text-xs text-gray-500 mt-1">準備ができているスキルを優先</p>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      依存関係閾値 ({minEffectThreshold.toFixed(2)})
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="0.5"
                      step="0.01"
                      value={minEffectThreshold}
                      onChange={(e) => setMinEffectThreshold(parseFloat(e.target.value))}
                      className="w-full"
                    />
                    <p className="text-xs text-gray-500 mt-1">スキル間の依存関係判定閾値</p>
                  </div>
                </div>
              </div>
            )
          }
        </div >

        {/* Analyze Button */}
        < div className="mb-6" >
          <button
            onClick={handleAnalyze}
            disabled={!selectedMember || !targetMember || !modelId || loadingGap || loadingPath}
            className="w-full bg-[#00A968] text-white py-4 rounded-md font-medium hover:bg-[#008F58] transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 text-lg"
          >
            {loadingGap || loadingPath ? (
              <>
                <Loader2 size={24} className="animate-spin" />
                分析中...
              </>
            ) : (
              <>
                <BarChart3 size={24} />
                キャリアパスを分析
              </>
            )}
          </button>
        </div >

        {/* Error Message */}
        {
          error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6 flex items-start gap-3">
              <AlertCircle size={20} className="text-red-600 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-red-800 font-medium">エラー</p>
                <p className="text-red-700 text-sm">{error}</p>
              </div>
            </div>
          )
        }

        {/* Gap Analysis Results */}
        {
          gapAnalysis && (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
              <h2 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
                <Target size={20} className="text-[#00A968]" />
                ギャップ分析結果
              </h2>

              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
                <div className="bg-blue-50 rounded-lg p-4">
                  <p className="text-sm text-gray-600 mb-1">現在の保有スキル</p>
                  <p className="text-2xl font-bold text-blue-600">{gapAnalysis.source_skill_count}</p>
                </div>
                <div className="bg-green-50 rounded-lg p-4">
                  <p className="text-sm text-gray-600 mb-1">目標スキル数</p>
                  <p className="text-2xl font-bold text-green-600">{gapAnalysis.target_skill_count}</p>
                </div>
                <div className="bg-orange-50 rounded-lg p-4">
                  <p className="text-sm text-gray-600 mb-1">不足スキル</p>
                  <p className="text-2xl font-bold text-orange-600">{gapAnalysis.gap_count}</p>
                </div>
                <div className="bg-purple-50 rounded-lg p-4">
                  <p className="text-sm text-gray-600 mb-1">達成率</p>
                  <p className="text-2xl font-bold text-purple-600">{gapAnalysis.completion_rate.toFixed(1)}%</p>
                </div>
              </div>

              <div className="bg-gray-50 rounded-lg p-4">
                <p className="text-sm font-medium text-gray-700 mb-2">
                  {gapAnalysis.source_member.member_name} → {gapAnalysis.target_member.member_name} (
                  {gapAnalysis.target_member.role})
                </p>
                <div className="h-4 bg-gray-200 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-[#00A968]"
                    style={{ width: `${gapAnalysis.completion_rate}%` }}
                  />
                </div>
              </div>
            </div>
          )
        }

        {/* Career Path Results */}
        {
          careerPath && careerPath.recommended_skills && careerPath.recommended_skills.length > 0 && (
            <>
              {/* Summary Metrics */}
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
                <h2 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
                  <Briefcase size={20} className="text-[#00A968]" />
                  Causal学習ロードマップ
                </h2>

                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  <div className="bg-blue-50 rounded-lg p-4 text-center">
                    <Award size={32} className="text-blue-600 mx-auto mb-2" />
                    <p className="text-sm text-gray-600 mb-1">推薦スキル数</p>
                    <p className="text-2xl font-bold text-blue-600">{careerPath.skill_count}</p>
                  </div>
                  <div className="bg-green-50 rounded-lg p-4 text-center">
                    <TrendingUp size={32} className="text-green-600 mx-auto mb-2" />
                    <p className="text-sm text-gray-600 mb-1">平均スコア</p>
                    <p className="text-2xl font-bold text-green-600">{careerPath.avg_score.toFixed(2)}</p>
                  </div>
                  <div className="bg-purple-50 rounded-lg p-4 text-center">
                    <Network size={32} className="text-purple-600 mx-auto mb-2" />
                    <p className="text-sm text-gray-600 mb-1">依存関係</p>
                    <p className="text-2xl font-bold text-purple-600">{careerPath.total_dependencies}</p>
                  </div>
                  <div className="bg-orange-50 rounded-lg p-4 text-center">
                    <Briefcase size={32} className="text-orange-600 mx-auto mb-2" />
                    <p className="text-sm text-gray-600 mb-1">推定期間</p>
                    <p className="text-2xl font-bold text-orange-600">{careerPath.estimated_months}ヶ月</p>
                  </div>
                </div>

                <div className="mt-4 flex justify-end">
                  <button
                    onClick={downloadCSV}
                    className="flex items-center gap-2 px-4 py-2 bg-[#00A968] text-white rounded-md hover:bg-[#008F58] transition-colors"
                  >
                    <Download size={18} />
                    CSVダウンロード
                  </button>
                </div>
              </div>

              {/* Recommended Skills List */}
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                <h3 className="text-lg font-semibold text-gray-800 mb-4">推薦スキル詳細（Causalスコア順）</h3>

                <div className="space-y-3">
                  {careerPath.recommended_skills.map((skill, index) => (
                    <div key={index} className="border border-gray-200 rounded-lg overflow-hidden">
                      {/* Skill Header */}
                      <div className="p-4 bg-gradient-to-r from-white to-gray-50">
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <div className="flex items-center gap-2 mb-2">
                              <span className="text-xl font-bold text-gray-400">#{index + 1}</span>
                              <h4 className="font-bold text-gray-800 text-lg">{skill.competence_name}</h4>
                              <span className="text-xs bg-gray-200 text-gray-700 px-2 py-1 rounded">
                                {skill.category}
                              </span>
                            </div>
                            <div className="grid grid-cols-4 gap-2 text-sm">
                              <div>
                                <span className="text-gray-600">総合: </span>
                                <span className="font-bold text-[#00A968]">
                                  {(skill.total_score * 100).toFixed(0)}%
                                </span>
                              </div>
                              <div>
                                <span className="text-gray-600">準備度: </span>
                                <span className="font-bold text-blue-600">
                                  {(skill.readiness_score * 100).toFixed(0)}%
                                </span>
                              </div>
                              <div>
                                <span className="text-gray-600">確率: </span>
                                <span className="font-bold text-purple-600">
                                  {(skill.bayesian_score * 100).toFixed(0)}%
                                </span>
                              </div>
                              <div>
                                <span className="text-gray-600">将来性: </span>
                                <span className="font-bold text-green-600">
                                  {(skill.utility_score * 100).toFixed(0)}%
                                </span>
                              </div>
                            </div>
                          </div>
                          <div className="flex flex-col items-end gap-2">
                            <div className="text-sm">
                              <span className="text-gray-600">前提:</span>{' '}
                              <span className="font-bold">{skill.prerequisites.length}</span> |{' '}
                              <span className="text-gray-600">次へ:</span>{' '}
                              <span className="font-bold">{skill.enables.length}</span>
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* Expandable Details */}
                      <button
                        onClick={() => setExpandedSkillIndex(expandedSkillIndex === index ? null : index)}
                        className="w-full px-4 py-2 bg-gray-50 hover:bg-gray-100 transition-colors flex items-center justify-between text-sm font-medium text-gray-700"
                      >
                        <span>詳細な推薦理由を表示</span>
                        {expandedSkillIndex === index ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
                      </button>

                      {expandedSkillIndex === index && (
                        <div className="p-4 bg-gray-50 border-t border-gray-200">
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            {/* Readiness Reasons */}
                            <div className="bg-blue-50 rounded-lg p-3">
                              <h5 className="font-semibold text-blue-800 mb-2 text-sm">準備度の根拠</h5>
                              {skill.readiness_reasons && skill.readiness_reasons.length > 0 ? (
                                <ul className="text-xs text-blue-700 space-y-1">
                                  {skill.readiness_reasons.map(([name, effect], idx) => (
                                    <li key={idx}>
                                      • {name} (因果効果: {effect.toFixed(3)})
                                    </li>
                                  ))}
                                </ul>
                              ) : (
                                <p className="text-xs text-blue-700">基礎スキルとして推奨</p>
                              )}
                            </div>

                            {/* Utility Reasons */}
                            <div className="bg-green-50 rounded-lg p-3">
                              <h5 className="font-semibold text-green-800 mb-2 text-sm">将来性の根拠</h5>
                              {skill.utility_reasons && skill.utility_reasons.length > 0 ? (
                                <ul className="text-xs text-green-700 space-y-1">
                                  {skill.utility_reasons.map(([name, effect], idx) => (
                                    <li key={idx}>
                                      • {name}の習得に役立つ (効果: {effect.toFixed(3)})
                                    </li>
                                  ))}
                                </ul>
                              ) : (
                                <p className="text-xs text-green-700">汎用スキル</p>
                              )}
                            </div>
                          </div>

                          {/* Dependencies */}
                          {(skill.prerequisites.length > 0 || skill.enables.length > 0) && (
                            <div className="mt-3 p-3 bg-purple-50 rounded-lg">
                              <h5 className="font-semibold text-purple-800 mb-2 text-sm">スキル依存関係</h5>
                              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                {skill.prerequisites.length > 0 && (
                                  <div>
                                    <p className="text-xs font-medium text-purple-700 mb-1">
                                      前提スキル ({skill.prerequisites.length}個):
                                    </p>
                                    <ul className="text-xs text-purple-600 space-y-0.5">
                                      {skill.prerequisites.map((prereq, idx) => (
                                        <li key={idx}>← {prereq.skill_name}</li>
                                      ))}
                                    </ul>
                                  </div>
                                )}
                                {skill.enables.length > 0 && (
                                  <div>
                                    <p className="text-xs font-medium text-purple-700 mb-1">
                                      このスキルが役立つ ({skill.enables.length}個):
                                    </p>
                                    <ul className="text-xs text-purple-600 space-y-0.5">
                                      {skill.enables.map((enable, idx) => (
                                        <li key={idx}>→ {enable.skill_name}</li>
                                      ))}
                                    </ul>
                                  </div>
                                )}
                              </div>
                            </div>
                          )}

                          {/* Causal Graph Visualization */}
                          <div className="mt-4">
                            <div className="flex items-center justify-between mb-3">
                              <h5 className="font-semibold text-gray-800 text-sm flex items-center gap-2">
                                <Network size={16} />
                                因果グラフ
                              </h5>
                              <div className="flex gap-2">
                                <button
                                  onClick={() => setShowGraphSettings(!showGraphSettings)}
                                  className="flex items-center gap-1 px-2 py-1 text-xs text-gray-600 hover:text-gray-800 hover:bg-gray-200 rounded transition-colors"
                                >
                                  <Settings size={14} />
                                  {showGraphSettings ? '閉じる' : '設定'}
                                </button>
                                <button
                                  onClick={() => loadEgoGraph(skill.competence_name)}
                                  disabled={loadingGraph}
                                  className="px-3 py-1 text-xs bg-[#00A968] text-white rounded hover:bg-[#008f5a] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                                >
                                  {loadingGraph ? '生成中...' : 'グラフを表示'}
                                </button>
                              </div>
                            </div>

                            {/* Graph Settings Panel */}
                            {showGraphSettings && (
                              <div className="mb-3 p-3 bg-white rounded-lg border border-gray-200">
                                <h6 className="text-xs font-semibold text-gray-800 mb-2">グラフ描画パラメータ</h6>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                  <div>
                                    <label className="block text-xs text-gray-700 mb-1">
                                      表示範囲（Radius）: {graphRadius}
                                    </label>
                                    <input
                                      type="range"
                                      min="1"
                                      max="3"
                                      step="1"
                                      value={graphRadius}
                                      onChange={(e) => setGraphRadius(parseInt(e.target.value))}
                                      className="w-full"
                                    />
                                    <p className="text-xs text-gray-500 mt-0.5">
                                      中心ノードから何ホップ先まで表示するか
                                    </p>
                                  </div>
                                  <div>
                                    <label className="block text-xs text-gray-700 mb-1">
                                      因果効果の閾値: {graphThreshold.toFixed(3)}
                                    </label>
                                    <input
                                      type="range"
                                      min="0.01"
                                      max="0.2"
                                      step="0.01"
                                      value={graphThreshold}
                                      onChange={(e) => setGraphThreshold(parseFloat(e.target.value))}
                                      className="w-full"
                                    />
                                    <p className="text-xs text-gray-500 mt-0.5">
                                      この値以下の因果効果は非表示
                                    </p>
                                  </div>
                                </div>
                              </div>
                            )}

                            {/* Loading State */}
                            {loadingGraph && (
                              <div className="bg-white rounded-lg border border-gray-200 p-8 text-center">
                                <Loader2 size={24} className="animate-spin text-[#00A968] mx-auto" />
                                <p className="text-sm text-gray-600 mt-2">グラフを生成中...</p>
                              </div>
                            )}

                            {/* Graph Display */}
                            {graphHtml && !loadingGraph && (
                              <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
                                <div className="p-3 bg-blue-50 text-xs text-blue-800">
                                  <p className="mb-2">
                                    <strong>グラフの見方:</strong> ノード（丸）がスキルを表し、エッジ（矢印）が因果関係を表します。
                                  </p>
                                  <div className="grid grid-cols-3 gap-2">
                                    <div className="flex items-center gap-1">
                                      <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#90EE90' }}></div>
                                      <span>取得済み</span>
                                    </div>
                                    <div className="flex items-center gap-1">
                                      <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#97C2FC' }}></div>
                                      <span>中心スキル</span>
                                    </div>
                                    <div className="flex items-center gap-1">
                                      <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#DDDDDD' }}></div>
                                      <span>その他</span>
                                    </div>
                                  </div>
                                </div>
                                <div className="border-t border-gray-200" style={{ height: '500px' }}>
                                  <iframe
                                    srcDoc={graphHtml}
                                    className="w-full h-full"
                                    title="Causal Graph Visualization"
                                    sandbox="allow-scripts allow-same-origin"
                                  />
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              {/* Action Plan */}
              <div className="bg-gradient-to-r from-[#00A968] to-[#008F58] rounded-lg p-6 mt-6 text-white">
                <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                  <TrendingUp size={24} />
                  次のアクションステップ
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="bg-white/10 rounded-lg p-4">
                    <h4 className="font-semibold mb-2">🔹 今週のアクション</h4>
                    <p className="text-sm">上位3つのスキルを確認し、学習リソースを探す</p>
                  </div>
                  <div className="bg-white/10 rounded-lg p-4">
                    <h4 className="font-semibold mb-2">🔹 今月の目標</h4>
                    <p className="text-sm">前提スキルが少ないスキルを最低1つ習得</p>
                  </div>
                  <div className="bg-white/10 rounded-lg p-4">
                    <h4 className="font-semibold mb-2">🔹 3ヶ月後の目標</h4>
                    <p className="text-sm">推薦スキルの30%以上を習得完了</p>
                  </div>
                </div>
              </div>
            </>
          )
        }

        {/* No Results Message */}
        {
          careerPath && careerPath.recommended_skills && careerPath.recommended_skills.length === 0 && (
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6">
              <p className="text-yellow-800">
                推薦スキルが見つかりませんでした。フィルタリング設定を調整してみてください。
              </p>
            </div>
          )
        }
      </div >
    </div >
  );
};
