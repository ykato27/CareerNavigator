import { useState, useEffect } from 'react';
import axios from 'axios';
import {
  LayoutGrid, TrendingDown, Users, Briefcase,
  AlertCircle, Info, Loader2, BarChart3,
  Target, Award, Clock, Download, Filter, X
} from 'lucide-react';
import { API_BASE_URL, API_ENDPOINTS } from '../config/constants';

interface OrganizationalMetrics {
  total_members: number;
  total_skills: number;
  avg_skills_per_member: number;
  coverage_rate: number;
  diversity_index: number;
  high_concentration_skills: number;
  low_concentration_skills: number;
}

interface TopSkill {
  skill_code: string;
  skill_name: string;
  member_count: number;
}

interface SkillGap {
  skill_code: string;
  skill_name: string;
  current_rate: number;
  target_rate: number;
  gap_rate: number;
  gap_percentage: number;
}

interface SuccessionCandidate {
  member_code: string;
  member_name: string;
  current_position: string;
  current_grade: string;
  readiness_score: number;
  skill_match_rate: number;
  owned_skills_count: number;
  missing_skills_count: number;
  estimated_timeline: string;
}

export const OrganizationalSkillMap = () => {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [dataUploaded, setDataUploaded] = useState(false);
  const [activeTab, setActiveTab] = useState<'dashboard' | 'gap' | 'matrix' | 'succession' | 'simulation'>('dashboard');

  // Dashboard data
  const [metrics, setMetrics] = useState<OrganizationalMetrics | null>(null);
  const [topSkills, setTopSkills] = useState<TopSkill[]>([]);
  const [loadingMetrics, setLoadingMetrics] = useState(false);

  // Gap analysis data
  const [percentile, setPercentile] = useState(20);
  const [gapAnalysis, setGapAnalysis] = useState<SkillGap[]>([]);
  const [criticalSkills, setCriticalSkills] = useState<SkillGap[]>([]);
  const [loadingGap, setLoadingGap] = useState(false);
  const [gapCalculated, setGapCalculated] = useState(false);

  // Succession planning data
  const [targetPosition, setTargetPosition] = useState('');
  const [successionCandidates, setSuccessionCandidates] = useState<SuccessionCandidate[]>([]);
  const [loadingSuccession, setLoadingSuccession] = useState(false);

  // Filter state
  const [skillFilter, setSkillFilter] = useState('');
  const [showFilter, setShowFilter] = useState(false);

  const [error, setError] = useState('');

  useEffect(() => {
    const sid = sessionStorage.getItem('career_session_id');
    const uploaded = sessionStorage.getItem('career_data_uploaded');
    setSessionId(sid);
    setDataUploaded(uploaded === 'true');
  }, []);

  const fetchMetrics = async () => {
    if (!sessionId) return;

    setLoadingMetrics(true);
    setError('');

    try {
      const response = await axios.post(`${API_BASE_URL}${API_ENDPOINTS.ORG_METRICS}`, {
        session_id: sessionId
      });

      setMetrics(response.data.metrics);
      setTopSkills(response.data.top_skills);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'メトリクスの取得に失敗しました');
    } finally {
      setLoadingMetrics(false);
    }
  };

  const analyzeGap = async () => {
    if (!sessionId) return;

    setLoadingGap(true);
    setError('');

    try {
      const response = await axios.post(`${API_BASE_URL}${API_ENDPOINTS.ORG_SKILL_GAP}`, {
        session_id: sessionId,
        percentile: percentile / 100
      });

      setGapAnalysis(response.data.gap_analysis);
      setCriticalSkills(response.data.critical_skills);
      setGapCalculated(true);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'ギャップ分析に失敗しました');
    } finally {
      setLoadingGap(false);
    }
  };

  const searchSuccessionCandidates = async () => {
    if (!sessionId || !targetPosition) return;

    setLoadingSuccession(true);
    setError('');

    try {
      const response = await axios.post(`${API_BASE_URL}${API_ENDPOINTS.ORG_SUCCESSION}`, {
        session_id: sessionId,
        target_position: targetPosition,
        max_candidates: 20
      });

      setSuccessionCandidates(response.data.candidates);
    } catch (err: any) {
      setError(err.response?.data?.detail || '後継者候補の検索に失敗しました');
    } finally {
      setLoadingSuccession(false);
    }
  };

  const getReadinessColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600 bg-green-50';
    if (score >= 0.6) return 'text-blue-600 bg-blue-50';
    if (score >= 0.4) return 'text-yellow-600 bg-yellow-50';
    return 'text-red-600 bg-red-50';
  };

  const getReadinessLabel = (score: number) => {
    if (score >= 0.8) return '即戦力';
    if (score >= 0.6) return '有望';
    if (score >= 0.4) return '要育成';
    return '要大幅育成';
  };

  // CSV Export function
  const exportToCSV = (data: any[], filename: string) => {
    if (data.length === 0) return;

    const keys = Object.keys(data[0]);
    const csvContent = [
      keys.join(','),
      ...data.map(row => keys.map(key => {
        const value = row[key];
        // エスケープ処理（カンマや改行を含む場合はクォートで囲む）
        if (typeof value === 'string' && (value.includes(',') || value.includes('\n'))) {
          return `"${value.replace(/"/g, '""')}"`;
        }
        return value;
      }).join(','))
    ].join('\n');

    const blob = new Blob([`\uFEFF${csvContent}`], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', `${filename}_${new Date().toISOString().split('T')[0]}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const exportMetrics = () => {
    if (!metrics) return;

    const data = [
      {
        '項目': '総メンバー数',
        '値': metrics.total_members,
        '単位': '人'
      },
      {
        '項目': '総スキル数',
        '値': metrics.total_skills,
        '単位': 'スキル'
      },
      {
        '項目': '1人あたり平均スキル数',
        '値': metrics.avg_skills_per_member,
        '単位': 'スキル'
      },
      {
        '項目': 'スキルカバレッジ率',
        '値': `${(metrics.coverage_rate * 100).toFixed(1)}%`,
        '単位': ''
      },
      {
        '項目': 'スキル多様性指標',
        '値': metrics.diversity_index,
        '単位': ''
      }
    ];
    exportToCSV(data, '組織メトリクス');
  };

  const exportTopSkills = () => {
    if (topSkills.length === 0) return;

    const data = topSkills.map((skill, idx) => ({
      '順位': idx + 1,
      'スキルコード': skill.skill_code,
      'スキル名': skill.skill_name,
      '保有人数': skill.member_count
    }));
    exportToCSV(data, 'トップスキル');
  };

  const exportGapAnalysis = () => {
    if (gapAnalysis.length === 0) return;

    const data = gapAnalysis.map((gap, idx) => ({
      '順位': idx + 1,
      'スキルコード': gap.skill_code,
      'スキル名': gap.skill_name,
      '現在保有率': `${(gap.current_rate * 100).toFixed(1)}%`,
      '目標保有率': `${(gap.target_rate * 100).toFixed(1)}%`,
      'ギャップ': `${(gap.gap_rate * 100).toFixed(1)}%`,
      'ギャップ率': `${(gap.gap_percentage * 100).toFixed(1)}%`
    }));
    exportToCSV(data, 'スキルギャップ分析');
  };

  const exportSuccessionCandidates = () => {
    if (successionCandidates.length === 0) return;

    const data = successionCandidates.map((candidate, idx) => ({
      '順位': idx + 1,
      'メンバーコード': candidate.member_code,
      'メンバー名': candidate.member_name,
      '現在役職': candidate.current_position,
      '現在等級': candidate.current_grade,
      '準備度スコア': `${(candidate.readiness_score * 100).toFixed(1)}%`,
      'スキルマッチ度': `${(candidate.skill_match_rate * 100).toFixed(1)}%`,
      '保有スキル数': candidate.owned_skills_count,
      '不足スキル数': candidate.missing_skills_count,
      '推定育成期間': candidate.estimated_timeline
    }));
    exportToCSV(data, `後継者候補_${targetPosition}`);
  };

  // Filter functions
  const filteredTopSkills = topSkills.filter(skill =>
    skill.skill_name.toLowerCase().includes(skillFilter.toLowerCase())
  );

  const filteredGapAnalysis = gapAnalysis.filter(gap =>
    gap.skill_name.toLowerCase().includes(skillFilter.toLowerCase())
  );

  if (!dataUploaded) {
    return (
      <div className="flex-1 px-8 py-12">
        <div className="max-w-4xl mx-auto">
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 flex items-start gap-3">
            <AlertCircle size={24} className="text-yellow-600 flex-shrink-0 mt-0.5" />
            <div>
              <h3 className="font-semibold text-yellow-800 mb-2">データが読み込まれていません</h3>
              <p className="text-yellow-700 mb-3">
                まず「データ管理」ページで6種類のCSVファイルをアップロードしてください。
              </p>
              <button
                onClick={() => window.location.href = '/data-upload'}
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

  return (
    <div className="flex-1 px-8 py-12">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div>
              <div className="flex items-center gap-3 mb-2">
                <LayoutGrid size={32} className="text-[#00A968]" />
                <h1 className="text-2xl sm:text-3xl font-bold text-gray-800">組織スキルマップ</h1>
              </div>
              <p className="text-sm sm:text-base text-gray-600">
                組織全体のスキル保有状況を可視化し、スキルギャップを分析し、戦略的な人材配置を支援します
              </p>
            </div>
            {/* Filter Toggle Button */}
            {(metrics || gapAnalysis.length > 0) && (
              <button
                onClick={() => setShowFilter(!showFilter)}
                className="flex items-center gap-2 px-4 py-2 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors self-start sm:self-auto"
              >
                <Filter size={18} />
                <span className="text-sm font-medium">フィルター</span>
              </button>
            )}
          </div>

          {/* Filter Panel */}
          {showFilter && (metrics || gapAnalysis.length > 0) && (
            <div className="mt-4 p-4 bg-white border border-gray-200 rounded-lg">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold text-gray-800">スキルフィルター</h3>
                <button
                  onClick={() => {
                    setSkillFilter('');
                    setShowFilter(false);
                  }}
                  className="text-gray-500 hover:text-gray-700"
                >
                  <X size={18} />
                </button>
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-2">
                  スキル名で検索
                </label>
                <input
                  type="text"
                  value={skillFilter}
                  onChange={(e) => setSkillFilter(e.target.value)}
                  placeholder="例: CAD, 製図, Python"
                  className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-[#00A968]"
                />
                {skillFilter && (
                  <p className="text-xs text-gray-500 mt-1">
                    「{skillFilter}」で絞り込み中
                  </p>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Tabs */}
        <div className="flex flex-wrap gap-2 mb-6 border-b border-gray-200 overflow-x-auto">
          <button
            onClick={() => setActiveTab('dashboard')}
            className={`px-4 py-2 font-medium transition-colors border-b-2 ${activeTab === 'dashboard'
                ? 'text-[#00A968] border-[#00A968]'
                : 'text-gray-600 border-transparent hover:text-gray-800'
              }`}
          >
            <div className="flex items-center gap-2">
              <BarChart3 size={18} />
              <span>ダッシュボード</span>
            </div>
          </button>
          <button
            onClick={() => setActiveTab('gap')}
            className={`px-4 py-2 font-medium transition-colors border-b-2 ${activeTab === 'gap'
                ? 'text-[#00A968] border-[#00A968]'
                : 'text-gray-600 border-transparent hover:text-gray-800'
              }`}
          >
            <div className="flex items-center gap-2">
              <TrendingDown size={18} />
              <span>スキルギャップ分析</span>
            </div>
          </button>
          <button
            onClick={() => setActiveTab('succession')}
            className={`px-4 py-2 font-medium transition-colors border-b-2 ${activeTab === 'succession'
                ? 'text-[#00A968] border-[#00A968]'
                : 'text-gray-600 border-transparent hover:text-gray-800'
              }`}
          >
            <div className="flex items-center gap-2">
              <Briefcase size={18} />
              <span>後継者計画</span>
            </div>
          </button>
        </div>

        {/* Error Message */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6 flex items-start gap-3">
            <AlertCircle size={20} className="text-red-600 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-red-800 font-medium">エラー</p>
              <p className="text-red-700 text-sm">{error}</p>
            </div>
          </div>
        )}

        {/* Tab Content */}
        {activeTab === 'dashboard' && (
          <div>
            {!metrics && !loadingMetrics && (
              <button
                onClick={fetchMetrics}
                className="bg-[#00A968] text-white px-6 py-3 rounded-lg font-semibold hover:bg-[#008F58] transition-colors flex items-center gap-2"
              >
                <BarChart3 size={20} />
                組織メトリクスを読み込む
              </button>
            )}

            {loadingMetrics && (
              <div className="flex items-center justify-center py-12">
                <Loader2 size={32} className="animate-spin text-[#00A968]" />
                <span className="ml-3 text-gray-600">メトリクスを読み込み中...</span>
              </div>
            )}

            {metrics && (
              <div>
                {/* KPI Metrics */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                  <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                    <p className="text-sm text-gray-500 mb-1">総メンバー数</p>
                    <p className="text-3xl font-bold text-gray-800">{metrics.total_members.toLocaleString()}</p>
                    <p className="text-xs text-gray-500 mt-1">人</p>
                  </div>
                  <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                    <p className="text-sm text-gray-500 mb-1">1人あたり平均スキル数</p>
                    <p className="text-3xl font-bold text-gray-800">{metrics.avg_skills_per_member}</p>
                    <p className="text-xs text-gray-500 mt-1">スキル</p>
                  </div>
                  <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                    <p className="text-sm text-gray-500 mb-1">スキルカバレッジ率</p>
                    <p className="text-3xl font-bold text-gray-800">{(metrics.coverage_rate * 100).toFixed(1)}%</p>
                    <p className="text-xs text-gray-500 mt-1">保有率</p>
                  </div>
                  <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                    <p className="text-sm text-gray-500 mb-1">スキル多様性指標</p>
                    <p className="text-3xl font-bold text-gray-800">{metrics.diversity_index}</p>
                    <p className="text-xs text-gray-500 mt-1">指標値</p>
                  </div>
                </div>

                {/* Top Skills */}
                {topSkills.length > 0 && (
                  <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-4">
                      <h2 className="text-lg font-semibold text-gray-800 flex items-center gap-2">
                        <Award size={20} className="text-[#00A968]" />
                        保有人数が多いスキル Top 10
                        {skillFilter && (
                          <span className="text-sm font-normal text-gray-500">({filteredTopSkills.length}件)</span>
                        )}
                      </h2>
                      <button
                        onClick={exportTopSkills}
                        className="flex items-center gap-2 px-3 py-2 bg-[#00A968] text-white rounded-md hover:bg-[#008F58] transition-colors text-sm self-start sm:self-auto"
                      >
                        <Download size={16} />
                        CSVエクスポート
                      </button>
                    </div>
                    <div className="space-y-3">
                      {(skillFilter ? filteredTopSkills : topSkills).map((skill, idx) => (
                        <div key={skill.skill_code} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                          <div className="flex items-center gap-3">
                            <span className="text-sm font-bold text-gray-500 w-6">#{idx + 1}</span>
                            <div>
                              <p className="font-medium text-gray-800">{skill.skill_name}</p>
                              <p className="text-xs text-gray-500">{skill.skill_code}</p>
                            </div>
                          </div>
                          <div className="text-right">
                            <p className="text-xl font-bold text-[#00A968]">{skill.member_count}</p>
                            <p className="text-xs text-gray-500">人</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {activeTab === 'gap' && (
          <div>
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-6 mb-6">
              <div className="flex items-start gap-3">
                <Info size={20} className="text-blue-600 flex-shrink-0 mt-0.5" />
                <div className="text-sm text-blue-800">
                  <p className="font-semibold mb-2">スキルギャップ分析について</p>
                  <p>
                    組織として目指すべきスキル水準と現状のギャップを分析します。
                    上位N%のメンバーの平均スキルを目標として設定します。
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
              <h2 className="text-lg font-semibold text-gray-800 mb-4">ターゲット設定</h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    上位何%のメンバーを目標とするか
                  </label>
                  <div className="flex items-center gap-4">
                    <input
                      type="range"
                      value={percentile}
                      onChange={(e) => setPercentile(parseInt(e.target.value))}
                      min="5"
                      max="50"
                      step="5"
                      className="flex-1"
                    />
                    <span className="text-lg font-bold text-gray-800 w-16">{percentile}%</span>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    スキル保有数が多い上位{percentile}%のメンバーの平均を目標として設定します
                  </p>
                </div>

                <button
                  onClick={analyzeGap}
                  disabled={loadingGap}
                  className="bg-[#00A968] text-white px-6 py-3 rounded-lg font-semibold hover:bg-[#008F58] transition-colors flex items-center gap-2 disabled:bg-gray-400"
                >
                  {loadingGap ? (
                    <>
                      <Loader2 size={20} className="animate-spin" />
                      ギャップを計算中...
                    </>
                  ) : (
                    <>
                      <Target size={20} />
                      ギャップを計算
                    </>
                  )}
                </button>
              </div>
            </div>

            {gapCalculated && gapAnalysis.length > 0 && (
              <div>
                <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
                  <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-4">
                    <h2 className="text-lg font-semibold text-gray-800 flex items-center gap-2">
                      <TrendingDown size={20} className="text-[#00A968]" />
                      ギャップが大きいスキル Top 20
                      {skillFilter && (
                        <span className="text-sm font-normal text-gray-500">({filteredGapAnalysis.length}件)</span>
                      )}
                    </h2>
                    <button
                      onClick={exportGapAnalysis}
                      className="flex items-center gap-2 px-3 py-2 bg-[#00A968] text-white rounded-md hover:bg-[#008F58] transition-colors text-sm self-start sm:self-auto"
                    >
                      <Download size={16} />
                      CSVエクスポート
                    </button>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead>
                        <tr className="border-b border-gray-200">
                          <th className="text-left py-3 px-4 text-sm font-semibold text-gray-700">順位</th>
                          <th className="text-left py-3 px-4 text-sm font-semibold text-gray-700">力量名</th>
                          <th className="text-right py-3 px-4 text-sm font-semibold text-gray-700">現在保有率</th>
                          <th className="text-right py-3 px-4 text-sm font-semibold text-gray-700">目標保有率</th>
                          <th className="text-right py-3 px-4 text-sm font-semibold text-gray-700">ギャップ</th>
                          <th className="text-right py-3 px-4 text-sm font-semibold text-gray-700">ギャップ率</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(skillFilter ? filteredGapAnalysis : gapAnalysis).map((gap, idx) => (
                          <tr key={gap.skill_code} className="border-b border-gray-100 hover:bg-gray-50">
                            <td className="py-3 px-4 text-sm font-bold text-gray-500">#{idx + 1}</td>
                            <td className="py-3 px-4">
                              <p className="font-medium text-gray-800">{gap.skill_name}</p>
                              <p className="text-xs text-gray-500">{gap.skill_code}</p>
                            </td>
                            <td className="py-3 px-4 text-right text-sm text-gray-700">
                              {(gap.current_rate * 100).toFixed(1)}%
                            </td>
                            <td className="py-3 px-4 text-right text-sm text-gray-700">
                              {(gap.target_rate * 100).toFixed(1)}%
                            </td>
                            <td className="py-3 px-4 text-right text-sm font-bold text-red-600">
                              {(gap.gap_rate * 100).toFixed(1)}%
                            </td>
                            <td className="py-3 px-4 text-right">
                              <span className={`inline-block px-2 py-1 rounded text-xs font-bold ${gap.gap_percentage >= 0.5 ? 'bg-red-100 text-red-700' :
                                  gap.gap_percentage >= 0.3 ? 'bg-yellow-100 text-yellow-700' :
                                    'bg-blue-100 text-blue-700'
                                }`}>
                                {(gap.gap_percentage * 100).toFixed(1)}%
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {criticalSkills.length > 0 && (
                  <div className="bg-red-50 border border-red-200 rounded-lg p-6">
                    <h2 className="text-lg font-semibold text-red-800 mb-4 flex items-center gap-2">
                      <AlertCircle size={20} />
                      クリティカルスキル（ギャップ率30%以上）
                    </h2>
                    <p className="text-sm text-red-700 mb-4">
                      {criticalSkills.length}件のクリティカルスキルが見つかりました
                    </p>
                    <div className="space-y-4">
                      {criticalSkills.slice(0, 5).map((skill, idx) => (
                        <div key={skill.skill_code} className="bg-white rounded-lg p-4">
                          <h3 className="font-bold text-gray-800 mb-3">{idx + 1}. {skill.skill_name}</h3>
                          <div className="grid grid-cols-3 gap-4">
                            <div>
                              <p className="text-xs text-gray-500 mb-1">現在保有率</p>
                              <p className="text-lg font-bold text-gray-800">{(skill.current_rate * 100).toFixed(1)}%</p>
                            </div>
                            <div>
                              <p className="text-xs text-gray-500 mb-1">目標保有率</p>
                              <p className="text-lg font-bold text-gray-800">{(skill.target_rate * 100).toFixed(1)}%</p>
                            </div>
                            <div>
                              <p className="text-xs text-gray-500 mb-1">ギャップ</p>
                              <p className="text-lg font-bold text-red-600">{(skill.gap_rate * 100).toFixed(1)}%</p>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {activeTab === 'succession' && (
          <div>
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-6 mb-6">
              <div className="flex items-start gap-3">
                <Info size={20} className="text-blue-600 flex-shrink-0 mt-0.5" />
                <div className="text-sm text-blue-800">
                  <p className="font-semibold mb-2">後継者計画（サクセッションプラン）について</p>
                  <p>
                    重要ポジション（役職）の後継者候補を特定し、準備度を評価します。
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
              <h2 className="text-lg font-semibold text-gray-800 mb-4">対象役職設定</h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    後継者を探す役職
                  </label>
                  <input
                    type="text"
                    value={targetPosition}
                    onChange={(e) => setTargetPosition(e.target.value)}
                    placeholder="例: 部長、課長、チームリーダー"
                    className="w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-[#00A968]"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    検索したい役職名を入力してください
                  </p>
                </div>

                <button
                  onClick={searchSuccessionCandidates}
                  disabled={loadingSuccession || !targetPosition}
                  className="bg-[#00A968] text-white px-6 py-3 rounded-lg font-semibold hover:bg-[#008F58] transition-colors flex items-center gap-2 disabled:bg-gray-400"
                >
                  {loadingSuccession ? (
                    <>
                      <Loader2 size={20} className="animate-spin" />
                      後継者候補を検索中...
                    </>
                  ) : (
                    <>
                      <Users size={20} />
                      後継者候補を検索
                    </>
                  )}
                </button>
              </div>
            </div>

            {successionCandidates.length > 0 && (
              <div>
                <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
                  <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-4">
                    <h2 className="text-lg font-semibold text-gray-800 flex items-center gap-2">
                      <Award size={20} className="text-[#00A968]" />
                      {targetPosition} の後継者候補ランキング
                    </h2>
                    <button
                      onClick={exportSuccessionCandidates}
                      className="flex items-center gap-2 px-3 py-2 bg-[#00A968] text-white rounded-md hover:bg-[#008F58] transition-colors text-sm self-start sm:self-auto"
                    >
                      <Download size={16} />
                      CSVエクスポート
                    </button>
                  </div>
                  <p className="text-sm text-gray-600 mb-6">
                    {successionCandidates.length}人の候補者が見つかりました
                  </p>

                  <div className="space-y-4">
                    {successionCandidates.map((candidate, idx) => (
                      <div key={candidate.member_code} className="border border-gray-200 rounded-lg p-5 hover:shadow-md transition-shadow">
                        <div className="flex items-start justify-between mb-4">
                          <div className="flex items-start gap-3">
                            <span className="text-2xl font-bold text-gray-400">#{idx + 1}</span>
                            <div>
                              <h3 className="text-lg font-bold text-gray-800">{candidate.member_name}</h3>
                              <div className="flex items-center gap-3 mt-1">
                                <span className="text-sm text-gray-600">{candidate.current_position}</span>
                                <span className="text-xs text-gray-500">•</span>
                                <span className="text-sm text-gray-600">{candidate.current_grade}</span>
                              </div>
                            </div>
                          </div>
                          <div className={`px-3 py-1 rounded-full text-sm font-bold ${getReadinessColor(candidate.readiness_score)}`}>
                            {getReadinessLabel(candidate.readiness_score)}
                          </div>
                        </div>

                        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                          <div>
                            <p className="text-xs text-gray-500 mb-1">準備度スコア</p>
                            <p className="text-xl font-bold text-gray-800">{(candidate.readiness_score * 100).toFixed(1)}%</p>
                          </div>
                          <div>
                            <p className="text-xs text-gray-500 mb-1">スキルマッチ度</p>
                            <p className="text-xl font-bold text-gray-800">{(candidate.skill_match_rate * 100).toFixed(1)}%</p>
                          </div>
                          <div>
                            <p className="text-xs text-gray-500 mb-1">保有スキル</p>
                            <p className="text-xl font-bold text-green-600">{candidate.owned_skills_count}個</p>
                          </div>
                          <div>
                            <p className="text-xs text-gray-500 mb-1">不足スキル</p>
                            <p className="text-xl font-bold text-red-600">{candidate.missing_skills_count}個</p>
                          </div>
                          <div>
                            <p className="text-xs text-gray-500 mb-1 flex items-center gap-1">
                              <Clock size={12} />
                              推定育成期間
                            </p>
                            <p className="text-lg font-bold text-blue-600">{candidate.estimated_timeline}</p>
                          </div>
                        </div>

                        {/* Progress Bar */}
                        <div className="mt-4">
                          <div className="flex justify-between text-xs text-gray-600 mb-1">
                            <span>準備度</span>
                            <span>{(candidate.readiness_score * 100).toFixed(0)}%</span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div
                              className="bg-[#00A968] h-2 rounded-full transition-all"
                              style={{ width: `${candidate.readiness_score * 100}%` }}
                            ></div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
