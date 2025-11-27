import { useState, useEffect } from 'react';
import axios from 'axios';
import { Brain, TrendingUp, AlertCircle, Loader2, Network, Info, ChevronDown, ChevronUp, Zap, Settings } from 'lucide-react';

interface Member {
  member_code: string;
  member_name: string;
  display_name: string;
}

interface Recommendation {
  skill_name: string;
  skill_code?: string;
  score: number;
  readiness_score: number;
  bayesian_score: number;
  utility_score: number;
  explanation: string;
}

interface RecommendationResponse {
  member_id: string;
  recommendations: Recommendation[];
  message?: string;
}

interface OptimizationResult {
  success: boolean;
  optimized_weights: {
    readiness: number;
    bayesian: number;
    utility: number;
  };
  message: string;
}

export const CausalRecommendation = () => {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [dataUploaded, setDataUploaded] = useState(false);
  const [modelId, setModelId] = useState('');

  // Member selection
  const [members, setMembers] = useState<Member[]>([]);
  const [selectedMember, setSelectedMember] = useState('');
  const [loadingMembers, setLoadingMembers] = useState(false);

  const [topN, setTopN] = useState(10);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [results, setResults] = useState<RecommendationResponse | null>(null);

  // Graph visualization state
  const [selectedSkill, setSelectedSkill] = useState<string | null>(null);
  const [graphHtml, setGraphHtml] = useState<string | null>(null);
  const [loadingGraph, setLoadingGraph] = useState(false);
  const [showGraph, setShowGraph] = useState(false);
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);

  // Weight optimization state
  const [showWeightOptimization, setShowWeightOptimization] = useState(false);
  const [optimizing, setOptimizing] = useState(false);
  const [nTrials, setNTrials] = useState(50);
  const [optimizedWeights, setOptimizedWeights] = useState<any>(null);

  // Manual weight adjustment state
  const [showManualWeights, setShowManualWeights] = useState(false);
  const [readinessWeight, setReadinessWeight] = useState(0.6);
  const [bayesianWeight, setBayesianWeight] = useState(0.3);
  const [utilityWeight, setUtilityWeight] = useState(0.1);
  const [currentWeights, setCurrentWeights] = useState<any>(null);
  const [loadingWeights, setLoadingWeights] = useState(false);
  const [updatingWeights, setUpdatingWeights] = useState(false);

  // Graph parameters
  const [graphRadius, setGraphRadius] = useState(1);
  const [graphThreshold, setGraphThreshold] = useState(0.05);

  useEffect(() => {
    const sid = sessionStorage.getItem('career_session_id');
    const uploaded = sessionStorage.getItem('career_data_uploaded');
    const savedModelId = sessionStorage.getItem('career_model_id');

    setSessionId(sid);
    setDataUploaded(uploaded === 'true');
    if (savedModelId) {
      setModelId(savedModelId);
    }

    // Load members list if session exists
    if (sid && uploaded === 'true') {
      loadMembers(sid);
    }
  }, []);

  const loadMembers = async (sid: string) => {
    setLoadingMembers(true);
    try {
      const response = await axios.get(`http://localhost:8000/api/session/${sid}/members`);
      setMembers(response.data.members);
    } catch (err: any) {
      console.error('Failed to load members:', err);
    } finally {
      setLoadingMembers(false);
    }
  };

  const handleOptimizeWeights = async () => {
    if (!modelId) {
      setError('モデルIDが設定されていません');
      return;
    }
    setOptimizing(true);
    setError('');
    try {
      const response = await axios.post<OptimizationResult>(
        'http://localhost:8000/api/weights/optimize',
        {
          model_id: modelId,
          n_trials: nTrials,
          n_jobs: -1,
          holdout_ratio: 0.2,
          top_k: 10
        }
      );
      setOptimizedWeights(response.data.optimized_weights);
      setCurrentWeights(response.data.optimized_weights);
      alert('重みの最適化が完了しました！最適化された重みが自動的に適用されます。');
    } catch (err: any) {
      setError(err.response?.data?.detail || '重みの最適化に失敗しました');
    } finally {
      setOptimizing(false);
    }
  };

  const loadCurrentWeights = async () => {
    if (!modelId) return;

    setLoadingWeights(true);
    try {
      const response = await axios.get(`http://localhost:8000/api/weights/${modelId}`);
      const weights = response.data.weights;
      setCurrentWeights(weights);
      setReadinessWeight(weights.readiness);
      setBayesianWeight(weights.bayesian);
      setUtilityWeight(weights.utility);
    } catch (err: any) {
      console.error('Failed to load weights:', err);
    } finally {
      setLoadingWeights(false);
    }
  };

  const handleUpdateWeights = async () => {
    if (!modelId) {
      setError('モデルIDが設定されていません');
      return;
    }

    setUpdatingWeights(true);
    setError('');

    try {
      const response = await axios.post('http://localhost:8000/api/weights/update', {
        model_id: modelId,
        weights: {
          readiness: readinessWeight,
          bayesian: bayesianWeight,
          utility: utilityWeight
        }
      });

      setCurrentWeights(response.data.weights);
      alert('重みを更新しました！推薦結果に反映されています。');
    } catch (err: any) {
      setError(err.response?.data?.detail || '重みの更新に失敗しました');
    } finally {
      setUpdatingWeights(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResults(null);
    setShowGraph(false);
    setGraphHtml(null);

    try {
      const response = await axios.post<RecommendationResponse>(
        'http://localhost:8000/api/recommend',
        {
          model_id: modelId,
          member_id: selectedMember,
          top_n: topN,
        }
      );
      setResults(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'エラーが発生しました');
    } finally {
      setLoading(false);
    }
  };

  const loadEgoGraph = async (skillName: string, skillCode?: string) => {
    if (!modelId) return;

    setLoadingGraph(true);
    setError('');
    setSelectedSkill(skillName);

    try {
      const response = await axios.post('http://localhost:8000/api/graph/ego', {
        model_id: modelId,
        center_node: skillCode || skillName,
        radius: graphRadius,
        threshold: graphThreshold,
        show_negative: false,
        member_skills: []
      });

      setGraphHtml(response.data.html);
      setShowGraph(true);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'グラフの生成に失敗しました');
    } finally {
      setLoadingGraph(false);
    }
  };

  const toggleExpanded = (index: number) => {
    setExpandedIndex(expandedIndex === index ? null : index);
  };

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
          <div className="flex items-center gap-3 mb-2">
            <Brain size={32} className="text-[#00A968]" />
            <h1 className="text-3xl font-bold text-gray-800">因果推論推薦</h1>
          </div>
          <p className="text-gray-600">
            データからスキル間の因果関係を発見し、説得力のある推薦を行います
          </p>
        </div>

        {/* Explanation */}
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-6 mb-6">
          <div className="flex items-start gap-3">
            <Info size={20} className="text-blue-600 flex-shrink-0 mt-0.5" />
            <div className="text-sm text-blue-800">
              <p className="font-semibold mb-2">因果推論推薦の仕組み</p>
              <ul className="space-y-1 list-disc list-inside">
                <li><strong>LiNGAM</strong>: スキル間の因果関係（原因→結果）を自動発見</li>
                <li><strong>Bayesian Network</strong>: 同様のスキルパターンを持つ人の習得確率を計算</li>
                <li><strong>3軸スコアリング</strong>: Readiness（準備度）、Bayesian（確率）、Utility（将来性）で評価</li>
                <li><strong>因果グラフ可視化</strong>: 各スキルの因果関係ネットワークを視覚的に理解</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Weight Optimization Section */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
          <button
            onClick={() => setShowWeightOptimization(!showWeightOptimization)}
            className="w-full flex items-center justify-between text-left"
            type="button"
          >
            <div className="flex items-center gap-2">
              <Settings size={20} className="text-[#00A968]" />
              <h2 className="text-lg font-semibold text-gray-800">重み最適化（オプション）</h2>
            </div>
            {showWeightOptimization ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
          </button>

          {showWeightOptimization && (
            <div className="mt-4 pt-4 border-t border-gray-200">
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
                <div className="flex items-start gap-2">
                  <Info size={16} className="text-blue-600 flex-shrink-0 mt-0.5" />
                  <p className="text-sm text-blue-800">
                    ベイズ最適化により、推薦精度を最大化する重み（Readiness・Bayesian・Utility）を自動で探索します。
                    計算には数分かかる場合があります。
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    試行回数 (n_trials)
                  </label>
                  <input
                    type="number"
                    value={nTrials}
                    onChange={(e) => setNTrials(parseInt(e.target.value))}
                    min="10"
                    max="200"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-[#00A968]"
                  />
                  <p className="text-xs text-gray-500 mt-1">多いほど精度が高いが時間がかかる（推奨: 50-100）</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    モデルID
                  </label>
                  <input
                    type="text"
                    value={modelId}
                    onChange={(e) => setModelId(e.target.value)}
                    placeholder="例: model_001"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-[#00A968]"
                  />
                  <p className="text-xs text-gray-500 mt-1">最適化対象のモデルID</p>
                </div>
              </div>

              <button
                onClick={handleOptimizeWeights}
                disabled={optimizing || !modelId}
                className="w-full bg-purple-600 text-white py-3 rounded-md font-medium hover:bg-purple-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {optimizing ? (
                  <>
                    <Loader2 size={20} className="animate-spin" />
                    最適化中...
                  </>
                ) : (
                  <>
                    <Zap size={20} />
                    重みを最適化
                  </>
                )}
              </button>

              {optimizedWeights && (
                <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
                  <h3 className="font-semibold text-green-800 mb-2">最適化された重み</h3>
                  <div className="grid grid-cols-3 gap-4 text-sm">
                    <div>
                      <p className="text-gray-600">Readiness</p>
                      <p className="font-bold text-green-700">{(optimizedWeights.readiness * 100).toFixed(1)}%</p>
                    </div>
                    <div>
                      <p className="text-gray-600">Bayesian</p>
                      <p className="font-bold text-green-700">{(optimizedWeights.bayesian * 100).toFixed(1)}%</p>
                    </div>
                    <div>
                      <p className="text-gray-600">Utility</p>
                      <p className="font-bold text-green-700">{(optimizedWeights.utility * 100).toFixed(1)}%</p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Manual Weight Adjustment Section */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
          <button
            onClick={() => {
              setShowManualWeights(!showManualWeights);
              if (!showManualWeights && modelId) {
                loadCurrentWeights();
              }
            }}
            className="w-full flex items-center justify-between text-left"
            type="button"
          >
            <div className="flex items-center gap-2">
              <Settings size={20} className="text-[#00A968]" />
              <h2 className="text-lg font-semibold text-gray-800">重み手動調整（オプション）</h2>
            </div>
            {showManualWeights ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
          </button>

          {showManualWeights && (
            <div className="mt-4 pt-4 border-t border-gray-200">
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
                <div className="flex items-start gap-2">
                  <Info size={16} className="text-blue-600 flex-shrink-0 mt-0.5" />
                  <p className="text-sm text-blue-800">
                    スライダーで重みを手動調整できます。推薦スコアは「総合スコア = Readiness × w₁ + Bayesian × w₂ + Utility × w₃」で計算されます。
                  </p>
                </div>
              </div>

              {loadingWeights ? (
                <div className="flex justify-center py-4">
                  <Loader2 size={24} className="animate-spin text-[#00A968]" />
                </div>
              ) : (
                <>
                  {/* Current Weights Display */}
                  {currentWeights && (
                    <div className="mb-4 p-3 bg-gray-50 rounded-lg">
                      <p className="text-sm font-medium text-gray-700 mb-2">現在の重み:</p>
                      <div className="grid grid-cols-3 gap-4 text-sm">
                        <div>
                          <span className="text-gray-600">Readiness: </span>
                          <span className="font-bold text-blue-600">{(currentWeights.readiness * 100).toFixed(1)}%</span>
                        </div>
                        <div>
                          <span className="text-gray-600">Bayesian: </span>
                          <span className="font-bold text-purple-600">{(currentWeights.bayesian * 100).toFixed(1)}%</span>
                        </div>
                        <div>
                          <span className="text-gray-600">Utility: </span>
                          <span className="font-bold text-green-600">{(currentWeights.utility * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Weight Sliders */}
                  <div className="space-y-4 mb-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Readiness（準備度）: {(readinessWeight * 100).toFixed(0)}%
                      </label>
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.05"
                        value={readinessWeight}
                        onChange={(e) => setReadinessWeight(parseFloat(e.target.value))}
                        className="w-full h-2 bg-blue-200 rounded-lg appearance-none cursor-pointer"
                      />
                      <p className="text-xs text-gray-500 mt-1">保有スキルから推奨スキルへの因果効果の重み</p>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Bayesian（確率）: {(bayesianWeight * 100).toFixed(0)}%
                      </label>
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.05"
                        value={bayesianWeight}
                        onChange={(e) => setBayesianWeight(parseFloat(e.target.value))}
                        className="w-full h-2 bg-purple-200 rounded-lg appearance-none cursor-pointer"
                      />
                      <p className="text-xs text-gray-500 mt-1">同様のスキルパターンを持つ人の習得確率の重み</p>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Utility（将来性）: {(utilityWeight * 100).toFixed(0)}%
                      </label>
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.05"
                        value={utilityWeight}
                        onChange={(e) => setUtilityWeight(parseFloat(e.target.value))}
                        className="w-full h-2 bg-green-200 rounded-lg appearance-none cursor-pointer"
                      />
                      <p className="text-xs text-gray-500 mt-1">推奨スキルから将来のスキルへの因果効果の重み</p>
                    </div>
                  </div>

                  {/* Weight Sum Warning */}
                  {(() => {
                    const total = readinessWeight + bayesianWeight + utilityWeight;
                    if (Math.abs(total - 1.0) > 0.01) {
                      return (
                        <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                          <p className="text-sm text-yellow-800">
                            ⚠️ 重みの合計: {total.toFixed(2)} （適用時に自動的に正規化されます）
                          </p>
                        </div>
                      );
                    } else {
                      return (
                        <div className="mb-4 p-3 bg-green-50 border border-green-200 rounded-lg">
                          <p className="text-sm text-green-800">
                            ✅ 重みの合計: {total.toFixed(2)}
                          </p>
                        </div>
                      );
                    }
                  })()}

                  {/* Update Button */}
                  <button
                    onClick={handleUpdateWeights}
                    disabled={updatingWeights || !modelId}
                    className="w-full bg-[#00A968] text-white py-3 rounded-md font-medium hover:bg-[#008F58] transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                  >
                    {updatingWeights ? (
                      <>
                        <Loader2 size={20} className="animate-spin" />
                        更新中...
                      </>
                    ) : (
                      <>
                        <Settings size={20} />
                        この重みを適用
                      </>
                    )}
                  </button>
                </>
              )}
            </div>
          )}
        </div>

        {/* Input Form */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
          <h2 className="text-lg font-semibold text-gray-800 mb-4">推薦パラメータ</h2>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  モデルID
                </label>
                <input
                  type="text"
                  value={modelId}
                  onChange={(e) => setModelId(e.target.value)}
                  placeholder="例: model_001"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-[#00A968]"
                  required
                />
                <p className="text-xs text-gray-500 mt-1">学習済みモデルのIDを入力</p>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  対象メンバー
                </label>
                <select
                  value={selectedMember}
                  onChange={(e) => setSelectedMember(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-[#00A968]"
                  required
                  disabled={loadingMembers}
                >
                  <option value="">メンバーを選択してください</option>
                  {members.map((member) => (
                    <option key={member.member_code} value={member.member_code}>
                      {member.display_name}
                    </option>
                  ))}
                </select>
                <p className="text-xs text-gray-500 mt-1">
                  {loadingMembers ? '読み込み中...' : 'アップロードしたCSVから選択'}
                </p>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  推薦数 (Top N)
                </label>
                <input
                  type="number"
                  value={topN}
                  onChange={(e) => setTopN(parseInt(e.target.value))}
                  min="1"
                  max="50"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-[#00A968]"
                />
                <p className="text-xs text-gray-500 mt-1">上位何件を表示するか</p>
              </div>
            </div>
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-[#00A968] text-white py-3 rounded-md font-medium hover:bg-[#008F58] transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <Loader2 size={20} className="animate-spin" />
                  推薦を生成中...
                </>
              ) : (
                <>
                  <TrendingUp size={20} />
                  推薦を取得
                </>
              )}
            </button>
          </form>
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

        {/* Results */}
        {results && (
          <div>
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
              <h2 className="text-lg font-semibold text-gray-800 mb-4">
                推薦結果 - {results.member_id}
              </h2>

              {results.message && (
                <p className="text-gray-600 mb-4">{results.message}</p>
              )}

              {results.recommendations && results.recommendations.length > 0 ? (
                <div className="space-y-4">
                  {results.recommendations.map((rec, index) => (
                    <div
                      key={index}
                      className="border border-gray-200 rounded-lg overflow-hidden hover:border-[#00A968] transition-colors"
                    >
                      {/* Recommendation Header */}
                      <div className="p-5 bg-gradient-to-r from-white to-gray-50">
                        <div className="flex items-start justify-between mb-3">
                          <div className="flex-1">
                            <div className="flex items-center gap-2 mb-2">
                              <span className="text-2xl font-bold text-gray-400">#{index + 1}</span>
                              <h3 className="font-bold text-gray-800 text-xl">
                                {rec.skill_name}
                              </h3>
                            </div>
                            {rec.explanation && (
                              <p className="text-sm text-gray-600">{rec.explanation}</p>
                            )}
                          </div>
                          <div className="flex flex-col items-end gap-2">
                            <div className="bg-[#00A968] text-white px-4 py-2 rounded-full text-lg font-bold">
                              {(rec.score * 100).toFixed(1)}%
                            </div>
                            <button
                              onClick={() => loadEgoGraph(rec.skill_name, rec.skill_code)}
                              disabled={loadingGraph}
                              className="text-sm text-[#00A968] hover:text-[#008F58] font-medium flex items-center gap-1 disabled:opacity-50"
                            >
                              <Network size={16} />
                              因果グラフを表示
                            </button>
                          </div>
                        </div>

                        {/* 3-Axis Scores */}
                        <div className="grid grid-cols-3 gap-4">
                          <div>
                            <div className="flex justify-between items-center mb-2">
                              <p className="text-xs font-medium text-gray-600">Readiness（準備度）</p>
                              <p className="text-sm font-bold text-blue-600">
                                {(rec.readiness_score * 100).toFixed(0)}%
                              </p>
                            </div>
                            <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
                              <div
                                className="h-full bg-blue-500 transition-all"
                                style={{ width: `${rec.readiness_score * 100}%` }}
                              />
                            </div>
                            <p className="text-xs text-gray-500 mt-1">保有スキルからの習得しやすさ</p>
                          </div>
                          <div>
                            <div className="flex justify-between items-center mb-2">
                              <p className="text-xs font-medium text-gray-600">Bayesian（確率）</p>
                              <p className="text-sm font-bold text-purple-600">
                                {(rec.bayesian_score * 100).toFixed(0)}%
                              </p>
                            </div>
                            <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
                              <div
                                className="h-full bg-purple-500 transition-all"
                                style={{ width: `${rec.bayesian_score * 100}%` }}
                              />
                            </div>
                            <p className="text-xs text-gray-500 mt-1">同様パターンでの習得確率</p>
                          </div>
                          <div>
                            <div className="flex justify-between items-center mb-2">
                              <p className="text-xs font-medium text-gray-600">Utility（将来性）</p>
                              <p className="text-sm font-bold text-green-600">
                                {(rec.utility_score * 100).toFixed(0)}%
                              </p>
                            </div>
                            <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
                              <div
                                className="h-full bg-green-500 transition-all"
                                style={{ width: `${rec.utility_score * 100}%` }}
                              />
                            </div>
                            <p className="text-xs text-gray-500 mt-1">習得後の将来的な価値</p>
                          </div>
                        </div>
                      </div>

                      {/* Expandable Details */}
                      <button
                        onClick={() => toggleExpanded(index)}
                        className="w-full px-5 py-3 bg-gray-50 hover:bg-gray-100 transition-colors flex items-center justify-between text-sm font-medium text-gray-700"
                      >
                        <span>詳細な推薦理由を表示</span>
                        {expandedIndex === index ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
                      </button>

                      {expandedIndex === index && (
                        <div className="p-5 bg-gray-50 border-t border-gray-200">
                          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <div className="bg-blue-50 rounded-lg p-4">
                              <h4 className="font-semibold text-blue-800 mb-2 text-sm">準備度の根拠</h4>
                              <p className="text-xs text-blue-700">
                                あなたが既に保有しているスキルから、このスキルへの因果的なつながりが強く、
                                習得に必要な基礎が整っています。
                              </p>
                            </div>
                            <div className="bg-purple-50 rounded-lg p-4">
                              <h4 className="font-semibold text-purple-800 mb-2 text-sm">確率の根拠</h4>
                              <p className="text-xs text-purple-700">
                                同様のスキルセットを持つメンバーがこのスキルを高確率で習得しているため、
                                あなたも習得できる可能性が高いです。
                              </p>
                            </div>
                            <div className="bg-green-50 rounded-lg p-4">
                              <h4 className="font-semibold text-green-800 mb-2 text-sm">将来性の根拠</h4>
                              <p className="text-xs text-green-700">
                                このスキルを習得すると、さらに多くの高度なスキルへの道が開かれ、
                                キャリアの選択肢が大きく広がります。
                              </p>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-gray-500 text-center py-8">推薦結果がありません</p>
              )}
            </div>

            {/* Graph Visualization */}
            {showGraph && graphHtml && (
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold text-gray-800 flex items-center gap-2">
                    <Network size={20} className="text-[#00A968]" />
                    因果グラフ - {selectedSkill}
                  </h2>
                  <button
                    onClick={() => setShowGraph(false)}
                    className="text-sm text-gray-600 hover:text-gray-800"
                  >
                    閉じる
                  </button>
                </div>

                <div className="mb-4 p-4 bg-blue-50 rounded-lg">
                  <p className="text-sm text-blue-800">
                    <strong>グラフの見方:</strong>
                    ノード（丸）がスキルを表し、エッジ（矢印）が因果関係を表します。
                    矢印の太さは因果効果の強さを示します。
                  </p>
                </div>

                <div className="border border-gray-200 rounded-lg overflow-hidden" style={{ height: '600px' }}>
                  <iframe
                    srcDoc={graphHtml}
                    className="w-full h-full"
                    title="Causal Graph Visualization"
                    sandbox="allow-scripts allow-same-origin"
                  />
                </div>
              </div>
            )}

            {loadingGraph && (
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                <div className="flex items-center justify-center py-12">
                  <Loader2 size={32} className="animate-spin text-[#00A968]" />
                  <span className="ml-3 text-gray-600">グラフを生成中...</span>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
