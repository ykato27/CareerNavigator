import { useState, useEffect } from 'react';
import axios from 'axios';
import { Brain, TrendingUp, AlertCircle, Loader2, Network, Info, ChevronDown, ChevronUp, Eye } from 'lucide-react';

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

export const CausalRecommendation = () => {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [dataUploaded, setDataUploaded] = useState(false);
  const [modelId, setModelId] = useState('');
  const [memberId, setMemberId] = useState('');
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
  }, []);

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
          member_id: memberId,
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
                  メンバーID
                </label>
                <input
                  type="text"
                  value={memberId}
                  onChange={(e) => setMemberId(e.target.value)}
                  placeholder="例: M001"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-[#00A968]"
                  required
                />
                <p className="text-xs text-gray-500 mt-1">推薦対象のメンバーコード</p>
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
