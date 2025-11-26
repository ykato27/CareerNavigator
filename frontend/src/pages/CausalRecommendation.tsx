import { useState } from 'react';
import axios from 'axios';
import { Brain, TrendingUp, AlertCircle, Loader2 } from 'lucide-react';

interface Recommendation {
  skill_name: string;
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
  const [modelId, setModelId] = useState('');
  const [memberId, setMemberId] = useState('');
  const [topN, setTopN] = useState(10);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [results, setResults] = useState<RecommendationResponse | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResults(null);

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

  return (
    <div className="flex-1 px-8 py-12">
      <div className="max-w-6xl mx-auto">
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
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h2 className="text-lg font-semibold text-gray-800 mb-4">
              推薦結果 - {results.member_id}
            </h2>

            {results.message && (
              <p className="text-gray-600 mb-4">{results.message}</p>
            )}

            {results.recommendations && results.recommendations.length > 0 ? (
              <div className="space-y-3">
                {results.recommendations.map((rec, index) => (
                  <div
                    key={index}
                    className="border border-gray-200 rounded-lg p-4 hover:border-[#00A968] transition-colors"
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div>
                        <h3 className="font-semibold text-gray-800 text-lg">
                          {index + 1}. {rec.skill_name}
                        </h3>
                        {rec.explanation && (
                          <p className="text-sm text-gray-600 mt-1">{rec.explanation}</p>
                        )}
                      </div>
                      <div className="bg-[#00A968] text-white px-3 py-1 rounded-full text-sm font-medium">
                        {(rec.score * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div className="grid grid-cols-3 gap-4 mt-3">
                      <div className="text-center">
                        <p className="text-xs text-gray-500 mb-1">Readiness</p>
                        <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-blue-500"
                            style={{ width: `${rec.readiness_score * 100}%` }}
                          />
                        </div>
                        <p className="text-xs text-gray-700 mt-1">
                          {(rec.readiness_score * 100).toFixed(0)}%
                        </p>
                      </div>
                      <div className="text-center">
                        <p className="text-xs text-gray-500 mb-1">Bayesian</p>
                        <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-purple-500"
                            style={{ width: `${rec.bayesian_score * 100}%` }}
                          />
                        </div>
                        <p className="text-xs text-gray-700 mt-1">
                          {(rec.bayesian_score * 100).toFixed(0)}%
                        </p>
                      </div>
                      <div className="text-center">
                        <p className="text-xs text-gray-500 mb-1">Utility</p>
                        <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-green-500"
                            style={{ width: `${rec.utility_score * 100}%` }}
                          />
                        </div>
                        <p className="text-xs text-gray-700 mt-1">
                          {(rec.utility_score * 100).toFixed(0)}%
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-500 text-center py-8">推薦結果がありません</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
