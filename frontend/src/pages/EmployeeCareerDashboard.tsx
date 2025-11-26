import { useState } from 'react';
import axios from 'axios';
import { User, Briefcase, Target, Award, TrendingUp, Loader2, AlertCircle } from 'lucide-react';

interface Skill {
  skill_name: string;
  level: number;
  category: string;
}

interface CareerRecommendation {
  skill_name: string;
  score: number;
  reason: string;
}

interface DashboardData {
  member_id: string;
  member_name: string;
  current_skills: Skill[];
  recommendations: CareerRecommendation[];
}

export const EmployeeCareerDashboard = () => {
  const [memberId, setMemberId] = useState('');
  const [modelId, setModelId] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);

  const handleLoadDashboard = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setDashboardData(null);

    try {
      // Get recommendations
      const response = await axios.post('http://localhost:8000/api/recommend', {
        model_id: modelId,
        member_id: memberId,
        top_n: 5,
      });

      // Mock current skills data (in production, this would come from another API endpoint)
      setDashboardData({
        member_id: memberId,
        member_name: `メンバー ${memberId}`,
        current_skills: [
          { skill_name: 'Python', level: 3, category: 'プログラミング' },
          { skill_name: 'データ分析', level: 2, category: 'Analytics' },
          { skill_name: 'SQL', level: 3, category: 'Database' },
        ],
        recommendations: response.data.recommendations.map((rec: any) => ({
          skill_name: rec.skill_name,
          score: rec.score,
          reason: rec.explanation || '因果推論に基づく推薦',
        })),
      });
    } catch (err: any) {
      setError(err.response?.data?.detail || 'データの読み込みに失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const getLevelColor = (level: number) => {
    if (level >= 4) return 'bg-green-500';
    if (level >= 3) return 'bg-blue-500';
    if (level >= 2) return 'bg-yellow-500';
    return 'bg-gray-400';
  };

  const getLevelText = (level: number) => {
    if (level >= 4) return '上級';
    if (level >= 3) return '中級';
    if (level >= 2) return '初級';
    return '基礎';
  };

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
            個人のスキルセットとキャリア開発の推薦を可視化します
          </p>
        </div>

        {/* Input Form */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
          <h2 className="text-lg font-semibold text-gray-800 mb-4">ダッシュボード設定</h2>
          <form onSubmit={handleLoadDashboard} className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
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
            </div>
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-[#00A968] text-white py-3 rounded-md font-medium hover:bg-[#008F58] transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <Loader2 size={20} className="animate-spin" />
                  読み込み中...
                </>
              ) : (
                <>
                  <Briefcase size={20} />
                  ダッシュボードを表示
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

        {/* Dashboard Content */}
        {dashboardData && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Current Skills */}
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <div className="flex items-center gap-2 mb-4">
                <Award size={24} className="text-[#00A968]" />
                <h2 className="text-lg font-semibold text-gray-800">現在のスキル</h2>
              </div>
              <div className="space-y-4">
                {dashboardData.current_skills.map((skill, index) => (
                  <div key={index} className="border-b border-gray-100 pb-3 last:border-0">
                    <div className="flex items-center justify-between mb-2">
                      <div>
                        <h3 className="font-medium text-gray-800">{skill.skill_name}</h3>
                        <p className="text-xs text-gray-500">{skill.category}</p>
                      </div>
                      <span
                        className={`px-3 py-1 rounded-full text-white text-sm font-medium ${getLevelColor(
                          skill.level
                        )}`}
                      >
                        {getLevelText(skill.level)}
                      </span>
                    </div>
                    <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${getLevelColor(skill.level)}`}
                        style={{ width: `${(skill.level / 5) * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Career Recommendations */}
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <div className="flex items-center gap-2 mb-4">
                <Target size={24} className="text-[#00A968]" />
                <h2 className="text-lg font-semibold text-gray-800">キャリア開発推薦</h2>
              </div>
              <div className="space-y-3">
                {dashboardData.recommendations.map((rec, index) => (
                  <div
                    key={index}
                    className="border border-gray-200 rounded-lg p-4 hover:border-[#00A968] transition-colors"
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <TrendingUp size={18} className="text-[#00A968]" />
                        <h3 className="font-semibold text-gray-800">
                          {index + 1}. {rec.skill_name}
                        </h3>
                      </div>
                      <div className="bg-[#00A968] text-white px-2 py-1 rounded-full text-xs font-medium">
                        {(rec.score * 100).toFixed(0)}%
                      </div>
                    </div>
                    <p className="text-sm text-gray-600 ml-6">{rec.reason}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Career Path */}
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 lg:col-span-2">
              <div className="flex items-center gap-2 mb-4">
                <Briefcase size={24} className="text-[#00A968]" />
                <h2 className="text-lg font-semibold text-gray-800">キャリアパス提案</h2>
              </div>
              <div className="flex items-center justify-between py-4">
                <div className="flex-1 text-center">
                  <div className="w-20 h-20 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-2">
                    <Award size={32} className="text-blue-600" />
                  </div>
                  <p className="font-medium text-gray-800">現在</p>
                  <p className="text-xs text-gray-500">中級エンジニア</p>
                </div>
                <div className="flex-shrink-0 px-4">
                  <div className="h-1 w-16 bg-[#00A968]"></div>
                </div>
                <div className="flex-1 text-center">
                  <div className="w-20 h-20 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-2">
                    <Target size={32} className="text-green-600" />
                  </div>
                  <p className="font-medium text-gray-800">次のステップ</p>
                  <p className="text-xs text-gray-500">シニアエンジニア</p>
                </div>
                <div className="flex-shrink-0 px-4">
                  <div className="h-1 w-16 bg-gray-300"></div>
                </div>
                <div className="flex-1 text-center">
                  <div className="w-20 h-20 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-2">
                    <TrendingUp size={32} className="text-purple-600" />
                  </div>
                  <p className="font-medium text-gray-800">将来の目標</p>
                  <p className="text-xs text-gray-500">テックリード</p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
