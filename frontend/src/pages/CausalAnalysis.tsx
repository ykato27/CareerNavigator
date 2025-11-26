import { useState, useEffect } from 'react';
import axios from 'axios';
import { Search, ArrowRight, Brain, AlertCircle, Upload, CheckCircle2 } from 'lucide-react';
import { clsx } from 'clsx';
import { useNavigate } from 'react-router-dom';
import { ModelTraining } from '../components/ModelTraining';
import { GraphVisualization } from '../components/GraphVisualization';

interface Recommendation {
    rank: number;
    competence_name: string;
    competence_code: string;
    score: number;
    details: {
        readiness_score_normalized: number;
        bayesian_score_normalized: number;
        utility_score_normalized: number;
        readiness_reasons: Array<[string, number]>;
        utility_reasons: Array<[string, number]>;
        bayesian_score: number;
    };
    explanation: string;
}

export const CausalAnalysis = () => {
    const navigate = useNavigate();
    const [memberId, setMemberId] = useState('');
    const [loading, setLoading] = useState(false);
    const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
    const [error, setError] = useState<string | null>(null);
    const [hasUploadedData, setHasUploadedData] = useState(false);
    const [modelId, setModelId] = useState<string | null>(null);
    const [modelTrained, setModelTrained] = useState(false);

    // Check if data has been uploaded and model trained
    useEffect(() => {
        const dataUploaded = sessionStorage.getItem('career_data_uploaded');
        const trainedModelId = sessionStorage.getItem('career_model_id');
        const trained = sessionStorage.getItem('career_model_trained');

        setHasUploadedData(dataUploaded === 'true');
        setModelId(trainedModelId);
        setModelTrained(trained === 'true');
    }, []);

    const handleTrainingComplete = (newModelId: string) => {
        setModelId(newModelId);
        setModelTrained(true);
    };

    const handleAnalyze = async () => {
        if (!memberId) {
            setError("メンバーIDを入力してください");
            return;
        }

        if (!modelId) {
            setError("先にモデルを学習してください");
            return;
        }

        setLoading(true);
        setError(null);

        try {
            const response = await axios.post('http://localhost:8000/api/recommend', {
                model_id: modelId,
                member_id: memberId,
                top_n: 10
            });

            setRecommendations(response.data.recommendations || []);
        } catch (err: any) {
            console.error(err);
            setError(err.response?.data?.detail || "分析に失敗しました");
        } finally {
            setLoading(false);
        }
    };

    // If no data uploaded, show upload prompt
    if (!hasUploadedData) {
        return (
            <div className="space-y-8">
                <div>
                    <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
                        <Brain className="text-primary" />
                        キャリア因果推論推薦
                    </h2>
                    <p className="text-gray-500 mt-1">
                        AIがあなたの保有スキルと目標から、最も効果的なスキル習得パスを提案します。
                    </p>
                </div>

                <div className="bg-yellow-50 border-2 border-yellow-200 rounded-xl p-8 text-center">
                    <Upload className="w-16 h-16 text-yellow-600 mx-auto mb-4" />
                    <h3 className="text-xl font-bold text-gray-800 mb-2">データがアップロードされていません</h3>
                    <p className="text-gray-600 mb-6">
                        分析を実行するには、まず6種類のCSVファイルをアップロードしてください。
                    </p>
                    <button
                        onClick={() => navigate('/upload')}
                        className="bg-primary text-white px-8 py-3 rounded-lg font-bold hover:bg-primary-hover transition-colors inline-flex items-center gap-2"
                    >
                        <Upload size={20} />
                        データ管理へ
                    </button>
                </div>
            </div>
        );
    }

    // If data uploaded but model not trained, show training UI
    if (!modelTrained) {
        const sessionId = sessionStorage.getItem('career_session_id') || 'latest';

        return (
            <div className="space-y-8">
                <div>
                    <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
                        <Brain className="text-primary" />
                        キャリア因果推論推薦
                    </h2>
                    <p className="text-gray-500 mt-1">
                        AIがあなたの保有スキルと目標から、最も効果的なスキル習得パスを提案します。
                    </p>
                </div>

                <ModelTraining
                    sessionId={sessionId}
                    onTrainingComplete={handleTrainingComplete}
                />
            </div>
        );
    }

    // Model trained - show recommendation UI
    return (
        <div className="space-y-8">
            <div>
                <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
                    <Brain className="text-primary" />
                    キャリア因果推論推薦
                </h2>
                <p className="text-gray-500 mt-1">
                    AIがあなたの保有スキルと目標から、最も効果的なスキル習得パスを提案します。
                </p>
            </div>

            {/* Model Status */}
            <div className="bg-green-50 border border-green-200 rounded-lg px-4 py-3 flex items-center gap-2">
                <CheckCircle2 className="text-green-600" size={20} />
                <span className="text-green-800 font-medium">因果モデル学習済み（Model ID: {modelId?.slice(0, 20)}...）</span>
            </div>

            {/* Search Section */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
                <label className="block text-sm font-medium text-gray-700 mb-2">対象メンバーID</label>
                <div className="flex gap-4">
                    <div className="relative flex-1 max-w-md">
                        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
                        <input
                            type="text"
                            value={memberId}
                            onChange={(e) => setMemberId(e.target.value)}
                            placeholder="例: 10001"
                            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent outline-none"
                        />
                    </div>
                    <button
                        onClick={handleAnalyze}
                        disabled={loading}
                        className="bg-primary text-white px-6 py-2 rounded-lg font-bold hover:bg-green-700 transition-colors disabled:bg-gray-400"
                    >
                        {loading ? "分析中..." : "分析実行"}
                    </button>
                </div>
                {error && (
                    <p className="text-red-500 text-sm mt-2 flex items-center gap-1">
                        <AlertCircle size={16} />
                        {error}
                    </p>
                )}
            </div>

            {/* Results Section */}
            {recommendations.length > 0 && (
                <div className="space-y-6">
                    <h3 className="text-xl font-bold text-gray-800">推薦されたスキルパス</h3>

                    <div className="grid gap-4">
                        {recommendations.map((rec, index) => (
                            <div key={rec.competence_code} className="bg-white p-6 rounded-xl shadow-sm border border-gray-200 hover:shadow-md transition-shadow">
                                <div className="flex items-start justify-between">
                                    <div className="flex items-start gap-6 flex-1">
                                        <div className={clsx(
                                            "w-12 h-12 rounded-full flex items-center justify-center text-xl font-bold flex-shrink-0",
                                            index === 0 ? "bg-yellow-100 text-yellow-700" :
                                                index === 1 ? "bg-gray-100 text-gray-700" :
                                                    "bg-orange-100 text-orange-700"
                                        )}>
                                            {rec.rank}
                                        </div>

                                        <div className="flex-1">
                                            <h4 className="text-lg font-bold text-gray-800 mb-3">{rec.competence_name}</h4>

                                            {/* Score Metrics */}
                                            <div className="grid grid-cols-4 gap-4 mb-3">
                                                <div className="bg-gray-50 p-3 rounded-lg">
                                                    <div className="text-xs text-gray-500 mb-1">総合スコア</div>
                                                    <div className="text-xl font-bold text-primary">{rec.score.toFixed(2)}</div>
                                                </div>
                                                <div className="bg-green-50 p-3 rounded-lg">
                                                    <div className="text-xs text-gray-500 mb-1">準備度</div>
                                                    <div className="text-xl font-bold text-green-600">
                                                        {rec.details.readiness_score_normalized.toFixed(2)}
                                                    </div>
                                                </div>
                                                <div className="bg-purple-50 p-3 rounded-lg">
                                                    <div className="text-xs text-gray-500 mb-1">確率</div>
                                                    <div className="text-xl font-bold text-purple-600">
                                                        {rec.details.bayesian_score_normalized.toFixed(2)}
                                                    </div>
                                                </div>
                                                <div className="bg-blue-50 p-3 rounded-lg">
                                                    <div className="text-xs text-gray-500 mb-1">将来性</div>
                                                    <div className="text-xl font-bold text-blue-600">
                                                        {rec.details.utility_score_normalized.toFixed(2)}
                                                    </div>
                                                </div>
                                            </div>

                                            <div className="bg-blue-50 border-l-4 border-blue-400 p-3 rounded text-sm text-gray-700">
                                                {rec.explanation}
                                            </div>

                                            {/* Detailed Reasons */}
                                            <details className="mt-3">
                                                <summary className="cursor-pointer text-sm font-medium text-gray-700 hover:text-primary">
                                                    📋 詳細な推薦理由を表示
                                                </summary>
                                                <div className="mt-3 space-y-3 text-sm">
                                                    {rec.details.readiness_reasons.length > 0 && (
                                                        <div>
                                                            <div className="font-medium text-green-700 mb-1">🟢 準備度（Readiness）</div>
                                                            <div className="pl-4 space-y-1">
                                                                {rec.details.readiness_reasons.slice(0, 5).map(([skill, effect], i) => (
                                                                    <div key={i} className="text-gray-600">
                                                                        • <span className="font-medium">{skill}</span> → 因果効果: {effect.toFixed(3)}
                                                                    </div>
                                                                ))}
                                                            </div>
                                                        </div>
                                                    )}

                                                    {rec.details.bayesian_score > 0 && (
                                                        <div>
                                                            <div className="font-medium text-purple-700 mb-1">🟣 確率（Bayesian）</div>
                                                            <div className="pl-4 text-gray-600">
                                                                あなたと同様のスキルセットを持つ方の {(rec.details.bayesian_score * 100).toFixed(1)}% がこのスキルを習得しています
                                                            </div>
                                                        </div>
                                                    )}

                                                    {rec.details.utility_reasons.length > 0 && (
                                                        <div>
                                                            <div className="font-medium text-blue-700 mb-1">🔵 将来性（Utility）</div>
                                                            <div className="pl-4 space-y-1">
                                                                {rec.details.utility_reasons.slice(0, 5).map(([skill, effect], i) => (
                                                                    <div key={i} className="text-gray-600">
                                                                        • <span className="font-medium">{skill}</span> ← 因果効果: {effect.toFixed(3)}
                                                                    </div>
                                                                ))}
                                                            </div>
                                                        </div>
                                                    )}
                                                </div>
                                            </details>
                                        </div>
                                    </div>

                                    <button className="text-primary hover:bg-green-50 p-2 rounded-full transition-colors">
                                        <ArrowRight size={24} />
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>

                    {/* Graph Visualization Section */}
                    <div className="mt-8">
                        <GraphVisualization
                            modelId={modelId!}
                            recommendations={recommendations}
                        />
                    </div>
                </div>
            )}
        </div>
    );
};
