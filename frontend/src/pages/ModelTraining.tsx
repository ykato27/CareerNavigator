import { useState, useEffect } from 'react';
import axios from 'axios';
import { Brain, Sliders, TrendingUp, Zap, Loader2, AlertCircle, CheckCircle, Info } from 'lucide-react';
import { API_BASE_URL, API_ENDPOINTS } from '../config/constants';

interface TrainingSummary {
  num_members: number;
  num_skills: number;
  learning_time: number;
  weights: {
    readiness: number;
    bayesian: number;
    utility: number;
  };
}

export const ModelTraining = () => {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [dataUploaded, setDataUploaded] = useState(false);

  // Training parameters
  const [minMembers, setMinMembers] = useState(5);
  const [corrThreshold, setCorrThreshold] = useState(0.2);

  // Weight mode
  const [weightMode, setWeightMode] = useState<'default' | 'manual' | 'auto'>('default');
  const [readinessWeight, setReadinessWeight] = useState(0.6);
  const [bayesianWeight, setBayesianWeight] = useState(0.3);
  const [utilityWeight, setUtilityWeight] = useState(0.1);

  // Training state
  const [training, setTraining] = useState(false);
  const [trained, setTrained] = useState(false);
  const [modelId, setModelId] = useState<string | null>(null);
  const [summary, setSummary] = useState<TrainingSummary | null>(null);
  const [error, setError] = useState('');

  // Weight optimization state
  const [optimizing, setOptimizing] = useState(false);
  const nTrials = 50;

  useEffect(() => {
    const sid = sessionStorage.getItem('career_session_id');
    const uploaded = sessionStorage.getItem('career_data_uploaded');
    setSessionId(sid);
    setDataUploaded(uploaded === 'true');

    // Check if model is already trained
    const savedModelId = sessionStorage.getItem('career_model_id');
    if (savedModelId) {
      setModelId(savedModelId);
      setTrained(true);
    }
  }, []);

  const handleOptimizeWeights = async () => {
    if (!modelId) {
      setError('モデルが学習されていません');
      return;
    }

    setOptimizing(true);
    setError('');

    try {
      const response = await axios.post(`${API_BASE_URL}${API_ENDPOINTS.WEIGHTS_OPTIMIZE}`, {
        model_id: modelId,
        n_trials: nTrials,
        n_jobs: -1,
        holdout_ratio: 0.2,
        top_k: 10
      });

      // Re-train with optimized weights
      setTraining(true);
      const trainResponse = await axios.post(`${API_BASE_URL}${API_ENDPOINTS.TRAIN}`, {
        session_id: sessionId!,
        min_members_per_skill: minMembers,
        correlation_threshold: corrThreshold,
        weights: response.data.optimized_weights
      });

      setModelId(trainResponse.data.model_id);
      setSummary(trainResponse.data.summary);
      sessionStorage.setItem('career_model_id', trainResponse.data.model_id);

      alert('重みを最適化してモデルを再学習しました！');
    } catch (err: any) {
      setError(err.response?.data?.detail || '重みの最適化に失敗しました');
    } finally {
      setOptimizing(false);
      setTraining(false);
    }
  };

  const handleTrain = async () => {
    if (!sessionId) {
      setError('セッションIDが見つかりません。データを再アップロードしてください。');
      return;
    }

    setTraining(true);
    setError('');

    try {
      const weights = weightMode === 'default' ? undefined : {
        readiness: readinessWeight,
        bayesian: bayesianWeight,
        utility: utilityWeight
      };

      const response = await axios.post(`${API_BASE_URL}${API_ENDPOINTS.TRAIN}`, {
        session_id: sessionId,
        min_members_per_skill: minMembers,
        correlation_threshold: corrThreshold,
        weights
      });

      setModelId(response.data.model_id);
      setSummary(response.data.summary);
      setTrained(true);

      // Store model ID for later use
      sessionStorage.setItem('career_model_id', response.data.model_id);
      sessionStorage.setItem('career_model_trained', 'true');

    } catch (err: any) {
      setError(err.response?.data?.detail || '学習に失敗しました');
    } finally {
      setTraining(false);
    }
  };

  const totalWeight = readinessWeight + bayesianWeight + utilityWeight;
  const isWeightValid = Math.abs(totalWeight - 1.0) < 0.01;

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
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <Brain size={32} className="text-[#00A968]" />
            <h1 className="text-3xl font-bold text-gray-800">因果モデルの学習</h1>
          </div>
          <p className="text-gray-600">
            LiNGAM (Linear Non-Gaussian Acyclic Model) を用いてスキル間の因果構造を学習します
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
              </ul>
            </div>
          </div>
        </div>

        {/* Training Parameters */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
          <h2 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
            <Sliders size={20} className="text-[#00A968]" />
            学習パラメータ
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                最小メンバー数/スキル
              </label>
              <input
                type="number"
                value={minMembers}
                onChange={(e) => setMinMembers(parseInt(e.target.value))}
                min="3"
                max="20"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-[#00A968]"
              />
              <p className="text-xs text-gray-500 mt-1">
                これより少ないメンバーしか持っていないスキルは除外されます
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                クラスタリング相関閾値
              </label>
              <input
                type="range"
                value={corrThreshold}
                onChange={(e) => setCorrThreshold(parseFloat(e.target.value))}
                min="0.1"
                max="0.5"
                step="0.05"
                className="w-full"
              />
              <div className="flex justify-between text-xs text-gray-600">
                <span>0.1</span>
                <span className="font-medium">{corrThreshold.toFixed(2)}</span>
                <span>0.5</span>
              </div>
              <p className="text-xs text-gray-500 mt-1">
                この値以上の相関があるスキル同士を同じグループにします
              </p>
            </div>
          </div>
        </div>

        {/* Weight Configuration */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
          <h2 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
            <TrendingUp size={20} className="text-[#00A968]" />
            推薦スコアの重み設定
          </h2>

          <div className="space-y-4">
            <div className="flex flex-col gap-2">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  checked={weightMode === 'default'}
                  onChange={() => setWeightMode('default')}
                  className="text-[#00A968]"
                />
                <span className="font-medium">デフォルト重み（推奨）</span>
              </label>
              <p className="text-sm text-gray-600 ml-6">
                Readiness: 60%, Bayesian: 30%, Utility: 10%
              </p>
            </div>

            <div className="flex flex-col gap-2">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  checked={weightMode === 'manual'}
                  onChange={() => setWeightMode('manual')}
                  className="text-[#00A968]"
                />
                <span className="font-medium">手動で重みを指定</span>
              </label>

              {weightMode === 'manual' && (
                <div className="ml-6 space-y-3 mt-2">
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>Readiness（準備度）</span>
                      <span className="font-medium">{(readinessWeight * 100).toFixed(0)}%</span>
                    </div>
                    <input
                      type="range"
                      value={readinessWeight}
                      onChange={(e) => setReadinessWeight(parseFloat(e.target.value))}
                      min="0"
                      max="1"
                      step="0.05"
                      className="w-full"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>Bayesian（確率）</span>
                      <span className="font-medium">{(bayesianWeight * 100).toFixed(0)}%</span>
                    </div>
                    <input
                      type="range"
                      value={bayesianWeight}
                      onChange={(e) => setBayesianWeight(parseFloat(e.target.value))}
                      min="0"
                      max="1"
                      step="0.05"
                      className="w-full"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>Utility（将来性）</span>
                      <span className="font-medium">{(utilityWeight * 100).toFixed(0)}%</span>
                    </div>
                    <input
                      type="range"
                      value={utilityWeight}
                      onChange={(e) => setUtilityWeight(parseFloat(e.target.value))}
                      min="0"
                      max="1"
                      step="0.05"
                      className="w-full"
                    />
                  </div>

                  <div className={`text-sm p-2 rounded ${isWeightValid ? 'bg-green-50 text-green-700' : 'bg-yellow-50 text-yellow-700'}`}>
                    {isWeightValid ? (
                      <div className="flex items-center gap-2">
                        <CheckCircle size={16} />
                        重みの合計: {totalWeight.toFixed(2)}
                      </div>
                    ) : (
                      <div className="flex items-center gap-2">
                        <AlertCircle size={16} />
                        重みの合計: {totalWeight.toFixed(2)} (適用時に正規化されます)
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
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

        {/* Training Summary */}
        {trained && summary && (
          <div className="bg-green-50 border border-green-200 rounded-lg p-6 mb-6">
            <div className="flex items-start gap-3 mb-4">
              <CheckCircle size={24} className="text-green-600 flex-shrink-0" />
              <div>
                <h3 className="font-semibold text-green-800 mb-1">学習完了！</h3>
                <p className="text-sm text-green-700">モデルID: {modelId}</p>
              </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-white rounded-lg p-3">
                <p className="text-xs text-gray-500 mb-1">メンバー数</p>
                <p className="text-2xl font-bold text-gray-800">{summary.num_members}</p>
              </div>
              <div className="bg-white rounded-lg p-3">
                <p className="text-xs text-gray-500 mb-1">スキル数</p>
                <p className="text-2xl font-bold text-gray-800">{summary.num_skills}</p>
              </div>
              <div className="bg-white rounded-lg p-3">
                <p className="text-xs text-gray-500 mb-1">学習時間</p>
                <p className="text-2xl font-bold text-gray-800">{summary.learning_time}s</p>
              </div>
              <div className="bg-white rounded-lg p-3">
                <p className="text-xs text-gray-500 mb-1">重み設定</p>
                <div className="text-xs text-gray-700">
                  <div>R: {(summary.weights.readiness * 100).toFixed(0)}%</div>
                  <div>B: {(summary.weights.bayesian * 100).toFixed(0)}%</div>
                  <div>U: {(summary.weights.utility * 100).toFixed(0)}%</div>
                </div>
              </div>
            </div>

            <div className="mt-4 flex items-center justify-between">
              <p className="text-sm text-green-700">
                💡 サイドバーから「因果推論」または「キャリア」ページで推薦機能を利用できます
              </p>
              <button
                onClick={handleOptimizeWeights}
                disabled={optimizing}
                className="bg-purple-600 text-white px-4 py-2 rounded-md hover:bg-purple-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              >
                {optimizing ? (
                  <>
                    <Loader2 size={16} className="animate-spin" />
                    最適化中...
                  </>
                ) : (
                  <>
                    <Zap size={16} />
                    重みを最適化
                  </>
                )}
              </button>
            </div>
          </div>
        )}

        {/* Train Button */}
        <button
          onClick={handleTrain}
          disabled={training}
          className={`w-full py-4 rounded-lg font-semibold text-white transition-all flex items-center justify-center gap-2 ${training ? 'bg-gray-400 cursor-not-allowed' :
              trained ? 'bg-blue-600 hover:bg-blue-700' :
                'bg-[#00A968] hover:bg-[#008F58]'
            }`}
        >
          {training ? (
            <>
              <Loader2 size={20} className="animate-spin" />
              因果構造を学習中...
            </>
          ) : trained ? (
            <>
              <Zap size={20} />
              モデルを再学習する
            </>
          ) : (
            <>
              <Zap size={20} />
              因果モデルを学習開始
            </>
          )}
        </button>
      </div>
    </div>
  );
};
