import { useState } from 'react';
import axios from 'axios';
import { Brain, Settings, Play, CheckCircle2, AlertCircle } from 'lucide-react';
import { clsx } from 'clsx';
import { API_BASE_URL, API_ENDPOINTS } from '../config/constants';

interface ModelTrainingProps {
  sessionId?: string;
  onTrainingComplete: (modelId: string) => void;
}

export const ModelTraining = ({ sessionId: _sessionId, onTrainingComplete }: ModelTrainingProps) => {
  const [training, setTraining] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  // Training parameters
  const [minMembers, setMinMembers] = useState(5);
  const [correlationThreshold, setCorrelationThreshold] = useState(0.3);
  const [weightMode, setWeightMode] = useState<'default' | 'manual' | 'auto'>('default');

  const handleTrain = async () => {
    console.log('=== Training started, setting training=true ===');
    setTraining(true);
    setError(null);
    setSuccess(false);

    try {
      const sessionId = sessionStorage.getItem('career_session_id');
      if (!sessionId) {
        setError("セッションIDが見つかりません。データを再アップロードしてください。");
        return;
      }

      const response = await axios.post(`${API_BASE_URL}${API_ENDPOINTS.TRAIN}`, {
        session_id: sessionId,
        min_members: minMembers,
        correlation_threshold: correlationThreshold,
        weight_mode: weightMode
      });

      const modelId = response.data.model_id;
      sessionStorage.setItem('career_model_id', modelId);
      sessionStorage.setItem('career_model_trained', 'true');

      setSuccess(true);
      onTrainingComplete(modelId);

      console.log("Training success:", response.data);
    } catch (err: any) {
      console.error(err);
      setError(err.response?.data?.detail || "モデル学習に失敗しました");
    } finally {
      setTraining(false);
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center">
          <Settings className="text-green-600" size={20} />
        </div>
        <div>
          <h3 className="text-lg font-bold text-gray-800">因果モデル学習</h3>
          <p className="text-sm text-gray-500">LiNGAMによる因果構造学習を実行</p>
        </div>
      </div>

      <div className="space-y-4">
        {/* Min Members */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            最小メンバー数
          </label>
          <input
            type="number"
            value={minMembers}
            onChange={(e) => setMinMembers(Number(e.target.value))}
            min={1}
            max={20}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent"
          />
          <p className="text-xs text-gray-500 mt-1">
            力量を持つメンバーの最小数（推奨: 5）
          </p>
        </div>

        {/* Correlation Threshold */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            相関閾値
          </label>
          <input
            type="number"
            value={correlationThreshold}
            onChange={(e) => setCorrelationThreshold(Number(e.target.value))}
            min={0}
            max={1}
            step={0.1}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent"
          />
          <p className="text-xs text-gray-500 mt-1">
            因果関係として扱う最小相関値（推奨: 0.3）
          </p>
        </div>

        {/* Weight Mode */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            重みモード
          </label>
          <select
            value={weightMode}
            onChange={(e) => setWeightMode(e.target.value as any)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent"
          >
            <option value="default">デフォルト (均等)</option>
            <option value="manual">手動設定</option>
            <option value="auto">自動最適化 (ベイズ)</option>
          </select>
          <p className="text-xs text-gray-500 mt-1">
            推薦スコアの重み付け方法を選択
          </p>
        </div>

        {/* Error */}
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg flex items-center gap-2">
            <AlertCircle size={20} />
            <span className="text-sm">{error}</span>
          </div>
        )}

        {/* Success */}
        {success && (
          <div className="bg-green-50 border border-green-200 text-green-700 px-4 py-3 rounded-lg flex items-center gap-2">
            <CheckCircle2 size={20} />
            <span className="text-sm">因果モデルの学習が完了しました</span>
          </div>
        )}

        {/* Training Progress - デバッグ用に目立つスタイル */}
        <div
          className="px-4 py-3 rounded-lg space-y-2"
          style={{
            display: training ? 'block' : 'none',
            backgroundColor: '#ff0000',  // デバッグ用：赤色
            border: '3px solid #000000'
          }}
        >
          <div className="flex items-center gap-2" style={{ color: '#ffffff' }}>
            <Brain className="animate-pulse" size={20} />
            <span className="text-sm font-medium">因果モデルを学習中...（60-90秒）</span>
          </div>
          <div className="w-full bg-gray-300 rounded-full h-2">
            <div className="bg-[#00A968] h-2 rounded-full animate-pulse" style={{ width: '100%' }} />
          </div>
        </div>

        {/* Train Button */}
        <button
          onClick={handleTrain}
          disabled={training || success}
          className={clsx(
            "w-full px-6 py-3 rounded-lg font-bold text-white shadow-sm transition-all flex items-center justify-center gap-2",
            training ? "bg-gray-400 cursor-not-allowed" :
              success ? "bg-green-600 cursor-default" : "bg-green-600 hover:bg-green-700"
          )}
        >
          {training ? (
            <>
              <Brain className="animate-pulse" size={20} />
              学習中...
            </>
          ) : success ? (
            <>
              <CheckCircle2 size={20} />
              学習完了
            </>
          ) : (
            <>
              <Play size={20} />
              因果モデルを学習開始
            </>
          )}
        </button>
      </div>
    </div>
  );
};
