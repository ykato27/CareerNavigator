import { useState } from 'react';
import axios from 'axios';
import { Network, Globe, AlertCircle, Loader } from 'lucide-react';
import { clsx } from 'clsx';
import { API_BASE_URL, API_ENDPOINTS } from '../config/constants';

interface GraphVisualizationProps {
  modelId: string | null;
  competenceCode?: string;
  recommendations?: any[];
}

export const GraphVisualization = ({ modelId, competenceCode, recommendations: _recommendations }: GraphVisualizationProps) => {
  const [activeTab, setActiveTab] = useState<'ego' | 'full'>('ego');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [graphHtml, setGraphHtml] = useState<string | null>(null);

  const loadGraph = async (type: 'ego' | 'full') => {
    if (!modelId) {
      setError("モデルIDが見つかりません");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      let response;
      if (type === 'ego') {
        if (!competenceCode) {
          setError("力量コードが指定されていません");
          return;
        }
        response = await axios.post(`${API_BASE_URL}${API_ENDPOINTS.GRAPH_EGO}`, {
          model_id: modelId,
          competence_code: competenceCode
        });
      } else {
        response = await axios.post(`${API_BASE_URL}${API_ENDPOINTS.GRAPH_FULL}`, {
          model_id: modelId
        });
      }

      setGraphHtml(response.data.html);
    } catch (err: any) {
      console.error(err);
      setError(err.response?.data?.detail || "グラフの読み込みに失敗しました");
    } finally {
      setLoading(false);
    }
  };

  const handleTabChange = (tab: 'ego' | 'full') => {
    setActiveTab(tab);
    loadGraph(tab);
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
            <Network className="text-blue-600" size={20} />
          </div>
          <div>
            <h3 className="text-lg font-bold text-gray-800">因果グラフ可視化</h3>
            <p className="text-sm text-gray-500">力量間の因果関係をグラフで表示</p>
          </div>
        </div>

        {/* Tab Selector */}
        <div className="flex gap-2">
          <button
            onClick={() => handleTabChange('ego')}
            className={clsx(
              "px-4 py-2 rounded-lg font-medium transition-all flex items-center gap-2",
              activeTab === 'ego'
                ? "bg-blue-600 text-white shadow-md"
                : "bg-gray-100 text-gray-600 hover:bg-gray-200"
            )}
          >
            <Network size={16} />
            エゴネットワーク
          </button>
          <button
            onClick={() => handleTabChange('full')}
            className={clsx(
              "px-4 py-2 rounded-lg font-medium transition-all flex items-center gap-2",
              activeTab === 'full'
                ? "bg-blue-600 text-white shadow-md"
                : "bg-gray-100 text-gray-600 hover:bg-gray-200"
            )}
          >
            <Globe size={16} />
            全体グラフ
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg flex items-center gap-2 mb-4">
          <AlertCircle size={20} />
          <span className="text-sm">{error}</span>
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="flex items-center justify-center py-12">
          <Loader className="animate-spin text-blue-600" size={32} />
        </div>
      )}

      {/* Graph Display */}
      {!loading && graphHtml && (
        <div className="border border-gray-200 rounded-lg overflow-hidden">
          <iframe
            srcDoc={graphHtml}
            className="w-full h-[600px]"
            title="Causal Graph"
          />
        </div>
      )}

      {/* Empty State */}
      {!loading && !graphHtml && !error && (
        <div className="text-center py-12 text-gray-500">
          タブを選択してグラフを表示
        </div>
      )}
    </div>
  );
};
