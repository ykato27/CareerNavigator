import { useState, useEffect } from 'react';
import axios from 'axios';
import { Network, Settings, Info, AlertCircle, Loader2 } from 'lucide-react';
import { API_BASE_URL } from '../config/constants';
import { ConstraintManagementPanel } from '../components/ConstraintManagementPanel';

// レイアウトオプション（保守性向上）
const LAYOUT_OPTIONS = [
    { value: 'hierarchical', label: '階層型', description: '上から下への階層構造' },
    { value: 'force', label: '力学モデル', description: '物理演算による自然配置' },
    { value: 'circular', label: '円形', description: '円形配置' }
] as const;

type LayoutType = typeof LAYOUT_OPTIONS[number]['value'];

export const CausalGraphExplorer = () => {
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [modelId, setModelId] = useState('');
    const [graphHtml, setGraphHtml] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    // Graph parameters
    const [threshold, setThreshold] = useState(0.3);
    const [topN, setTopN] = useState(100);
    const [showNegative, setShowNegative] = useState(false);
    const [layout, setLayout] = useState<LayoutType>('hierarchical');

    // Graph stats
    const [nodeCount, setNodeCount] = useState(0);

    useEffect(() => {
        const sid = sessionStorage.getItem('career_session_id');
        const savedModelId = sessionStorage.getItem('career_model_id');

        setSessionId(sid);
        if (savedModelId) {
            setModelId(savedModelId);
            // Auto-load graph on mount
            loadFullGraph(savedModelId);
        }
    }, []);

    const loadFullGraph = async (mid?: string) => {
        const targetModelId = mid || modelId;

        if (!targetModelId) {
            setError('モデルIDが見つかりません。先にモデルを学習してください。');
            return;
        }

        setLoading(true);
        setError('');

        try {
            const response = await axios.post(`${API_BASE_URL}/api/graph/full`, {
                model_id: targetModelId,
                threshold,
                top_n: topN,
                show_negative: showNegative,
                layout
            });

            setGraphHtml(response.data.html);
            setNodeCount(response.data.node_count || topN);
        } catch (err: any) {
            console.error('Failed to load full graph:', err);
            setError(err.response?.data?.detail || '因果グラフの生成に失敗しました');
        } finally {
            setLoading(false);
        }
    };

    const handleConstraintsApplied = () => {
        // Reload graph after constraint application
        loadFullGraph();
    };

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center">
                    <Network className="text-purple-600" size={20} />
                </div>
                <div>
                    <h2 className="text-2xl font-bold text-gray-800">因果グラフエクスプローラー</h2>
                    <p className="text-sm text-gray-500">因果モデルの全体構造を可視化・編集</p>
                </div>
            </div>

            {/* Main Layout */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Left Column: Graph Display */}
                <div className="lg:col-span-2 space-y-6">
                    {/* Graph Settings */}
                    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                        <div className="flex items-center gap-3 mb-4">
                            <Settings size={20} className="text-gray-600" />
                            <h3 className="text-lg font-bold text-gray-800">グラフ設定</h3>
                        </div>

                        <div className="grid grid-cols-4 gap-4">
                            {/* Threshold */}
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">
                                    閾値
                                </label>
                                <input
                                    type="number"
                                    value={threshold}
                                    onChange={(e) => setThreshold(Number(e.target.value))}
                                    min={0}
                                    max={1}
                                    step={0.01}
                                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                                />
                                <p className="text-xs text-gray-500 mt-1">
                                    エッジ表示の最小値
                                </p>
                            </div>

                            {/* Top N */}
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">
                                    表示ノード数
                                </label>
                                <input
                                    type="number"
                                    value={topN}
                                    onChange={(e) => setTopN(Number(e.target.value))}
                                    min={10}
                                    max={1000}
                                    step={10}
                                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                                />
                                <p className="text-xs text-gray-500 mt-1">
                                    上位N個のノード（最大1000）
                                </p>
                            </div>

                            {/* Show Negative */}
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">
                                    負の関係
                                </label>
                                <div className="flex items-center h-10">
                                    <label className="flex items-center gap-2 cursor-pointer">
                                        <input
                                            type="checkbox"
                                            checked={showNegative}
                                            onChange={(e) => setShowNegative(e.target.checked)}
                                            className="w-4 h-4 text-purple-600 border-gray-300 rounded focus:ring-purple-500"
                                        />
                                        <span className="text-sm text-gray-700">表示する</span>
                                    </label>
                                </div>
                            </div>

                            {/* Layout */}
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">
                                    レイアウト
                                </label>
                                <select
                                    value={layout}
                                    onChange={(e) => setLayout(e.target.value as LayoutType)}
                                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                                >
                                    {LAYOUT_OPTIONS.map(option => (
                                        <option key={option.value} value={option.value}>
                                            {option.label}
                                        </option>
                                    ))}
                                </select>
                                <p className="text-xs text-gray-500 mt-1">
                                    {LAYOUT_OPTIONS.find(o => o.value === layout)?.description}
                                </p>
                            </div>
                        </div>

                        <button
                            onClick={() => loadFullGraph()}
                            disabled={loading}
                            className="mt-4 w-full bg-purple-600 text-white py-2 px-4 rounded-lg font-medium hover:bg-purple-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                        >
                            {loading ? (
                                <>
                                    <Loader2 size={20} className="animate-spin" />
                                    生成中...
                                </>
                            ) : (
                                <>
                                    <Network size={20} />
                                    グラフを生成
                                </>
                            )}
                        </button>
                    </div>

                    {/* Error */}
                    {error && (
                        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg flex items-center gap-2">
                            <AlertCircle size={20} />
                            <span className="text-sm">{error}</span>
                        </div>
                    )}

                    {/* Graph Display */}
                    {graphHtml && (
                        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                            <div className="flex items-center justify-between mb-4">
                                <div className="flex items-center gap-3">
                                    <Network size={20} className="text-gray-600" />
                                    <h3 className="text-lg font-bold text-gray-800">因果グラフ全体</h3>
                                </div>
                                <div className="text-sm text-gray-500">
                                    ノード数: {nodeCount}
                                </div>
                            </div>

                            {/* Info */}
                            <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 mb-4 flex items-start gap-2">
                                <Info size={16} className="text-blue-600 flex-shrink-0 mt-0.5" />
                                <div className="text-xs text-blue-800">
                                    <p className="font-medium mb-1">操作方法</p>
                                    <ul className="list-disc list-inside space-y-0.5 text-blue-700">
                                        <li>マウスホイールでズーム</li>
                                        <li>ドラッグで移動</li>
                                        <li>ノードをクリックして詳細表示</li>
                                    </ul>
                                </div>
                            </div>

                            {/* Graph iframe */}
                            <iframe
                                srcDoc={graphHtml}
                                className="w-full border border-gray-200 rounded-lg"
                                style={{ height: '800px' }}
                                title="Causal Graph"
                            />
                        </div>
                    )}

                    {/* No Graph Message */}
                    {!graphHtml && !loading && !error && (
                        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-12 text-center">
                            <Network size={48} className="text-gray-300 mx-auto mb-4" />
                            <p className="text-gray-500">
                                「グラフを生成」ボタンをクリックして、因果グラフ全体を表示します
                            </p>
                        </div>
                    )}
                </div>

                {/* Right Column: Constraint Management */}
                <div className="lg:col-span-1">
                    {modelId && sessionId ? (
                        <ConstraintManagementPanel
                            modelId={modelId}
                            sessionId={sessionId}
                            onConstraintsApplied={handleConstraintsApplied}
                        />
                    ) : (
                        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                            <div className="flex items-start gap-2">
                                <AlertCircle size={16} className="text-yellow-600 flex-shrink-0 mt-0.5" />
                                <div className="text-sm text-yellow-800">
                                    <p className="font-medium">モデルが見つかりません</p>
                                    <p className="mt-1 text-yellow-700">
                                        先にモデルを学習してください。
                                    </p>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};
