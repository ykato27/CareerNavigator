import { useMemo } from 'react';
import Plot from 'react-plotly.js';
import { Loader2 } from 'lucide-react';

interface CareerRoadmapGanttProps {
  ganttChart: any;
  loading?: boolean;
}

export const CareerRoadmapGantt = ({ ganttChart, loading }: CareerRoadmapGanttProps) => {
  const plotData = useMemo(() => {
    if (!ganttChart || !ganttChart.data) {
      return null;
    }
    return ganttChart;
  }, [ganttChart]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96 bg-gray-50 rounded-lg border border-gray-200">
        <div className="text-center">
          <Loader2 size={40} className="animate-spin text-[#00A968] mx-auto mb-3" />
          <p className="text-gray-600">ロードマップを生成中...</p>
        </div>
      </div>
    );
  }

  if (!plotData || !plotData.data || plotData.data.length === 0) {
    return (
      <div className="flex items-center justify-center h-96 bg-gray-50 rounded-lg border border-gray-200">
        <p className="text-gray-600">ガントチャートのデータがありません</p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
      <div className="p-4 bg-blue-50 border-b border-blue-200">
        <h3 className="text-sm font-semibold text-blue-800 mb-2">ロードマップの見方</h3>
        <div className="grid grid-cols-3 gap-3 text-xs text-blue-700">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded" style={{ backgroundColor: '#2ecc71' }}></div>
            <span>高優先度 (スコア ≥ 0.7)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded" style={{ backgroundColor: '#3498db' }}></div>
            <span>中優先度 (スコア ≥ 0.4)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded" style={{ backgroundColor: '#95a5a6' }}></div>
            <span>低優先度</span>
          </div>
        </div>
        <p className="text-xs text-blue-700 mt-2">
          💡 横軸は週を表し、同じ開始週のスキルは並行して学習できます
        </p>
      </div>
      <div className="p-4">
        <Plot
          data={plotData.data}
          layout={plotData.layout}
          config={{
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
          }}
          style={{ width: '100%', height: '100%' }}
          useResizeHandler
        />
      </div>
    </div>
  );
};
