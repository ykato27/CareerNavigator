import { useState, useEffect } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import { Map, AlertCircle, Loader2, Users } from 'lucide-react';
import { API_BASE_URL, API_ENDPOINTS } from '../config/constants';

interface Member {
  member_code: string;
  member_name: string;
  display_name: string;
}

interface SkillData {
  skill_code: string;
  skill_name: string;
  category: string;
  readiness_score: number;
  utility_score: number;
  final_score: number;
}

interface ScatterPlotResponse {
  success: boolean;
  model_id: string;
  member_id: string;
  skills: SkillData[];
  metadata: {
    weights: { readiness: number; utility: number; bayesian: number };
    total_skills: number;
  };
}

export const SkillRecommendationMap = () => {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [dataUploaded, setDataUploaded] = useState(false);
  const [modelId, setModelId] = useState('');

  // Member selection
  const [members, setMembers] = useState<Member[]>([]);
  const [selectedMember, setSelectedMember] = useState('');
  const [loadingMembers, setLoadingMembers] = useState(false);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [scatterData, setScatterData] = useState<SkillData[]>([]);
  const [highlightTopN, setHighlightTopN] = useState(5);

  useEffect(() => {
    const sid = sessionStorage.getItem('career_session_id');
    const uploaded = sessionStorage.getItem('career_data_uploaded');
    const savedModelId = sessionStorage.getItem('career_model_id');

    setSessionId(sid);
    setDataUploaded(uploaded === 'true');
    if (savedModelId) {
      setModelId(savedModelId);
    }

    if (sid && uploaded === 'true') {
      loadMembers(sid);
    }
  }, []);

  const loadMembers = async (sid: string) => {
    setLoadingMembers(true);
    try {
      const response = await axios.get(`${API_BASE_URL}${API_ENDPOINTS.SESSION_MEMBERS(sid)}`);
      setMembers(response.data.members);
    } catch (err: any) {
      console.error('Failed to load members:', err);
    } finally {
      setLoadingMembers(false);
    }
  };

  const fetchScatterData = async () => {
    if (!modelId || !selectedMember) {
      setError('モデルIDとメンバーを選択してください');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await axios.post<ScatterPlotResponse>(
        `${API_BASE_URL}${API_ENDPOINTS.SCATTER_PLOT}`,
        {
          model_id: modelId,
          member_id: selectedMember,
        }
      );

      setScatterData(response.data.skills);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'データの取得に失敗しました');
    } finally {
      setLoading(false);
    }
  };

  // Prepare Plotly data
  const getPlotData = () => {
    if (scatterData.length === 0) return [];

    // Sort by final_score to get top N
    const sortedData = [...scatterData].sort((a, b) => b.final_score - a.final_score);
    const topSkills = new Set(sortedData.slice(0, highlightTopN).map(s => s.skill_code));

    const otherSkills = scatterData.filter(s => !topSkills.has(s.skill_code));
    const highlightedSkills = scatterData.filter(s => topSkills.has(s.skill_code));

    const traces: Plotly.Data[] = [];

    // Other skills (blue)
    if (otherSkills.length > 0) {
      traces.push({
        x: otherSkills.map(s => s.readiness_score * 100),
        y: otherSkills.map(s => s.utility_score * 100),
        mode: 'markers',
        type: 'scatter',
        name: 'その他のスキル',
        marker: {
          size: 8,
          color: '#4A90E2',
          opacity: 0.6,
          line: { width: 1, color: 'white' }
        },
        text: otherSkills.map(s => s.skill_name),
        customdata: otherSkills.map(s => [s.final_score, s.category]),
        hovertemplate:
          '<b>%{text}</b><br>' +
          '準備度: %{x:.1f}%<br>' +
          '将来性: %{y:.1f}%<br>' +
          '総合スコア: %{customdata[0]:.3f}<br>' +
          'カテゴリ: %{customdata[1]}<br>' +
          '<extra></extra>'
      });
    }

    // Top skills (red)
    if (highlightedSkills.length > 0) {
      traces.push({
        x: highlightedSkills.map(s => s.readiness_score * 100),
        y: highlightedSkills.map(s => s.utility_score * 100),
        mode: 'markers+text',
        type: 'scatter',
        name: `上位${highlightTopN}推薦`,
        marker: {
          size: 15,
          color: '#E24A4A',
          line: { width: 2, color: 'white' }
        },
        text: highlightedSkills.map(s => s.skill_name),
        textposition: 'top center',
        textfont: { size: 10 },
        customdata: highlightedSkills.map(s => [s.final_score, s.category]),
        hovertemplate:
          '<b>%{text}</b><br>' +
          '準備度: %{x:.1f}%<br>' +
          '将来性: %{y:.1f}%<br>' +
          '総合スコア: %{customdata[0]:.3f}<br>' +
          'カテゴリ: %{customdata[1]}<br>' +
          '<extra></extra>'
      });
    }

    return traces;
  };

  const plotLayout: Partial<Plotly.Layout> = {
    title: {
      text: `スキル推薦マップ: Readiness × Utility<br><sub>メンバー: ${selectedMember} | 総スキル数: ${scatterData.length}</sub>`,
      x: 0.5,
      xanchor: 'center'
    },
    xaxis: {
      title: {
        text: '← 準備不足　　　Readiness（準備度）%　　　準備OK →',
        font: { size: 16, color: '#333', family: 'Arial, sans-serif' },
        standoff: 20
      },
      range: [-5, 105],
      gridcolor: 'lightgray',
      zerolinecolor: 'gray',
      tickfont: { size: 12 },
      ticksuffix: '%'
    },
    yaxis: {
      title: {
        text: '← 将来性低　　　Utility（将来性）%　　　将来性高 →',
        font: { size: 16, color: '#333', family: 'Arial, sans-serif' },
        standoff: 20
      },
      range: [-5, 105],
      gridcolor: 'lightgray',
      zerolinecolor: 'gray',
      tickfont: { size: 12 },
      ticksuffix: '%'
    },
    hovermode: 'closest',
    showlegend: true,
    legend: {
      yanchor: 'top',
      y: 0.99,
      xanchor: 'left',
      x: 0.01
    },
    plot_bgcolor: 'white',
    paper_bgcolor: 'white',
    annotations: [
      {
        x: 90, y: 90,
        text: '🎯 最優先<br>(準備OK・将来性高)',
        showarrow: false,
        font: { size: 10, color: 'green' },
        bgcolor: 'rgba(144, 238, 144, 0.3)'
      },
      {
        x: 10, y: 90,
        text: '📚 基盤構築が必要<br>(準備不足・将来性高)',
        showarrow: false,
        font: { size: 10, color: 'orange' },
        bgcolor: 'rgba(255, 200, 100, 0.3)'
      },
      {
        x: 90, y: 10,
        text: '✅ すぐ習得可能<br>(準備OK・将来性低)',
        showarrow: false,
        font: { size: 10, color: 'blue' },
        bgcolor: 'rgba(173, 216, 230, 0.3)'
      },
      {
        x: 10, y: 10,
        text: '⏸️ 後回し<br>(準備不足・将来性低)',
        showarrow: false,
        font: { size: 10, color: 'gray' },
        bgcolor: 'rgba(200, 200, 200, 0.3)'
      }
    ]
  };

  if (!sessionId || !dataUploaded) {
    return (
      <div className="p-6">
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <div className="flex items-center gap-2 text-yellow-700">
            <AlertCircle className="w-5 h-5" />
            <span>データをアップロードしてからご利用ください</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Map className="w-8 h-8 text-indigo-600" />
        <div>
          <h1 className="text-2xl font-bold text-gray-900">スキル推薦マップ</h1>
          <p className="text-gray-600">Readiness × Utility の散布図で推薦スキルを可視化</p>
        </div>
      </div>

      {/* Controls */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {/* Model ID */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              モデルID
            </label>
            <input
              type="text"
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
              placeholder="model_session_xxx"
            />
          </div>

          {/* Member Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              <Users className="w-4 h-4 inline mr-1" />
              メンバー
            </label>
            <select
              value={selectedMember}
              onChange={(e) => setSelectedMember(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
              disabled={loadingMembers}
            >
              <option value="">メンバーを選択</option>
              {members.map((m) => (
                <option key={m.member_code} value={m.member_code}>
                  {m.display_name}
                </option>
              ))}
            </select>
          </div>

          {/* Highlight Top N */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              ハイライト上位
            </label>
            <select
              value={highlightTopN}
              onChange={(e) => setHighlightTopN(Number(e.target.value))}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
            >
              <option value={3}>上位3件</option>
              <option value={5}>上位5件</option>
              <option value={10}>上位10件</option>
              <option value={20}>上位20件</option>
            </select>
          </div>

          {/* Generate Button */}
          <div className="flex items-end">
            <button
              onClick={fetchScatterData}
              disabled={loading || !modelId || !selectedMember}
              className="w-full px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:bg-gray-400 flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  生成中...
                </>
              ) : (
                <>
                  <Map className="w-4 h-4" />
                  マップ生成
                </>
              )}
            </button>
          </div>
        </div>

        {error && (
          <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 flex items-center gap-2">
            <AlertCircle className="w-4 h-4" />
            {error}
          </div>
        )}
      </div>

      {/* Scatter Plot */}
      {scatterData.length > 0 && (
        <div className="bg-white rounded-lg shadow p-6">
          <Plot
            data={getPlotData()}
            layout={plotLayout}
            config={{
              responsive: true,
              displayModeBar: true,
              modeBarButtonsToRemove: ['lasso2d', 'select2d']
            }}
            style={{ width: '100%', height: '600px' }}
          />

          {/* Legend explanation */}
          <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div className="p-3 bg-green-50 rounded-lg">
              <div className="font-medium text-green-700">🎯 最優先（右上）</div>
              <div className="text-green-600">準備OK・将来性高</div>
            </div>
            <div className="p-3 bg-orange-50 rounded-lg">
              <div className="font-medium text-orange-700">📚 基盤構築が必要（左上）</div>
              <div className="text-orange-600">準備不足・将来性高</div>
            </div>
            <div className="p-3 bg-blue-50 rounded-lg">
              <div className="font-medium text-blue-700">✅ すぐ習得可能（右下）</div>
              <div className="text-blue-600">準備OK・将来性低</div>
            </div>
            <div className="p-3 bg-gray-50 rounded-lg">
              <div className="font-medium text-gray-700">⏸️ 後回し（左下）</div>
              <div className="text-gray-600">準備不足・将来性低</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
