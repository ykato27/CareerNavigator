import { Routes, Route, Link, useLocation } from 'react-router-dom';
import { LayoutGrid, Brain, Database, Settings, HelpCircle, User, FolderOpen, Zap, Network } from 'lucide-react';
import { Dashboard } from './pages/Dashboard';
import { DataUpload } from './pages/DataUpload';
import { CausalAnalysis } from './pages/CausalAnalysis';
import { CausalRecommendation } from './pages/CausalRecommendation';
import { CausalGraphExplorer } from './pages/CausalGraphExplorer';
import { EmployeeCareerDashboard } from './pages/EmployeeCareerDashboard';
import { ModelTraining } from './pages/ModelTraining';
import { OrganizationalSkillMap } from './pages/OrganizationalSkillMap';

const NavItem = ({ to, icon: Icon, label }: { to: string; icon: any; label: string }) => {
  const location = useLocation();
  const isActive = location.pathname === to;

  return (
    <Link
      to={to}
      className={`
        flex items-center gap-3 py-3 px-4 rounded-lg transition-all duration-200 group relative overflow-hidden
        ${isActive
          ? 'bg-white text-[#00A968] shadow-sm font-bold'
          : 'text-white hover:bg-[#00965E] font-medium'
        }
      `}
    >
      {isActive && (
        <div className="absolute left-0 top-0 bottom-0 w-1 bg-[#008F58] rounded-l-lg" />
      )}
      <Icon size={20} strokeWidth={isActive ? 2.5 : 2} className={`transition-transform duration-200 ${isActive ? 'scale-110' : 'group-hover:scale-110'}`} />
      <span className="text-sm whitespace-nowrap tracking-wide">{label}</span>
      {!isActive && (
        <div className="absolute right-2 opacity-0 group-hover:opacity-100 transition-opacity">
          <div className="w-1.5 h-1.5 rounded-full bg-white/50" />
        </div>
      )}
    </Link>
  );
};

function App() {
  const location = useLocation();

  // Get page title based on current route
  const getPageTitle = () => {
    switch (location.pathname) {
      case '/':
        return '力量管理';
      case '/data-upload':
        return 'データ管理';
      case '/model-training':
        return 'モデル学習';
      case '/causal-recommendation':
        return '因果推論';
      case '/causal-graph':
        return '因果グラフ全体';
      case '/career-dashboard':
        return 'キャリア';
      case '/skill-map':
        return 'スキルマップ';
      case '/settings':
        return '設定';
      case '/help':
        return 'ガイド';
      default:
        return '力量管理';
    }
  };

  return (
    <div className="flex min-h-screen bg-[#F5F7F9]">
      {/* Sidebar */}
      <aside className="w-[240px] bg-[#00A968] flex flex-col h-screen sticky top-0 shadow-lg z-50">
        {/* Logo */}
        <div className="h-16 flex items-center px-6 border-b border-[#00965E]">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-white rounded flex items-center justify-center flex-shrink-0 shadow-sm">
              <LayoutGrid size={20} className="text-[#00A968]" />
            </div>
            <span className="text-white text-lg font-bold tracking-tight">Skillnote</span>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 flex flex-col py-4 overflow-y-auto custom-scrollbar">
          <div className="space-y-1 px-3">
            <NavItem to="/" icon={LayoutGrid} label="力量管理" />
            <NavItem to="/skill-map" icon={Network} label="スキルマップ" />
            <NavItem to="/career-dashboard" icon={User} label="キャリア" />
            <div className="my-4 border-t border-[#00965E]/50 mx-2"></div>
            <NavItem to="/data-upload" icon={Database} label="データ管理" />
            <NavItem to="/model-training" icon={Zap} label="モデル学習" />
            <NavItem to="/causal-recommendation" icon={Brain} label="因果推論" />
            <NavItem to="/causal-graph" icon={Network} label="因果グラフ全体" />
          </div>

          <div className="mt-auto px-3 pb-4">
            <div className="my-4 border-t border-[#00965E]/50 mx-2"></div>
            <NavItem to="/settings" icon={Settings} label="設定" />
            <NavItem to="/help" icon={HelpCircle} label="ガイド" />
          </div>
        </nav>
      </aside>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top Navigation Bar */}
        <header className="h-16 bg-white border-b border-gray-200 flex items-center justify-between px-8 sticky top-0 z-40 shadow-sm">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-sm text-gray-600 bg-gray-50 px-3 py-1.5 rounded-md border border-gray-200">
              <FolderOpen size={16} className="text-[#00A968]" />
              <span className="text-gray-400">/</span>
              <span className="font-medium text-gray-800">{getPageTitle()}</span>
            </div>
          </div>
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-[#00A968]/10 rounded-full flex items-center justify-center text-[#00A968]">
                <User size={16} />
              </div>
              <span className="text-sm font-medium text-gray-700">システム管理者</span>
            </div>
            <button className="text-sm text-gray-500 hover:text-[#00A968] transition-colors font-medium">ログアウト</button>
          </div>
        </header>

        {/* Main Content */}
        <main className="flex-1 p-8 overflow-x-hidden">
          <div className="max-w-7xl mx-auto w-full">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/data-upload" element={<DataUpload />} />
              <Route path="/model-training" element={<ModelTraining />} />
              <Route path="/causal-recommendation" element={<CausalRecommendation />} />
              <Route path="/causal-graph" element={<CausalGraphExplorer />} />
              <Route path="/career-dashboard" element={<EmployeeCareerDashboard />} />
              <Route path="/skill-map" element={<OrganizationalSkillMap />} />
              <Route path="/causal-analysis" element={<CausalAnalysis />} />
            </Routes>
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
