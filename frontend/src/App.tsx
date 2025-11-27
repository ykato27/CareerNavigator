import { Routes, Route, Link, useLocation } from 'react-router-dom';
import { LayoutGrid, Upload, Brain, Shield, TrendingUp, Database, Settings, HelpCircle, User, FolderOpen, Zap, Network } from 'lucide-react';
import { Dashboard } from './pages/Dashboard';
import { DataUpload } from './pages/DataUpload';
import { CausalAnalysis } from './pages/CausalAnalysis';
import { CausalRecommendation } from './pages/CausalRecommendation';
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
        flex items-center gap-3 py-3 px-4 transition-colors
        ${isActive
          ? 'bg-white text-[#00A968]'
          : 'text-white hover:bg-white/10'
        }
      `}
    >
      <Icon size={20} strokeWidth={2} />
      <span className="text-xs font-medium whitespace-nowrap">{label}</span>
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
      <aside className="w-[90px] bg-[#00A968] flex flex-col">
        {/* Logo */}
        <div className="h-14 flex items-center justify-center px-3 py-3">
          <div className="flex items-center gap-1">
            <div className="w-6 h-6 bg-white rounded flex items-center justify-center flex-shrink-0">
              <LayoutGrid size={14} className="text-[#00A968]" />
            </div>
            <span className="text-white text-xs font-bold">Skillnote</span>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 flex flex-col py-2">
          <NavItem to="/" icon={LayoutGrid} label="力量管理" />
          <NavItem to="/data-upload" icon={Database} label="データ管理" />
          <NavItem to="/model-training" icon={Zap} label="モデル学習" />
          <NavItem to="/causal-recommendation" icon={Brain} label="因果推論" />
          <NavItem to="/career-dashboard" icon={User} label="キャリア" />
          <NavItem to="/skill-map" icon={Network} label="スキルマップ" />
          <div className="mt-auto">
            <NavItem to="/settings" icon={Settings} label="設定" />
            <NavItem to="/help" icon={HelpCircle} label="ガイド" />
          </div>
        </nav>
      </aside>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col">
        {/* Top Navigation Bar */}
        <header className="h-14 bg-white border-b border-gray-200 flex items-center justify-between px-6">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-sm text-gray-700">
              <FolderOpen size={18} className="text-gray-500" />
              <span>{getPageTitle()}</span>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-sm text-gray-700">システム管理者</span>
            <button className="text-sm text-blue-600 hover:text-blue-700">ログアウト</button>
          </div>
        </header>

        {/* Main Content */}
        <main className="flex-1">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/data-upload" element={<DataUpload />} />
            <Route path="/model-training" element={<ModelTraining />} />
            <Route path="/causal-recommendation" element={<CausalRecommendation />} />
            <Route path="/career-dashboard" element={<EmployeeCareerDashboard />} />
            <Route path="/skill-map" element={<OrganizationalSkillMap />} />
            <Route path="/causal-analysis" element={<CausalAnalysis />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}

export default App;
