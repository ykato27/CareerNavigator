import { Routes, Route, Link, useLocation } from 'react-router-dom';
import { LayoutGrid, Upload, Brain, Database, Settings, HelpCircle, User } from 'lucide-react';
import { Dashboard } from './pages/Dashboard';
import { DataUpload } from './pages/DataUpload';
import { CausalAnalysis } from './pages/CausalAnalysis';

const NavItem = ({ to, icon: Icon, label }: { to: string; icon: any; label: string }) => {
  const location = useLocation();
  const isActive = location.pathname === to;

  return (
    <Link
      to={to}
      className={`
        flex flex-col items-center justify-center py-4 px-2 text-center transition-colors
        ${isActive
          ? 'bg-[#008F58] text-white'
          : 'text-white hover:bg-[#008F58]'
        }
      `}
    >
      <Icon size={24} className="mb-1" />
      <span className="text-xs font-medium">{label}</span>
    </Link>
  );
};

function App() {
  return (
    <div className="flex min-h-screen bg-[#F5F7F9]">
      {/* Sidebar */}
      <aside className="w-20 bg-[#00A968] flex flex-col">
        {/* Logo */}
        <div className="h-16 flex items-center justify-center border-b border-[#008F58]">
          <div className="w-10 h-10 bg-white rounded-md flex items-center justify-center">
            <span className="text-[#00A968] font-bold text-lg">S</span>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 flex flex-col">
          <NavItem to="/" icon={LayoutGrid} label="力量管理" />
          <NavItem to="/causal-analysis" icon={Brain} label="Career" />
          <NavItem to="/data-upload" icon={Upload} label="データ管理" />
          <div className="mt-auto">
            <NavItem to="/settings" icon={Settings} label="設定" />
            <NavItem to="/help" icon={HelpCircle} label="ガイド" />
            <NavItem to="/profile" icon={User} label="Myページ" />
          </div>
        </nav>
      </aside>

      {/* Main Content */}
      <main className="flex-1">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/data-upload" element={<DataUpload />} />
          <Route path="/causal-analysis" element={<CausalAnalysis />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
