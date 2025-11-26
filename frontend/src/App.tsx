import { Routes, Route, Link, useLocation } from 'react-router-dom';
import { Home, Upload, Brain } from 'lucide-react';
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
        flex items-center gap-3 px-4 py-3 rounded-lg transition-all
        ${isActive
          ? 'bg-green-600 text-white shadow-md'
          : 'text-gray-600 hover:bg-gray-100'
        }
      `}
    >
      <Icon size={20} />
      <span className="font-medium">{label}</span>
    </Link>
  );
};

function App() {
  return (
    <div className="flex min-h-screen bg-background">
      {/* Sidebar */}
      <aside className="w-64 bg-white shadow-lg">
        <div className="p-6">
          <div className="flex items-center gap-2 mb-8">
            <div className="w-10 h-10 bg-green-600 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-xl">S</span>
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-800">CareerNavigator</h1>
              <p className="text-xs text-gray-500">AI因果推論推薦</p>
            </div>
          </div>

          <nav className="space-y-2">
            <NavItem to="/" icon={Home} label="ダッシュボード" />
            <NavItem to="/data-upload" icon={Upload} label="データ管理" />
            <NavItem to="/causal-analysis" icon={Brain} label="Career" />
          </nav>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 p-8">
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
