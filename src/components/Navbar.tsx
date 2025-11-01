import { useAuth } from '../contexts/AuthContext';
import { Flower2, LogOut, Upload, History } from 'lucide-react';

interface NavbarProps {
  currentView: 'upload' | 'history';
  onViewChange: (view: 'upload' | 'history') => void;
}

export function Navbar({ currentView, onViewChange }: NavbarProps) {
  const { user, signOut } = useAuth();

  return (
    <nav className="bg-white border-b border-gray-200 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-emerald-500 to-teal-600 rounded-lg flex items-center justify-center">
              <Flower2 className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">HSSAN</h1>
              <p className="text-xs text-gray-500">Flower Classification</p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="hidden sm:flex items-center gap-2 bg-gray-100 rounded-lg p-1">
              <button
                onClick={() => onViewChange('upload')}
                className={`flex items-center gap-2 px-4 py-2 rounded-md transition-all ${
                  currentView === 'upload'
                    ? 'bg-white text-emerald-600 shadow-sm'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                <Upload className="w-4 h-4" />
                <span className="font-medium">Upload</span>
              </button>
              <button
                onClick={() => onViewChange('history')}
                className={`flex items-center gap-2 px-4 py-2 rounded-md transition-all ${
                  currentView === 'history'
                    ? 'bg-white text-emerald-600 shadow-sm'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                <History className="w-4 h-4" />
                <span className="font-medium">History</span>
              </button>
            </div>

            <div className="flex items-center gap-3">
              <div className="text-right hidden sm:block">
                <p className="text-sm font-medium text-gray-900">
                  {user?.email?.split('@')[0]}
                </p>
                <p className="text-xs text-gray-500">{user?.email}</p>
              </div>
              <button
                onClick={() => signOut()}
                className="flex items-center gap-2 px-4 py-2 text-gray-700 hover:text-red-600 hover:bg-red-50 rounded-lg transition-all"
                title="Sign Out"
              >
                <LogOut className="w-4 h-4" />
                <span className="hidden sm:inline font-medium">Sign Out</span>
              </button>
            </div>
          </div>
        </div>

        <div className="sm:hidden flex gap-2 pb-3">
          <button
            onClick={() => onViewChange('upload')}
            className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-lg transition-all ${
              currentView === 'upload'
                ? 'bg-emerald-600 text-white'
                : 'bg-gray-100 text-gray-600'
            }`}
          >
            <Upload className="w-4 h-4" />
            <span className="font-medium">Upload</span>
          </button>
          <button
            onClick={() => onViewChange('history')}
            className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-lg transition-all ${
              currentView === 'history'
                ? 'bg-emerald-600 text-white'
                : 'bg-gray-100 text-gray-600'
            }`}
          >
            <History className="w-4 h-4" />
            <span className="font-medium">History</span>
          </button>
        </div>
      </div>
    </nav>
  );
}
