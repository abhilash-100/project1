import { useState } from 'react';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { AuthForm } from './components/AuthForm';
import { Navbar } from './components/Navbar';
import { UploadView } from './components/UploadView';
import { HistoryView } from './components/HistoryView';

function MainApp() {
  const { user, loading } = useAuth();
  const [currentView, setCurrentView] = useState<'upload' | 'history'>('upload');

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-emerald-50 via-teal-50 to-cyan-50 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-emerald-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-600 font-medium">Loading...</p>
        </div>
      </div>
    );
  }

  if (!user) {
    return <AuthForm />;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-50 via-teal-50 to-cyan-50">
      <Navbar currentView={currentView} onViewChange={setCurrentView} />
      <main className="py-8 px-4 sm:px-6 lg:px-8">
        {currentView === 'upload' ? <UploadView /> : <HistoryView />}
      </main>
    </div>
  );
}

function App() {
  return (
    <AuthProvider>
      <MainApp />
    </AuthProvider>
  );
}

export default App;
