import { useState, useEffect } from 'react';
import { Clock, Flower2, TrendingUp, AlertCircle } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { supabase } from '../lib/supabase';

interface HistoryItem {
  id: string;
  image_path: string;
  created_at: string;
  status: string;
  imageUrl?: string;
  topPrediction?: {
    class_name: string;
    confidence: number;
  };
}

export function HistoryView() {
  const { user } = useAuth();
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    loadHistory();

    const channel = supabase
      .channel('history-changes')
      .on(
        'postgres_changes',
        {
          event: '*',
          schema: 'public',
          table: 'uploads',
          filter: `user_id=eq.${user?.id}`,
        },
        () => {
          loadHistory();
        }
      )
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, [user]);

  const loadHistory = async () => {
    if (!user) return;

    setLoading(true);
    setError('');

    try {
      const { data: uploads, error: uploadsError } = await supabase
        .from('uploads')
        .select('*')
        .eq('user_id', user.id)
        .order('created_at', { ascending: false });

      if (uploadsError) throw uploadsError;

      const historyWithImages = await Promise.all(
        (uploads || []).map(async (upload) => {
          const { data: imageData } = supabase.storage
            .from('flower_images')
            .getPublicUrl(upload.image_path);

          const { data: predictions } = await supabase
            .from('predictions')
            .select('*')
            .eq('upload_id', upload.id)
            .eq('is_top_prediction', true)
            .maybeSingle();

          return {
            ...upload,
            imageUrl: imageData.publicUrl,
            topPrediction: predictions
              ? {
                  class_name: predictions.predicted_class,
                  confidence: predictions.confidence_score,
                }
              : undefined,
          };
        })
      );

      setHistory(historyWithImages);
    } catch (err: any) {
      console.error('History error:', err);
      setError('Failed to load history');
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    }).format(date);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-emerald-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-600">Loading your history...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-2xl mx-auto">
        <div className="bg-red-50 border border-red-200 rounded-lg p-6 flex items-center gap-3">
          <AlertCircle className="w-6 h-6 text-red-600 flex-shrink-0" />
          <p className="text-red-600">{error}</p>
        </div>
      </div>
    );
  }

  if (history.length === 0) {
    return (
      <div className="max-w-2xl mx-auto">
        <div className="bg-white rounded-2xl shadow-xl p-12 text-center">
          <div className="w-20 h-20 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <Flower2 className="w-10 h-10 text-gray-400" />
          </div>
          <h3 className="text-2xl font-bold text-gray-900 mb-2">
            No Classifications Yet
          </h3>
          <p className="text-gray-600 mb-6">
            Start by uploading your first flower image to see your classification history here.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-8">
        <h2 className="text-3xl font-bold text-gray-900 mb-2">
          Classification History
        </h2>
        <p className="text-gray-600">
          View all your previous flower classifications
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {history.map((item) => (
          <div
            key={item.id}
            className="bg-white rounded-xl shadow-lg overflow-hidden hover:shadow-xl transition-shadow"
          >
            <div className="aspect-square relative overflow-hidden bg-gray-100">
              {item.imageUrl && (
                <img
                  src={item.imageUrl}
                  alt="Flower"
                  className="w-full h-full object-cover"
                  onError={(e) => {
                    const target = e.target as HTMLImageElement;
                    target.style.display = 'none';
                  }}
                />
              )}
            </div>

            <div className="p-4 space-y-3">
              {item.topPrediction ? (
                <>
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <Flower2 className="w-4 h-4 text-emerald-600" />
                      <span className="text-sm font-medium text-gray-500">
                        Classification
                      </span>
                    </div>
                    <p className="font-semibold text-gray-900 capitalize">
                      {item.topPrediction.class_name}
                    </p>
                  </div>

                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <TrendingUp className="w-4 h-4 text-emerald-600" />
                      <span className="text-sm font-medium text-gray-500">
                        Confidence
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 bg-gray-200 rounded-full h-2 overflow-hidden">
                        <div
                          className="bg-gradient-to-r from-emerald-500 to-teal-600 h-full rounded-full"
                          style={{
                            width: `${item.topPrediction.confidence * 100}%`,
                          }}
                        />
                      </div>
                      <span className="text-sm font-semibold text-gray-900">
                        {(item.topPrediction.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </>
              ) : (
                <div className="py-2">
                  <span
                    className={`inline-block px-3 py-1 rounded-full text-sm font-medium ${
                      item.status === 'processing'
                        ? 'bg-yellow-100 text-yellow-700'
                        : 'bg-red-100 text-red-700'
                    }`}
                  >
                    {item.status === 'processing' ? 'Processing' : 'Failed'}
                  </span>
                </div>
              )}

              <div className="pt-3 border-t border-gray-100">
                <div className="flex items-center gap-2 text-sm text-gray-500">
                  <Clock className="w-4 h-4" />
                  <span>{formatDate(item.created_at)}</span>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
