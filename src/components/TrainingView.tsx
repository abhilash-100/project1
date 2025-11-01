import { useState, useEffect } from 'react';
import { Database, Plus, Trash2, Play, AlertCircle, CheckCircle, Loader2, Beaker } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { supabase } from '../lib/supabase';

interface TrainingDataset {
  id: string;
  name: string;
  dataset_url: string;
  description: string;
  is_active: boolean;
  created_at: string;
}

interface TrainingJob {
  id: string;
  job_name: string;
  status: string;
  progress: number;
  current_epoch: number;
  total_epochs: number;
  train_accuracy: number;
  val_accuracy: number;
  error_message: string;
  created_at: string;
}

export function TrainingView() {
  const { user } = useAuth();
  const [datasets, setDatasets] = useState<TrainingDataset[]>([]);
  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [loading, setLoading] = useState(true);
  const [showAddDataset, setShowAddDataset] = useState(false);
  const [showStartTraining, setShowStartTraining] = useState(false);

  const [newDataset, setNewDataset] = useState({
    name: '',
    dataset_url: '',
    description: '',
  });

  const [trainingConfig, setTrainingConfig] = useState({
    job_name: '',
    batch_size: 32,
    learning_rate: 0.0001,
    num_epochs: 10,
    use_pretrained: true,
  });

  useEffect(() => {
    loadDatasets();
    loadJobs();

    const datasetsChannel = supabase
      .channel('datasets-changes')
      .on(
        'postgres_changes',
        {
          event: '*',
          schema: 'public',
          table: 'training_datasets',
          filter: `user_id=eq.${user?.id}`,
        },
        () => {
          loadDatasets();
        }
      )
      .subscribe();

    const jobsChannel = supabase
      .channel('jobs-changes')
      .on(
        'postgres_changes',
        {
          event: '*',
          schema: 'public',
          table: 'training_jobs',
          filter: `user_id=eq.${user?.id}`,
        },
        () => {
          loadJobs();
        }
      )
      .subscribe();

    return () => {
      supabase.removeChannel(datasetsChannel);
      supabase.removeChannel(jobsChannel);
    };
  }, [user]);

  const loadDatasets = async () => {
    if (!user) return;

    try {
      const { data, error } = await supabase
        .from('training_datasets')
        .select('*')
        .eq('user_id', user.id)
        .order('created_at', { ascending: false });

      if (error) throw error;
      setDatasets(data || []);
    } catch (err) {
      console.error('Error loading datasets:', err);
    } finally {
      setLoading(false);
    }
  };

  const loadJobs = async () => {
    if (!user) return;

    try {
      const { data, error } = await supabase
        .from('training_jobs')
        .select('*')
        .eq('user_id', user.id)
        .order('created_at', { ascending: false })
        .limit(10);

      if (error) throw error;
      setJobs(data || []);
    } catch (err) {
      console.error('Error loading jobs:', err);
    }
  };

  const addDataset = async () => {
    if (!user || !newDataset.name || !newDataset.dataset_url) return;

    try {
      const { error } = await supabase.from('training_datasets').insert({
        user_id: user.id,
        name: newDataset.name,
        dataset_url: newDataset.dataset_url,
        description: newDataset.description,
        is_active: true,
      });

      if (error) throw error;

      setNewDataset({ name: '', dataset_url: '', description: '' });
      setShowAddDataset(false);
      loadDatasets();
    } catch (err: any) {
      console.error('Error adding dataset:', err);
      alert(`Failed to add dataset: ${err.message}`);
    }
  };

  const deleteDataset = async (id: string) => {
    if (!confirm('Are you sure you want to delete this dataset?')) return;

    try {
      const { error } = await supabase
        .from('training_datasets')
        .delete()
        .eq('id', id);

      if (error) throw error;
      loadDatasets();
    } catch (err: any) {
      console.error('Error deleting dataset:', err);
      alert(`Failed to delete dataset: ${err.message}`);
    }
  };

  const toggleDataset = async (id: string, currentState: boolean) => {
    try {
      const { error } = await supabase
        .from('training_datasets')
        .update({ is_active: !currentState })
        .eq('id', id);

      if (error) throw error;
      loadDatasets();
    } catch (err: any) {
      console.error('Error toggling dataset:', err);
    }
  };

  const startTraining = async () => {
    if (!user || !trainingConfig.job_name) {
      alert('Please provide a job name');
      return;
    }

    const activeDatasets = datasets.filter((d) => d.is_active);
    if (activeDatasets.length === 0) {
      alert('Please activate at least one dataset');
      return;
    }

    try {
      const { error } = await supabase.from('training_jobs').insert({
        user_id: user.id,
        job_name: trainingConfig.job_name,
        status: 'pending',
        config: {
          batch_size: trainingConfig.batch_size,
          learning_rate: trainingConfig.learning_rate,
          use_pretrained: trainingConfig.use_pretrained,
        },
        dataset_ids: activeDatasets.map((d) => d.id),
        total_epochs: trainingConfig.num_epochs,
      });

      if (error) throw error;

      setTrainingConfig({
        job_name: '',
        batch_size: 32,
        learning_rate: 0.0001,
        num_epochs: 10,
        use_pretrained: true,
      });
      setShowStartTraining(false);
      loadJobs();
      alert('Training job created! Note: Backend training script needs to be run separately.');
    } catch (err: any) {
      console.error('Error starting training:', err);
      alert(`Failed to start training: ${err.message}`);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-600" />;
      case 'running':
        return <Loader2 className="w-5 h-5 text-blue-600 animate-spin" />;
      case 'failed':
        return <AlertCircle className="w-5 h-5 text-red-600" />;
      default:
        return <Loader2 className="w-5 h-5 text-gray-400" />;
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="text-center">
          <Loader2 className="w-16 h-16 text-emerald-500 animate-spin mx-auto mb-4" />
          <p className="text-gray-600">Loading training workspace...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold text-gray-900">Training Manager</h2>
          <p className="text-gray-600 mt-1">
            Manage Hugging Face datasets and train custom models
          </p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={() => setShowAddDataset(true)}
            className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-emerald-500 to-teal-600 text-white rounded-lg font-semibold hover:from-emerald-600 hover:to-teal-700 transition-all shadow-lg"
          >
            <Plus className="w-5 h-5" />
            Add Dataset
          </button>
          <button
            onClick={() => setShowStartTraining(true)}
            className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-lg font-semibold hover:from-blue-600 hover:to-indigo-700 transition-all shadow-lg"
          >
            <Play className="w-5 h-5" />
            Start Training
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="bg-white rounded-2xl shadow-xl p-6">
          <div className="flex items-center gap-3 mb-6">
            <Database className="w-6 h-6 text-emerald-600" />
            <h3 className="text-xl font-bold text-gray-900">Training Datasets</h3>
          </div>

          {datasets.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <Database className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p>No datasets added yet</p>
              <p className="text-sm mt-1">Add Hugging Face dataset URLs to get started</p>
            </div>
          ) : (
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {datasets.map((dataset) => (
                <div
                  key={dataset.id}
                  className={`border rounded-lg p-4 transition-all ${
                    dataset.is_active
                      ? 'border-emerald-300 bg-emerald-50'
                      : 'border-gray-200 bg-gray-50'
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <h4 className="font-semibold text-gray-900">{dataset.name}</h4>
                      <p className="text-sm text-gray-600 mt-1 break-all">
                        {dataset.dataset_url}
                      </p>
                      {dataset.description && (
                        <p className="text-sm text-gray-500 mt-2">{dataset.description}</p>
                      )}
                    </div>
                    <div className="flex gap-2 ml-3">
                      <button
                        onClick={() => toggleDataset(dataset.id, dataset.is_active)}
                        className={`px-3 py-1 text-xs rounded-full font-medium transition-colors ${
                          dataset.is_active
                            ? 'bg-emerald-600 text-white'
                            : 'bg-gray-300 text-gray-700'
                        }`}
                      >
                        {dataset.is_active ? 'Active' : 'Inactive'}
                      </button>
                      <button
                        onClick={() => deleteDataset(dataset.id)}
                        className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="bg-white rounded-2xl shadow-xl p-6">
          <div className="flex items-center gap-3 mb-6">
            <Beaker className="w-6 h-6 text-blue-600" />
            <h3 className="text-xl font-bold text-gray-900">Training Jobs</h3>
          </div>

          {jobs.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <Beaker className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p>No training jobs yet</p>
              <p className="text-sm mt-1">Start training to see job history</p>
            </div>
          ) : (
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {jobs.map((job) => (
                <div key={job.id} className="border border-gray-200 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-semibold text-gray-900">{job.job_name}</h4>
                    {getStatusIcon(job.status)}
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600">Status:</span>
                      <span className="font-medium capitalize">{job.status}</span>
                    </div>
                    {job.status === 'running' && (
                      <>
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-600">Epoch:</span>
                          <span className="font-medium">
                            {job.current_epoch}/{job.total_epochs}
                          </span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-gradient-to-r from-blue-500 to-indigo-600 h-2 rounded-full transition-all"
                            style={{ width: `${job.progress}%` }}
                          />
                        </div>
                        {job.train_accuracy > 0 && (
                          <div className="flex justify-between text-sm">
                            <span className="text-gray-600">Train Acc:</span>
                            <span className="font-medium">
                              {job.train_accuracy.toFixed(2)}%
                            </span>
                          </div>
                        )}
                        {job.val_accuracy > 0 && (
                          <div className="flex justify-between text-sm">
                            <span className="text-gray-600">Val Acc:</span>
                            <span className="font-medium text-emerald-600">
                              {job.val_accuracy.toFixed(2)}%
                            </span>
                          </div>
                        )}
                      </>
                    )}
                    {job.status === 'completed' && job.val_accuracy > 0 && (
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600">Final Accuracy:</span>
                        <span className="font-medium text-emerald-600">
                          {job.val_accuracy.toFixed(2)}%
                        </span>
                      </div>
                    )}
                    {job.error_message && (
                      <p className="text-sm text-red-600">{job.error_message}</p>
                    )}
                    <p className="text-xs text-gray-500">
                      {new Date(job.created_at).toLocaleString()}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {showAddDataset && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-2xl shadow-2xl max-w-2xl w-full p-8">
            <h3 className="text-2xl font-bold text-gray-900 mb-6">Add Training Dataset</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Dataset Name
                </label>
                <input
                  type="text"
                  value={newDataset.name}
                  onChange={(e) => setNewDataset({ ...newDataset, name: e.target.value })}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none"
                  placeholder="e.g., Flowers 102 Dataset"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Hugging Face Dataset URL
                </label>
                <input
                  type="text"
                  value={newDataset.dataset_url}
                  onChange={(e) =>
                    setNewDataset({ ...newDataset, dataset_url: e.target.value })
                  }
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none"
                  placeholder="hf://datasets/huggan/flowers-102-categories/data/train-00000-of-00001.parquet"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Description (Optional)
                </label>
                <textarea
                  value={newDataset.description}
                  onChange={(e) =>
                    setNewDataset({ ...newDataset, description: e.target.value })
                  }
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none"
                  rows={3}
                  placeholder="Brief description of the dataset..."
                />
              </div>
            </div>
            <div className="flex gap-3 mt-6">
              <button
                onClick={() => setShowAddDataset(false)}
                className="flex-1 px-6 py-3 border border-gray-300 text-gray-700 rounded-lg font-semibold hover:bg-gray-50 transition-all"
              >
                Cancel
              </button>
              <button
                onClick={addDataset}
                disabled={!newDataset.name || !newDataset.dataset_url}
                className="flex-1 px-6 py-3 bg-gradient-to-r from-emerald-500 to-teal-600 text-white rounded-lg font-semibold hover:from-emerald-600 hover:to-teal-700 transition-all disabled:opacity-50"
              >
                Add Dataset
              </button>
            </div>
          </div>
        </div>
      )}

      {showStartTraining && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-2xl shadow-2xl max-w-2xl w-full p-8">
            <h3 className="text-2xl font-bold text-gray-900 mb-6">Start Training Job</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Job Name
                </label>
                <input
                  type="text"
                  value={trainingConfig.job_name}
                  onChange={(e) =>
                    setTrainingConfig({ ...trainingConfig, job_name: e.target.value })
                  }
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
                  placeholder="e.g., Flower Model v1"
                />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Batch Size
                  </label>
                  <input
                    type="number"
                    value={trainingConfig.batch_size}
                    onChange={(e) =>
                      setTrainingConfig({
                        ...trainingConfig,
                        batch_size: parseInt(e.target.value),
                      })
                    }
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Epochs
                  </label>
                  <input
                    type="number"
                    value={trainingConfig.num_epochs}
                    onChange={(e) =>
                      setTrainingConfig({
                        ...trainingConfig,
                        num_epochs: parseInt(e.target.value),
                      })
                    }
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
                  />
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Learning Rate
                </label>
                <input
                  type="number"
                  step="0.0001"
                  value={trainingConfig.learning_rate}
                  onChange={(e) =>
                    setTrainingConfig({
                      ...trainingConfig,
                      learning_rate: parseFloat(e.target.value),
                    })
                  }
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
                />
              </div>
              <div className="flex items-center gap-3 p-4 bg-gray-50 rounded-lg">
                <input
                  type="checkbox"
                  id="pretrained"
                  checked={trainingConfig.use_pretrained}
                  onChange={(e) =>
                    setTrainingConfig({
                      ...trainingConfig,
                      use_pretrained: e.target.checked,
                    })
                  }
                  className="w-5 h-5 text-blue-600 rounded"
                />
                <label htmlFor="pretrained" className="text-sm font-medium text-gray-700">
                  Use Pre-trained MobileNetV2 (Recommended)
                </label>
              </div>
              <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                <p className="text-sm text-blue-800">
                  <strong>Active Datasets:</strong>{' '}
                  {datasets.filter((d) => d.is_active).length} dataset(s) will be used for
                  training
                </p>
              </div>
            </div>
            <div className="flex gap-3 mt-6">
              <button
                onClick={() => setShowStartTraining(false)}
                className="flex-1 px-6 py-3 border border-gray-300 text-gray-700 rounded-lg font-semibold hover:bg-gray-50 transition-all"
              >
                Cancel
              </button>
              <button
                onClick={startTraining}
                disabled={!trainingConfig.job_name}
                className="flex-1 px-6 py-3 bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-lg font-semibold hover:from-blue-600 hover:to-indigo-700 transition-all disabled:opacity-50"
              >
                Start Training
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
