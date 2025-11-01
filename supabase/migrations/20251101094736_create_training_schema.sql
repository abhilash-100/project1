/*
  # Training Schema for Hugging Face Integration
  
  1. New Tables
    - `training_datasets`
      - Stores Hugging Face dataset URLs for training
      - User can add multiple datasets to train on
    - `training_jobs`
      - Tracks training jobs and their status
      - Stores configuration and results
    - `trained_models`
      - Stores information about trained models
      - Links to training jobs and stores metrics
  
  2. Security
    - Enable RLS on all tables
    - Users can only manage their own datasets and models
*/

-- Training Datasets Table
CREATE TABLE IF NOT EXISTS training_datasets (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  name text NOT NULL,
  dataset_url text NOT NULL,
  description text DEFAULT '',
  dataset_type text DEFAULT 'huggingface',
  is_active boolean DEFAULT true,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

ALTER TABLE training_datasets ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own training datasets"
  ON training_datasets FOR SELECT
  TO authenticated
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own training datasets"
  ON training_datasets FOR INSERT
  TO authenticated
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own training datasets"
  ON training_datasets FOR UPDATE
  TO authenticated
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own training datasets"
  ON training_datasets FOR DELETE
  TO authenticated
  USING (auth.uid() = user_id);

-- Training Jobs Table
CREATE TABLE IF NOT EXISTS training_jobs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  job_name text NOT NULL,
  status text DEFAULT 'pending',
  config jsonb DEFAULT '{}',
  dataset_ids uuid[] DEFAULT '{}',
  progress integer DEFAULT 0,
  current_epoch integer DEFAULT 0,
  total_epochs integer DEFAULT 10,
  train_loss float DEFAULT 0,
  train_accuracy float DEFAULT 0,
  val_loss float DEFAULT 0,
  val_accuracy float DEFAULT 0,
  error_message text DEFAULT '',
  started_at timestamptz,
  completed_at timestamptz,
  created_at timestamptz DEFAULT now()
);

ALTER TABLE training_jobs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own training jobs"
  ON training_jobs FOR SELECT
  TO authenticated
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own training jobs"
  ON training_jobs FOR INSERT
  TO authenticated
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own training jobs"
  ON training_jobs FOR UPDATE
  TO authenticated
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own training jobs"
  ON training_jobs FOR DELETE
  TO authenticated
  USING (auth.uid() = user_id);

-- Trained Models Table
CREATE TABLE IF NOT EXISTS trained_models (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  training_job_id uuid REFERENCES training_jobs(id) ON DELETE CASCADE,
  model_name text NOT NULL,
  model_path text NOT NULL,
  model_type text DEFAULT 'pytorch',
  num_classes integer NOT NULL,
  class_names jsonb DEFAULT '[]',
  final_accuracy float DEFAULT 0,
  best_accuracy float DEFAULT 0,
  total_parameters integer DEFAULT 0,
  is_active boolean DEFAULT true,
  created_at timestamptz DEFAULT now()
);

ALTER TABLE trained_models ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own trained models"
  ON trained_models FOR SELECT
  TO authenticated
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own trained models"
  ON trained_models FOR INSERT
  TO authenticated
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own trained models"
  ON trained_models FOR UPDATE
  TO authenticated
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own trained models"
  ON trained_models FOR DELETE
  TO authenticated
  USING (auth.uid() = user_id);

-- Indexes for better performance
CREATE INDEX IF NOT EXISTS idx_training_datasets_user_id ON training_datasets(user_id);
CREATE INDEX IF NOT EXISTS idx_training_jobs_user_id ON training_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON training_jobs(status);
CREATE INDEX IF NOT EXISTS idx_trained_models_user_id ON trained_models(user_id);
CREATE INDEX IF NOT EXISTS idx_trained_models_active ON trained_models(user_id, is_active) WHERE is_active = true;
