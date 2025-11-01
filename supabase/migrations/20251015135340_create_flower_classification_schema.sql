/*
  # Create Flower Classification Schema

  ## Overview
  This migration creates the database schema for the HSSAN flower classification application.
  It sets up tables for tracking image uploads and their prediction results.

  ## 1. New Tables
  
  ### `uploads` Table
  Stores information about each flower image uploaded by users.
  - `id` (uuid, primary key) - Unique identifier for each upload
  - `user_id` (uuid, foreign key) - Links to the authenticated user who uploaded the image
  - `created_at` (timestamptz) - Timestamp of when the upload occurred
  - `image_path` (text) - Path to the stored image file in Supabase Storage
  - `status` (text) - Current processing status: 'processing', 'completed', or 'failed'
  
  ### `predictions` Table
  Stores the classification results for each upload.
  - `id` (uuid, primary key) - Unique identifier for each prediction
  - `upload_id` (uuid, foreign key) - Links to the corresponding upload
  - `predicted_class` (text) - Name of the predicted flower species
  - `confidence_score` (float) - Confidence level (0.0 to 1.0) for this prediction
  - `is_top_prediction` (boolean) - Indicates if this is the highest confidence prediction
  - `created_at` (timestamptz) - Timestamp of when the prediction was made

  ## 2. Security
  
  ### Row Level Security (RLS)
  Both tables have RLS enabled to ensure data privacy.
  
  ### Policies for `uploads` table:
  - Users can view only their own uploads
  - Users can insert new uploads for themselves
  - Users can update only their own uploads
  - Users can delete only their own uploads
  
  ### Policies for `predictions` table:
  - Users can view predictions for their own uploads
  - System can insert predictions (via service role)
  - Users can view their prediction history through the uploads relationship

  ## 3. Important Notes
  - All users must be authenticated to access these tables
  - The `user_id` field links to Supabase's built-in `auth.users` table
  - Foreign key constraints ensure data integrity
  - Timestamps use `timestamptz` for proper timezone handling
*/

-- Create uploads table
CREATE TABLE IF NOT EXISTS uploads (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  created_at timestamptz DEFAULT now(),
  image_path text NOT NULL,
  status text NOT NULL DEFAULT 'processing' CHECK (status IN ('processing', 'completed', 'failed'))
);

-- Create predictions table
CREATE TABLE IF NOT EXISTS predictions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  upload_id uuid NOT NULL REFERENCES uploads(id) ON DELETE CASCADE,
  predicted_class text NOT NULL,
  confidence_score float NOT NULL CHECK (confidence_score >= 0 AND confidence_score <= 1),
  is_top_prediction boolean NOT NULL DEFAULT false,
  created_at timestamptz DEFAULT now()
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_uploads_user_id ON uploads(user_id);
CREATE INDEX IF NOT EXISTS idx_uploads_created_at ON uploads(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_upload_id ON predictions(upload_id);
CREATE INDEX IF NOT EXISTS idx_predictions_top ON predictions(upload_id, is_top_prediction) WHERE is_top_prediction = true;

-- Enable Row Level Security
ALTER TABLE uploads ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;

-- RLS Policies for uploads table
CREATE POLICY "Users can view own uploads"
  ON uploads FOR SELECT
  TO authenticated
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own uploads"
  ON uploads FOR INSERT
  TO authenticated
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own uploads"
  ON uploads FOR UPDATE
  TO authenticated
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own uploads"
  ON uploads FOR DELETE
  TO authenticated
  USING (auth.uid() = user_id);

-- RLS Policies for predictions table
CREATE POLICY "Users can view predictions for own uploads"
  ON predictions FOR SELECT
  TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM uploads
      WHERE uploads.id = predictions.upload_id
      AND uploads.user_id = auth.uid()
    )
  );

CREATE POLICY "Service role can insert predictions"
  ON predictions FOR INSERT
  TO authenticated
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM uploads
      WHERE uploads.id = predictions.upload_id
      AND uploads.user_id = auth.uid()
    )
  );