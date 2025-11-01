/*
  # Create Storage Bucket for Flower Images

  ## Overview
  This migration sets up a storage bucket for user-uploaded flower images with appropriate security policies.

  ## 1. Storage Bucket
  
  ### `flower_images` Bucket
  A secure storage bucket for flower image uploads.
  - Only authenticated users can upload images
  - Users can only access their own images
  - Public access is disabled for privacy
  
  ## 2. Security Policies
  
  ### Upload Policy
  - Authenticated users can upload images to their own user folder
  - Path structure: `{user_id}/{filename}`
  
  ### Select Policy
  - Users can view/download only their own uploaded images
  
  ### Update Policy
  - Users can update only their own images
  
  ### Delete Policy
  - Users can delete only their own images

  ## 3. Important Notes
  - All images must be stored in a user-specific folder structure
  - File size limits and allowed MIME types can be enforced at the application level
  - The bucket is private by default - no anonymous access
*/

-- Create the storage bucket for flower images
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
  'flower_images',
  'flower_images',
  false,
  10485760, -- 10MB limit
  ARRAY['image/jpeg', 'image/jpg', 'image/png', 'image/webp']::text[]
)
ON CONFLICT (id) DO NOTHING;

-- Allow authenticated users to upload images to their own folder
CREATE POLICY "Users can upload own images"
  ON storage.objects FOR INSERT
  TO authenticated
  WITH CHECK (
    bucket_id = 'flower_images' AND
    (storage.foldername(name))[1] = auth.uid()::text
  );

-- Allow authenticated users to view their own images
CREATE POLICY "Users can view own images"
  ON storage.objects FOR SELECT
  TO authenticated
  USING (
    bucket_id = 'flower_images' AND
    (storage.foldername(name))[1] = auth.uid()::text
  );

-- Allow authenticated users to update their own images
CREATE POLICY "Users can update own images"
  ON storage.objects FOR UPDATE
  TO authenticated
  USING (
    bucket_id = 'flower_images' AND
    (storage.foldername(name))[1] = auth.uid()::text
  )
  WITH CHECK (
    bucket_id = 'flower_images' AND
    (storage.foldername(name))[1] = auth.uid()::text
  );

-- Allow authenticated users to delete their own images
CREATE POLICY "Users can delete own images"
  ON storage.objects FOR DELETE
  TO authenticated
  USING (
    bucket_id = 'flower_images' AND
    (storage.foldername(name))[1] = auth.uid()::text
  );