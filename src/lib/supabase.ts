import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error('Missing Supabase environment variables');
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

export type Database = {
  public: {
    Tables: {
      uploads: {
        Row: {
          id: string;
          user_id: string;
          created_at: string;
          image_path: string;
          status: 'processing' | 'completed' | 'failed';
        };
        Insert: {
          id?: string;
          user_id: string;
          created_at?: string;
          image_path: string;
          status?: 'processing' | 'completed' | 'failed';
        };
        Update: {
          id?: string;
          user_id?: string;
          created_at?: string;
          image_path?: string;
          status?: 'processing' | 'completed' | 'failed';
        };
      };
      predictions: {
        Row: {
          id: string;
          upload_id: string;
          predicted_class: string;
          confidence_score: number;
          is_top_prediction: boolean;
          created_at: string;
        };
        Insert: {
          id?: string;
          upload_id: string;
          predicted_class: string;
          confidence_score: number;
          is_top_prediction?: boolean;
          created_at?: string;
        };
        Update: {
          id?: string;
          upload_id?: string;
          predicted_class?: string;
          confidence_score?: number;
          is_top_prediction?: boolean;
          created_at?: string;
        };
      };
    };
  };
};
