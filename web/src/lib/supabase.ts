import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;

if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error('Missing Supabase environment variables');
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

// Server-side client with service role key (for admin operations)
export const createServerSupabaseClient = () => {
  const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!supabaseServiceKey) {
    throw new Error('Missing SUPABASE_SERVICE_ROLE_KEY');
  }
  return createClient(supabaseUrl, supabaseServiceKey, {
    auth: {
      autoRefreshToken: false,
      persistSession: false,
    },
  });
};

// Types for Supabase database
export interface Database {
  public: {
    Tables: {
      users: {
        Row: {
          id: string;
          email: string;
          name: string | null;
          image: string | null;
          subscription_tier: 'free' | 'monthly' | 'annual';
          lemon_squeezy_customer_id: string | null;
          analyses_this_month: number;
          analyses_limit: number;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          email: string;
          name?: string | null;
          image?: string | null;
          subscription_tier?: 'free' | 'monthly' | 'annual';
          lemon_squeezy_customer_id?: string | null;
          analyses_this_month?: number;
          analyses_limit?: number;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          email?: string;
          name?: string | null;
          image?: string | null;
          subscription_tier?: 'free' | 'monthly' | 'annual';
          lemon_squeezy_customer_id?: string | null;
          analyses_this_month?: number;
          analyses_limit?: number;
          created_at?: string;
          updated_at?: string;
        };
      };
      analyses: {
        Row: {
          id: string;
          user_id: string;
          tic_id: string;
          status: 'pending' | 'processing' | 'completed' | 'failed';
          result: Record<string, unknown> | null;
          error_message: string | null;
          created_at: string;
          completed_at: string | null;
        };
        Insert: {
          id?: string;
          user_id: string;
          tic_id: string;
          status?: 'pending' | 'processing' | 'completed' | 'failed';
          result?: Record<string, unknown> | null;
          error_message?: string | null;
          created_at?: string;
          completed_at?: string | null;
        };
        Update: {
          id?: string;
          user_id?: string;
          tic_id?: string;
          status?: 'pending' | 'processing' | 'completed' | 'failed';
          result?: Record<string, unknown> | null;
          error_message?: string | null;
          created_at?: string;
          completed_at?: string | null;
        };
      };
      subscriptions: {
        Row: {
          id: string;
          user_id: string;
          lemon_squeezy_subscription_id: string;
          lemon_squeezy_customer_id: string;
          variant_id: string;
          plan: 'monthly' | 'annual';
          status: 'active' | 'cancelled' | 'expired' | 'past_due' | 'paused' | 'on_trial';
          current_period_start: string;
          current_period_end: string;
          cancel_at_period_end: boolean;
          created_at: string;
          updated_at: string;
        };
        Insert: {
          id?: string;
          user_id: string;
          lemon_squeezy_subscription_id: string;
          lemon_squeezy_customer_id: string;
          variant_id: string;
          plan: 'monthly' | 'annual';
          status?: 'active' | 'cancelled' | 'expired' | 'past_due' | 'paused' | 'on_trial';
          current_period_start: string;
          current_period_end: string;
          cancel_at_period_end?: boolean;
          created_at?: string;
          updated_at?: string;
        };
        Update: {
          id?: string;
          user_id?: string;
          lemon_squeezy_subscription_id?: string;
          lemon_squeezy_customer_id?: string;
          variant_id?: string;
          plan?: 'monthly' | 'annual';
          status?: 'active' | 'cancelled' | 'expired' | 'past_due' | 'paused' | 'on_trial';
          current_period_start?: string;
          current_period_end?: string;
          cancel_at_period_end?: boolean;
          created_at?: string;
          updated_at?: string;
        };
      };
    };
  };
}
