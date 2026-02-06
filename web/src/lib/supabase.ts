import { createClient, SupabaseClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || '';
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || '';

// Only create client if env vars are set (allows build without them)
export const supabase: SupabaseClient | null =
  supabaseUrl && supabaseAnonKey
    ? createClient(supabaseUrl, supabaseAnonKey)
    : null;

// Server-side client with service role key (for admin operations)
export const createServerSupabaseClient = () => {
  const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

  if (!supabaseUrl || !supabaseServiceKey) {
    console.warn('Supabase environment variables not configured');
    // Return a mock client with chainable builder pattern (for build/dev without Supabase)
    const mockQueryBuilder = (table: string) => {
      const builder: Record<string, unknown> = {
        select: (..._args: unknown[]) => builder,
        insert: async (data: unknown) => {
          console.log(`[Supabase Mock] Insert into ${table}:`, data);
          return { data: null, error: null };
        },
        update: (data: unknown) => {
          console.log(`[Supabase Mock] Update ${table}:`, data);
          return builder;
        },
        upsert: async (data: unknown) => {
          console.log(`[Supabase Mock] Upsert into ${table}:`, data);
          return { data: null, error: null };
        },
        delete: () => builder,
        eq: (_column: string, _value: unknown) => builder,
        neq: (_column: string, _value: unknown) => builder,
        gt: (_column: string, _value: unknown) => builder,
        lt: (_column: string, _value: unknown) => builder,
        gte: (_column: string, _value: unknown) => builder,
        lte: (_column: string, _value: unknown) => builder,
        like: (_column: string, _value: unknown) => builder,
        in: (_column: string, _values: unknown[]) => builder,
        order: (_column: string, _options?: unknown) => builder,
        limit: (_count: number) => builder,
        range: (_from: number, _to: number) => builder,
        single: async () => ({ data: null, error: null }),
        maybeSingle: async () => ({ data: null, error: null }),
        then: (resolve: (value: { data: null; error: null; count: null }) => void) =>
          Promise.resolve({ data: null, error: null, count: null }).then(resolve),
      };
      return builder;
    };

    return {
      from: (table: string) => mockQueryBuilder(table),
    } as unknown as SupabaseClient;
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
