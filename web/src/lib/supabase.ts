/**
 * Supabase Client Configuration
 *
 * Handles authentication and database access
 */

import { createClient } from '@supabase/supabase-js'

const supabaseUrl = (process.env.NEXT_PUBLIC_SUPABASE_URL || '').trim()
const supabaseAnonKey = (process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || '').trim()

// Create a no-op client if env vars are missing (cloud/* pages gracefully degrade)
const isConfigured = !!(supabaseUrl && supabaseAnonKey)

export const supabase = isConfigured
  ? createClient(supabaseUrl, supabaseAnonKey)
  : createClient('https://placeholder.supabase.co', 'placeholder-key')

// Server-side Supabase client for API routes — uses service role key to bypass RLS
export const createServerSupabaseClient = () => {
  const serviceKey = (
    process.env.SUPABASE_SERVICE_ROLE_KEY ||
    process.env.SUPABASE_SERVICE_KEY ||
    supabaseAnonKey
  ).trim();
  const url = supabaseUrl || 'https://placeholder.supabase.co';
  return createClient(url, serviceKey || 'placeholder-key');
}

// Database types
export interface User {
  id: string
  email: string
  subscription_tier: 'free' | 'pro' | 'enterprise'
  stripe_customer_id?: string
  created_at: string
  updated_at: string
}

export interface Analysis {
  id: string
  user_id: string
  model_id: string
  fits_file_url?: string
  result: InferenceResult
  classification: string
  confidence: number
  inference_time_ms: number
  created_at: string
}

export interface InferenceResult {
  classification: string
  confidence: number
  probabilities: Record<string, number>
  inference_time_ms: number
  memory_used_kb?: number
}

export interface UsageQuota {
  id: string
  user_id: string
  month: string
  analyses_count: number
  quota_limit: number | null
  created_at: string
}

export interface Subscription {
  id: string
  user_id: string
  stripe_subscription_id?: string
  tier: 'free' | 'pro' | 'enterprise'
  status: 'active' | 'cancelled' | 'past_due'
  current_period_start?: string
  current_period_end?: string
  created_at: string
  updated_at: string
}

// Auth helpers
export const signUp = async (email: string, password: string) => {
  const { data, error } = await supabase.auth.signUp({
    email,
    password,
  })
  return { data, error }
}

export const signIn = async (email: string, password: string) => {
  const { data, error } = await supabase.auth.signInWithPassword({
    email,
    password,
  })
  return { data, error }
}

export const signOut = async () => {
  const { error } = await supabase.auth.signOut()
  return { error }
}

export const getCurrentUser = async () => {
  const { data: { user }, error } = await supabase.auth.getUser()
  return { user, error }
}

// OAuth helpers
export const signInWithGithub = async () => {
  const { data, error } = await supabase.auth.signInWithOAuth({
    provider: 'github',
    options: {
      redirectTo: `${window.location.origin}/dashboard`
    }
  })
  return { data, error }
}

export const signInWithGoogle = async () => {
  const { data, error } = await supabase.auth.signInWithOAuth({
    provider: 'google',
    options: {
      redirectTo: `${window.location.origin}/dashboard`
    }
  })
  return { data, error }
}

// Database helpers
export const getUserProfile = async (userId: string): Promise<User | null> => {
  const { data, error } = await supabase
    .from('users')
    .select('*')
    .eq('id', userId)
    .single()

  if (error) {
    console.error('Error fetching user profile:', error)
    return null
  }

  return data
}

export const getUserAnalyses = async (userId: string): Promise<Analysis[]> => {
  const { data, error } = await supabase
    .from('analyses')
    .select('*')
    .eq('user_id', userId)
    .order('created_at', { ascending: false })

  if (error) {
    console.error('Error fetching analyses:', error)
    return []
  }

  return data
}

export const getUserQuota = async (userId: string): Promise<UsageQuota | null> => {
  // Read quota directly from the users table (analyses_this_month + analyses_limit)
  const { data, error } = await supabase
    .from('users')
    .select('analyses_this_month, analyses_limit')
    .eq('id', userId)
    .single()

  if (error || !data) return null

  return {
    id: userId,
    user_id: userId,
    month: new Date().toISOString().slice(0, 7),
    analyses_count: (data as any).analyses_this_month ?? 0,
    quota_limit: (data as any).analyses_limit ?? 5,
    created_at: new Date().toISOString(),
  }
}

export const getUserSubscription = async (userId: string): Promise<Subscription | null> => {
  const { data, error } = await supabase
    .from('subscriptions')
    .select('*')
    .eq('user_id', userId)
    .eq('status', 'active')
    .order('created_at', { ascending: false })
    .limit(1)
    .maybeSingle()

  if (error) {
    console.error('Error fetching subscription:', error)
    return null
  }

  return data
}

export const createAnalysis = async (analysis: Partial<Analysis>) => {
  const { data, error } = await supabase
    .from('analyses')
    .insert(analysis)
    .select()
    .single()

  return { data, error }
}

export const incrementUsageQuota = async (userId: string) => {
  const currentMonth = new Date().toISOString().slice(0, 7)

  // Upsert (insert or update)
  const { error } = await supabase.rpc('increment_usage', {
    p_user_id: userId,
    p_month: currentMonth
  })

  return { error }
}
