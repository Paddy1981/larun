/**
 * Supabase Client Configuration
 *
 * Handles authentication and database access
 */

import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!

if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error('Missing Supabase environment variables')
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey)

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
  const currentMonth = new Date().toISOString().slice(0, 7) // '2026-02'

  const { data, error } = await supabase
    .from('usage_quotas')
    .select('*')
    .eq('user_id', userId)
    .eq('month', currentMonth)
    .single()

  if (error && error.code !== 'PGRST116') { // PGRST116 = no rows
    console.error('Error fetching quota:', error)
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
