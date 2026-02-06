/**
 * Analysis Database Service
 *
 * Supabase-based storage for analyses that works in serverless environments.
 * Falls back to in-memory store when Supabase is unavailable, so analyses
 * still run without a database (results won't persist across restarts).
 */

import { createServerSupabaseClient } from './supabase';

export type AnalysisStatus = 'pending' | 'processing' | 'completed' | 'failed';

export interface VettingTest {
  flag: 'PASS' | 'FAIL' | 'WARNING';
  message: string;
  confidence?: number;
}

export interface VettingResult {
  disposition: 'PLANET_CANDIDATE' | 'LIKELY_FALSE_POSITIVE' | 'INCONCLUSIVE';
  confidence: number;
  odd_even: VettingTest;
  v_shape: VettingTest;
  secondary_eclipse: VettingTest;
}

export interface AnalysisResult {
  detection: boolean;
  confidence: number;
  period_days: number | null;
  depth_ppm: number | null;
  duration_hours: number | null;
  epoch_btjd?: number | null;
  snr?: number | null;
  vetting?: VettingResult;
  sectors_used?: number[];
  processing_time_seconds?: number;
}

export interface Analysis {
  id: string;
  user_id: string;
  tic_id: string;
  status: AnalysisStatus;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  result?: AnalysisResult;
  error?: string;
}

// In-memory fallback store for when Supabase is unavailable
const memoryStore = new Map<string, Analysis>();

function isSupabaseAvailable(): boolean {
  return !!(process.env.NEXT_PUBLIC_SUPABASE_URL && process.env.SUPABASE_SERVICE_ROLE_KEY);
}

/**
 * Create a new analysis record.
 * Tries Supabase first, falls back to in-memory store.
 */
export async function createAnalysisInDB(
  userId: string,
  userEmail: string,
  ticId: string
): Promise<Analysis> {
  const id = `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

  const analysis: Analysis = {
    id,
    user_id: userId,
    tic_id: ticId.replace(/\D/g, ''),
    status: 'pending',
    created_at: new Date().toISOString(),
  };

  if (isSupabaseAvailable()) {
    try {
      const supabase = createServerSupabaseClient();
      const { error } = await supabase.from('analyses').insert({
        id: analysis.id,
        user_id: analysis.user_id,
        user_email: userEmail,
        tic_id: analysis.tic_id,
        status: analysis.status,
        created_at: analysis.created_at,
      });

      if (error) {
        console.error('Supabase insert error, using memory store:', error);
        memoryStore.set(id, analysis);
      }
    } catch (err) {
      console.error('Supabase unavailable, using memory store:', err);
      memoryStore.set(id, analysis);
    }
  } else {
    memoryStore.set(id, analysis);
  }

  return analysis;
}

/**
 * Get analysis by ID.
 * Checks Supabase first, then in-memory store.
 */
export async function getAnalysisFromDB(
  analysisId: string,
  userId?: string
): Promise<Analysis | null> {
  // Try Supabase first
  if (isSupabaseAvailable()) {
    try {
      const supabase = createServerSupabaseClient();
      let query = supabase
        .from('analyses')
        .select('*')
        .eq('id', analysisId);

      if (userId) {
        query = query.eq('user_id', userId);
      }

      const { data, error } = await query.single();

      if (!error && data) {
        return {
          id: data.id,
          user_id: data.user_id,
          tic_id: data.tic_id,
          status: data.status,
          created_at: data.created_at,
          started_at: data.started_at,
          completed_at: data.completed_at,
          result: data.result as AnalysisResult | undefined,
          error: data.error_message,
        };
      }
    } catch (err) {
      console.error('Supabase read error, checking memory store:', err);
    }
  }

  // Fall back to memory store
  const analysis = memoryStore.get(analysisId);
  if (analysis && (!userId || analysis.user_id === userId)) {
    return analysis;
  }

  return null;
}

/**
 * Update analysis record.
 * Tries Supabase first, falls back to in-memory store.
 */
export async function updateAnalysisInDB(
  analysisId: string,
  updates: {
    status?: AnalysisStatus;
    started_at?: string;
    completed_at?: string;
    result?: AnalysisResult;
    error?: string;
  }
): Promise<void> {
  // Always update memory store if entry exists
  const memEntry = memoryStore.get(analysisId);
  if (memEntry) {
    if (updates.status) memEntry.status = updates.status;
    if (updates.started_at) memEntry.started_at = updates.started_at;
    if (updates.completed_at) memEntry.completed_at = updates.completed_at;
    if (updates.result) memEntry.result = updates.result;
    if (updates.error) memEntry.error = updates.error;
    memoryStore.set(analysisId, memEntry);
  }

  // Try Supabase
  if (isSupabaseAvailable()) {
    try {
      const supabase = createServerSupabaseClient();
      const dbUpdates: Record<string, unknown> = {};

      if (updates.status) dbUpdates.status = updates.status;
      if (updates.started_at) dbUpdates.started_at = updates.started_at;
      if (updates.completed_at) dbUpdates.completed_at = updates.completed_at;
      if (updates.result) dbUpdates.result = updates.result;
      if (updates.error) dbUpdates.error_message = updates.error;

      const { error } = await supabase
        .from('analyses')
        .update(dbUpdates)
        .eq('id', analysisId);

      if (error) {
        console.error('Supabase update error (memory store used):', error);
      }
    } catch (err) {
      console.error('Supabase unavailable for update (memory store used):', err);
    }
  }
}

/**
 * List analyses for a user.
 * Merges Supabase results with in-memory entries.
 */
export async function listAnalysesFromDB(
  userId: string,
  options?: { status?: AnalysisStatus; limit?: number; offset?: number }
): Promise<{ analyses: Analysis[]; total: number }> {
  let dbAnalyses: Analysis[] = [];

  // Try Supabase
  if (isSupabaseAvailable()) {
    try {
      const supabase = createServerSupabaseClient();
      let query = supabase
        .from('analyses')
        .select('*', { count: 'exact' })
        .eq('user_id', userId)
        .order('created_at', { ascending: false });

      if (options?.status) {
        query = query.eq('status', options.status);
      }

      if (options?.limit) {
        query = query.limit(options.limit);
      }

      if (options?.offset) {
        query = query.range(options.offset, options.offset + (options.limit || 10) - 1);
      }

      const { data, error } = await query;

      if (!error && data) {
        dbAnalyses = data.map((row) => ({
          id: row.id,
          user_id: row.user_id,
          tic_id: row.tic_id,
          status: row.status,
          created_at: row.created_at,
          started_at: row.started_at,
          completed_at: row.completed_at,
          result: row.result as AnalysisResult | undefined,
          error: row.error_message,
        }));
      }
    } catch (err) {
      console.error('Supabase list error, using memory store only:', err);
    }
  }

  // Merge in-memory analyses for this user (deduplicating by ID)
  const dbIds = new Set(dbAnalyses.map(a => a.id));
  const memAnalyses = Array.from(memoryStore.values())
    .filter(a => a.user_id === userId && !dbIds.has(a.id))
    .filter(a => !options?.status || a.status === options.status);

  const allAnalyses = [...dbAnalyses, ...memAnalyses]
    .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());

  const offset = options?.offset || 0;
  const limit = options?.limit || allAnalyses.length;
  const paged = allAnalyses.slice(offset, offset + limit);

  return { analyses: paged, total: allAnalyses.length };
}

/**
 * Run detection and update analysis in DB
 */
export async function runAnalysisWithDB(analysisId: string): Promise<AnalysisResult> {
  // Import detection services dynamically
  const { fetchLightCurve, fetchTICInfo } = await import('./mast-service');
  const { runTransitDetection } = await import('./transit-detection');

  // Get analysis record
  const analysis = await getAnalysisFromDB(analysisId);
  if (!analysis) {
    throw new Error('Analysis not found');
  }

  // Update to processing
  await updateAnalysisInDB(analysisId, {
    status: 'processing',
    started_at: new Date().toISOString(),
  });

  try {
    // Fetch light curve data
    console.log(`Fetching light curve for TIC ${analysis.tic_id}...`);
    const lightCurve = await fetchLightCurve(analysis.tic_id);

    // Fetch TIC info
    const ticInfo = await fetchTICInfo(analysis.tic_id);

    // Run detection
    console.log(`Running detection for TIC ${analysis.tic_id}...`);
    const result = await runTransitDetection(lightCurve, ticInfo);

    // Build the analysis result
    const analysisResult: AnalysisResult = {
      detection: result.detection,
      confidence: result.confidence,
      period_days: result.period_days,
      depth_ppm: result.depth_ppm,
      duration_hours: result.duration_hours,
      epoch_btjd: result.epoch_btjd,
      snr: result.snr,
      vetting: result.vetting || undefined,
      sectors_used: result.sectors_used,
      processing_time_seconds: result.processing_time_seconds,
    };

    // Update with results
    await updateAnalysisInDB(analysisId, {
      status: 'completed',
      completed_at: new Date().toISOString(),
      result: analysisResult,
    });

    return analysisResult;
  } catch (error) {
    console.error(`Detection error for analysis ${analysisId}:`, error);

    await updateAnalysisInDB(analysisId, {
      status: 'failed',
      completed_at: new Date().toISOString(),
      error: error instanceof Error ? error.message : 'Unknown error',
    });

    throw error;
  }
}

/**
 * Increment user's analysis count.
 * Best-effort — never throws. Count is tracked in JWT session as fallback.
 */
export async function incrementUserAnalysisCount(userEmail: string): Promise<void> {
  if (!isSupabaseAvailable()) return;

  try {
    const supabase = createServerSupabaseClient();
    const { data: user } = await supabase
      .from('users')
      .select('analyses_this_month')
      .eq('email', userEmail)
      .single();

    if (user) {
      const newCount = (user.analyses_this_month || 0) + 1;
      await supabase
        .from('users')
        .update({ analyses_this_month: newCount })
        .eq('email', userEmail);
    }
  } catch (err) {
    console.error('Failed to increment analysis count (non-fatal):', err);
  }
}
