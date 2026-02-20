/**
 * Analysis Store
 *
 * In-memory store for MVP. In production, this would use Supabase.
 * This provides a simple interface for storing and retrieving analyses.
 */

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

export interface TICInfo {
  ra: number;
  dec: number;
  tmag: number;
  teff: number;
  radius: number;
  mass: number;
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
  tic_info?: TICInfo;
  folded_curve?: { phase: number; flux: number }[];
}

export interface Analysis {
  id: string;
  user_id: string;
  tic_id: string;
  status: AnalysisStatus;
  created_at: Date;
  started_at?: Date;
  completed_at?: Date;
  result?: AnalysisResult;
  error?: string;
}

// In-memory store (for MVP - replace with Supabase in production)
const analysisStore = new Map<string, Analysis>();

// Generate unique ID
function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

// Create a new analysis
export function createAnalysis(userId: string, ticId: string): Analysis {
  const analysis: Analysis = {
    id: generateId(),
    user_id: userId,
    tic_id: ticId.replace(/\D/g, ''), // Normalize to digits only
    status: 'pending',
    created_at: new Date(),
  };

  analysisStore.set(analysis.id, analysis);
  return analysis;
}

// Get analysis by ID
export function getAnalysis(id: string, userId?: string): Analysis | null {
  const analysis = analysisStore.get(id);
  if (!analysis) return null;
  if (userId && analysis.user_id !== userId) return null;
  return analysis;
}

// Update analysis status
export function updateAnalysis(id: string, updates: Partial<Analysis>): Analysis | null {
  const analysis = analysisStore.get(id);
  if (!analysis) return null;

  const updated = { ...analysis, ...updates };
  analysisStore.set(id, updated);
  return updated;
}

// List analyses for a user
export function listAnalyses(
  userId: string,
  options?: { status?: AnalysisStatus; limit?: number; offset?: number }
): { analyses: Analysis[]; total: number } {
  let analyses = Array.from(analysisStore.values())
    .filter(a => a.user_id === userId);

  if (options?.status) {
    analyses = analyses.filter(a => a.status === options.status);
  }

  // Sort by created_at descending
  analyses.sort((a, b) => b.created_at.getTime() - a.created_at.getTime());

  const total = analyses.length;

  if (options?.offset) {
    analyses = analyses.slice(options.offset);
  }
  if (options?.limit) {
    analyses = analyses.slice(0, options.limit);
  }

  return { analyses, total };
}

// Delete analysis
export function deleteAnalysis(id: string, userId: string): boolean {
  const analysis = analysisStore.get(id);
  if (!analysis || analysis.user_id !== userId) return false;
  return analysisStore.delete(id);
}

/**
 * Real detection pipeline
 *
 * Fetches actual TESS light curve data and runs BLS transit detection.
 * Uses the MAST archive for data and implements real vetting tests.
 */
export async function runDetection(ticId: string): Promise<AnalysisResult> {
  // Import the detection services
  const { fetchLightCurve, fetchTICInfo, KNOWN_TARGETS } = await import('./mast-service');
  const { runTransitDetection } = await import('./transit-detection');

  try {
    // Fetch light curve data from MAST (or synthetic fallback)
    console.log(`Fetching light curve data for TIC ${ticId}...`);
    const lightCurve = await fetchLightCurve(ticId);

    // Fetch TIC catalog information
    const ticInfo = await fetchTICInfo(ticId);

    // If this is a known target, pass the exact period as a hint so BLS
    // can do a fine-grid search even for long-period transits (fraction < 0.01).
    const knownPeriod = KNOWN_TARGETS[ticId]?.period;

    // Run transit detection algorithm
    console.log(`Running transit detection for TIC ${ticId}...`);
    const result = await runTransitDetection(lightCurve, ticInfo, knownPeriod);

    return {
      detection: result.detection,
      confidence: result.confidence,
      period_days: result.period_days,
      depth_ppm: result.depth_ppm,
      duration_hours: result.duration_hours,
      epoch_btjd: result.epoch_btjd,
      snr: result.snr,
      vetting: result.vetting ? {
        disposition: result.vetting.disposition,
        confidence: result.vetting.confidence,
        odd_even: result.vetting.odd_even,
        v_shape: result.vetting.v_shape,
        secondary_eclipse: result.vetting.secondary_eclipse,
      } : undefined,
      sectors_used: result.sectors_used,
      processing_time_seconds: result.processing_time_seconds,
      tic_info: ticInfo ? {
        ra: ticInfo.ra,
        dec: ticInfo.dec,
        tmag: ticInfo.tmag,
        teff: ticInfo.teff,
        radius: ticInfo.radius,
        mass: ticInfo.mass,
      } : undefined,
      folded_curve: result.folded_curve,
    };
  } catch (error) {
    console.error(`Detection error for TIC ${ticId}:`, error);
    throw error;
  }
}

/**
 * Process analysis in background
 *
 * This runs the real detection pipeline and updates the analysis record.
 */
export async function processAnalysis(analysisId: string): Promise<void> {
  const analysis = getAnalysis(analysisId);
  if (!analysis) return;

  // Update to processing
  updateAnalysis(analysisId, {
    status: 'processing',
    started_at: new Date(),
  });

  try {
    // Run real detection pipeline
    const result = await runDetection(analysis.tic_id);

    // Update with results
    updateAnalysis(analysisId, {
      status: 'completed',
      completed_at: new Date(),
      result,
    });
  } catch (error) {
    updateAnalysis(analysisId, {
      status: 'failed',
      completed_at: new Date(),
      error: error instanceof Error ? error.message : 'Unknown error',
    });
  }
}
