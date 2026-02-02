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
 * Mock detection pipeline
 *
 * This simulates what the real Python detection service would return.
 * In production, this would call the actual detection API or
 * integrate with the Python backend.
 */
export async function runMockDetection(ticId: string): Promise<AnalysisResult> {
  // Simulate processing time
  await new Promise(resolve => setTimeout(resolve, 2000 + Math.random() * 3000));

  // Known targets with real-ish data (for demo purposes)
  const knownTargets: Record<string, AnalysisResult> = {
    '470710327': {
      detection: true,
      confidence: 0.92,
      period_days: 3.4252,
      depth_ppm: 850,
      duration_hours: 2.8,
      epoch_btjd: 1326.45,
      snr: 15.3,
      vetting: {
        disposition: 'PLANET_CANDIDATE',
        confidence: 0.88,
        odd_even: { flag: 'PASS', message: 'Odd/even depth difference: 0.8 sigma', confidence: 0.95 },
        v_shape: { flag: 'PASS', message: 'Transit shape consistent with planetary', confidence: 0.91 },
        secondary_eclipse: { flag: 'PASS', message: 'No secondary eclipse detected', confidence: 0.89 },
      },
      sectors_used: [1, 2, 3],
      processing_time_seconds: 45.2,
    },
    '307210830': {
      detection: true,
      confidence: 0.78,
      period_days: 8.138,
      depth_ppm: 1200,
      duration_hours: 3.5,
      epoch_btjd: 1355.12,
      snr: 11.8,
      vetting: {
        disposition: 'PLANET_CANDIDATE',
        confidence: 0.72,
        odd_even: { flag: 'PASS', message: 'Odd/even depth difference: 1.2 sigma', confidence: 0.88 },
        v_shape: { flag: 'WARNING', message: 'Slightly V-shaped transit', confidence: 0.65 },
        secondary_eclipse: { flag: 'PASS', message: 'No secondary eclipse detected', confidence: 0.85 },
      },
      sectors_used: [5, 6],
      processing_time_seconds: 38.7,
    },
    '261136679': {
      detection: true,
      confidence: 0.95,
      period_days: 2.7972,
      depth_ppm: 650,
      duration_hours: 2.2,
      epoch_btjd: 1312.78,
      snr: 22.1,
      vetting: {
        disposition: 'PLANET_CANDIDATE',
        confidence: 0.94,
        odd_even: { flag: 'PASS', message: 'Odd/even depth difference: 0.3 sigma', confidence: 0.98 },
        v_shape: { flag: 'PASS', message: 'Transit shape consistent with planetary', confidence: 0.96 },
        secondary_eclipse: { flag: 'PASS', message: 'No secondary eclipse detected', confidence: 0.92 },
      },
      sectors_used: [1, 2, 3, 4],
      processing_time_seconds: 52.1,
    },
  };

  // Check if it's a known target
  if (knownTargets[ticId]) {
    return knownTargets[ticId];
  }

  // Generate random result for unknown targets
  const hasDetection = Math.random() > 0.6; // 40% chance of detection

  if (!hasDetection) {
    return {
      detection: false,
      confidence: 0.1 + Math.random() * 0.3,
      period_days: null,
      depth_ppm: null,
      duration_hours: null,
      processing_time_seconds: 25 + Math.random() * 20,
    };
  }

  // Random detection
  const snr = 7 + Math.random() * 15;
  const confidence = Math.min(0.95, 0.5 + (snr / 30));
  const period = 0.5 + Math.random() * 20;
  const depth = 200 + Math.random() * 2000;

  const vettingPassed = Math.random() > 0.3;

  return {
    detection: true,
    confidence,
    period_days: Number(period.toFixed(4)),
    depth_ppm: Number(depth.toFixed(0)),
    duration_hours: Number((1 + Math.random() * 5).toFixed(2)),
    epoch_btjd: Number((1300 + Math.random() * 100).toFixed(2)),
    snr: Number(snr.toFixed(1)),
    vetting: {
      disposition: vettingPassed ? 'PLANET_CANDIDATE' : 'INCONCLUSIVE',
      confidence: vettingPassed ? 0.7 + Math.random() * 0.25 : 0.3 + Math.random() * 0.3,
      odd_even: {
        flag: Math.random() > 0.2 ? 'PASS' : 'WARNING',
        message: `Odd/even depth difference: ${(Math.random() * 3).toFixed(1)} sigma`,
        confidence: 0.6 + Math.random() * 0.35,
      },
      v_shape: {
        flag: Math.random() > 0.3 ? 'PASS' : 'WARNING',
        message: Math.random() > 0.3 ? 'Transit shape consistent with planetary' : 'Slightly V-shaped transit',
        confidence: 0.5 + Math.random() * 0.45,
      },
      secondary_eclipse: {
        flag: Math.random() > 0.1 ? 'PASS' : 'FAIL',
        message: Math.random() > 0.1 ? 'No secondary eclipse detected' : 'Possible secondary eclipse at phase 0.5',
        confidence: 0.7 + Math.random() * 0.25,
      },
    },
    sectors_used: [1, 2].filter(() => Math.random() > 0.3),
    processing_time_seconds: Number((25 + Math.random() * 35).toFixed(1)),
  };
}

/**
 * Process analysis in background
 *
 * This runs the mock detection and updates the analysis record.
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
    // Run detection
    const result = await runMockDetection(analysis.tic_id);

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
