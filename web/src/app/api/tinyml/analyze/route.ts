import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';
import { parseFITSLightCurve } from '@/lib/mast-service';
import { hashApiKey, isValidKeyFormat } from '@/lib/api-key-utils';
import type { InferenceResult } from '@/lib/supabase';

export const maxDuration = 30;

const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL || '';
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || '';

/**
 * Validate an API key from the request headers.
 * Returns the key record if valid, null if no API key present,
 * or throws a Response if the key is invalid / rate-limited.
 */
async function validateApiKey(request: NextRequest): Promise<
  | { type: 'web' }               // no API key header — web-UI flow
  | { type: 'api'; keyId: string; userId: string }  // valid API key
> {
  const header =
    request.headers.get('X-API-Key') ||
    request.headers.get('Authorization')?.replace(/^Bearer /, '') ||
    null;

  if (!header || !isValidKeyFormat(header)) {
    return { type: 'web' };
  }

  if (!SUPABASE_URL || !SUPABASE_SERVICE_KEY) {
    return { type: 'web' }; // graceful degradation if Supabase not configured
  }

  const hash = hashApiKey(header);
  const service = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);

  // Try full column set first; fall back to base schema if migration pending
  let keyRes = await service
    .from('api_keys')
    .select('id, user_id, is_active, calls_this_month, calls_limit')
    .eq('key_hash', hash)
    .single();

  let keyFull = true;
  if (keyRes.error?.code === 'PGRST204') {
    keyFull = false;
    keyRes = await service
      .from('api_keys')
      .select('id, user_id')
      .eq('key_hash', hash)
      .single() as typeof keyRes;
  }

  if (keyRes.error || !keyRes.data) {
    throw NextResponse.json({ detail: 'Invalid API key.' }, { status: 401 });
  }

  const key = keyRes.data as {
    id: string; user_id: string;
    is_active?: boolean; calls_this_month?: number; calls_limit?: number;
  };

  if (keyFull && key.is_active === false) {
    throw NextResponse.json({ detail: 'API key has been revoked.' }, { status: 401 });
  }
  const callsLimit = key.calls_limit ?? 10000;
  const callsUsed = key.calls_this_month ?? 0;
  if (callsLimit !== -1 && callsUsed >= callsLimit) {
    throw NextResponse.json(
      {
        detail: `Monthly API limit reached (${callsLimit} calls). Upgrade to Enterprise for unlimited access.`,
        calls_used: callsUsed,
        calls_limit: callsLimit,
      },
      { status: 429 }
    );
  }

  // Increment usage counter — fire-and-forget (non-blocking)
  const updatePayload: Record<string, unknown> = { last_used_at: new Date().toISOString() };
  if (keyFull) updatePayload.calls_this_month = callsUsed + 1;
  service
    .from('api_keys')
    .update(updatePayload)
    .eq('id', key.id)
    .then(() => {/* ignore */});

  return { type: 'api', keyId: key.id, userId: key.user_id };
}

// ─── Feature extraction ──────────────────────────────────────────────────────

interface LCFeatures {
  n: number;
  mean: number;
  std: number;
  median: number;
  skewness: number;
  kurtosis: number;
  peak_to_trough: number;
  p5: number;
  p95: number;
  /** Median-normalised depth: (median - p5) / median */
  depth_ratio: number;
  /** Fraction of points more than 3σ from mean */
  outlier_frac: number;
  /** Rough inter-quartile-range / σ  (≈1.35 for Gaussian) */
  variability: number;
  /** Autocorrelation-based periodicity 0-1 */
  periodicity: number;
  /** Dominant lag in time units (0 if no periodicity) */
  period_est: number;
}

function extractFeatures(time: number[], flux: number[]): LCFeatures {
  const n = flux.length;
  const sorted = [...flux].sort((a, b) => a - b);

  const mean = flux.reduce((s, v) => s + v, 0) / n;
  const variance = flux.reduce((s, v) => s + (v - mean) ** 2, 0) / n;
  const std = Math.sqrt(Math.max(variance, 1e-12));
  const median = sorted[Math.floor(n / 2)];
  const p5 = sorted[Math.max(0, Math.floor(n * 0.05))];
  const p95 = sorted[Math.min(n - 1, Math.floor(n * 0.95))];
  const q25 = sorted[Math.floor(n * 0.25)];
  const q75 = sorted[Math.floor(n * 0.75)];

  const skewness = flux.reduce((s, v) => s + ((v - mean) / std) ** 3, 0) / n;
  const kurtosis = flux.reduce((s, v) => s + ((v - mean) / std) ** 4, 0) / n - 3;
  const outlier_frac = flux.filter(v => Math.abs(v - mean) > 3 * std).length / n;
  const variability = (q75 - q25) / std;

  // Autocorrelation periodicity (check lags up to n/4, step by 1)
  const maxLag = Math.min(200, Math.floor(n / 4));
  let bestCorr = 0;
  let bestLag = 0;
  for (let lag = 1; lag <= maxLag; lag++) {
    let corr = 0;
    for (let i = 0; i < n - lag; i++) {
      corr += (flux[i] - mean) * (flux[i + lag] - mean);
    }
    corr /= (n - lag) * variance;
    if (corr > bestCorr) { bestCorr = corr; bestLag = lag; }
  }
  const dt = n > 1 ? (time[n - 1] - time[0]) / (n - 1) : 1;

  return {
    n,
    mean,
    std,
    median,
    skewness,
    kurtosis,
    peak_to_trough: sorted[n - 1] - sorted[0],
    p5,
    p95,
    depth_ratio: median > 0 ? (median - p5) / median : 0,
    outlier_frac,
    variability,
    periodicity: Math.max(0, bestCorr),
    period_est: bestCorr > 0.25 ? bestLag * dt : 0,
  };
}

// ─── Per-model classifiers ────────────────────────────────────────────────────

type Probs = Record<string, number>;

function softmax(raw: Probs): Probs {
  const keys = Object.keys(raw);
  const vals = keys.map(k => raw[k]);
  const maxV = Math.max(...vals);
  const exps = vals.map(v => Math.exp(v - maxV));
  const sum = exps.reduce((s, e) => s + e, 0);
  const out: Probs = {};
  keys.forEach((k, i) => { out[k] = exps[i] / sum; });
  return out;
}

function topClass(probs: Probs): string {
  return Object.entries(probs).reduce((b, [k, v]) => v > probs[b] ? k : b, Object.keys(probs)[0]);
}

function classifyExoplanet(f: LCFeatures): { classification: string; probabilities: Probs } {
  const raw: Probs = {
    noise: 0,
    stellar_signal: 0,
    planetary_transit: 0,
    eclipsing_binary: 0,
    instrument_artifact: 0,
    unknown_anomaly: 0,
  };

  // Transit evidence: negative skew (dips), depth_ratio > threshold, some periodicity
  const hasNegativeSkew = f.skewness < -0.3;
  const deepDip = f.depth_ratio > 0.003;
  const veryDeepDip = f.depth_ratio > 0.02;   // >2% → likely EB
  const hasPeriodicity = f.periodicity > 0.3;
  const highVariability = f.variability > 2.0;

  if (veryDeepDip && hasNegativeSkew) {
    // Deep periodic dips → eclipsing binary
    raw.eclipsing_binary = 3.0;
    raw.planetary_transit = 1.0;
  } else if (deepDip && hasNegativeSkew && hasPeriodicity) {
    // Moderate depth, periodic, box-shaped → planet transit
    raw.planetary_transit = 3.0;
    raw.eclipsing_binary = 0.8;
  } else if (deepDip && hasNegativeSkew) {
    // Dip but not periodic → maybe transit, maybe artifact
    raw.planetary_transit = 2.0;
    raw.instrument_artifact = 0.5;
  } else if (highVariability && !hasNegativeSkew) {
    raw.stellar_signal = 2.5;
    raw.eclipsing_binary = 0.5;
  } else if (f.variability < 1.2 && !deepDip) {
    raw.noise = 3.0;
  } else {
    raw.stellar_signal = 1.5;
    raw.unknown_anomaly = 0.3;
  }

  const probs = softmax(raw);
  return { classification: topClass(probs), probabilities: probs };
}

function classifyVariableStar(f: LCFeatures): { classification: string; probabilities: Probs } {
  const raw: Probs = {
    cepheid: 0, rr_lyrae: 0, delta_scuti: 0,
    eclipsing_binary: 0, rotational: 0, irregular: 0, constant: 0,
  };

  if (f.variability < 1.2 && f.periodicity < 0.2) {
    raw.constant = 4.0;
  } else if (f.periodicity > 0.5 && f.period_est > 1 && f.peak_to_trough > 0.05) {
    raw.cepheid = 3.0; raw.rotational = 0.5;
  } else if (f.periodicity > 0.5 && f.period_est < 1 && f.peak_to_trough > 0.15) {
    raw.rr_lyrae = 3.0; raw.delta_scuti = 0.5;
  } else if (f.periodicity > 0.5 && f.period_est < 0.3) {
    raw.delta_scuti = 3.0;
  } else if (f.variability > 2.5 && f.skewness < -0.5) {
    raw.eclipsing_binary = 2.5; raw.irregular = 0.5;
  } else if (f.variability > 1.5 && f.periodicity > 0.25) {
    raw.rotational = 2.5; raw.cepheid = 0.3;
  } else {
    raw.irregular = 2.0; raw.rotational = 0.5;
  }

  const probs = softmax(raw);
  return { classification: topClass(probs), probabilities: probs };
}

function classifyFlare(f: LCFeatures): { classification: string; probabilities: Probs } {
  const raw: Probs = {
    no_flare: 0, weak_flare: 0, moderate_flare: 0, strong_flare: 0, superflare: 0,
  };

  // Flares → positive skew (brief bright spike), excess kurtosis, outliers
  const flareScore = Math.max(0, f.skewness) * 0.6
    + Math.max(0, f.kurtosis) * 0.2
    + f.outlier_frac * 4
    + f.peak_to_trough * 8;

  if (flareScore < 0.5) {
    raw.no_flare = 4.0;
  } else if (flareScore < 1.5) {
    raw.no_flare = 1.0; raw.weak_flare = 3.0;
  } else if (flareScore < 3.0) {
    raw.weak_flare = 1.0; raw.moderate_flare = 3.0;
  } else if (flareScore < 5.0) {
    raw.moderate_flare = 1.0; raw.strong_flare = 3.0;
  } else {
    raw.strong_flare = 1.0; raw.superflare = 3.0;
  }

  const probs = softmax(raw);
  return { classification: topClass(probs), probabilities: probs };
}

function classifyAstero(f: LCFeatures): { classification: string; probabilities: Probs } {
  const raw: Probs = {
    solar_like: 0, classical_pulsator: 0, hybrid: 0, non_pulsating: 0,
  };

  if (f.periodicity > 0.5 && f.variability > 2.5 && f.period_est < 0.5) {
    raw.classical_pulsator = 3.5; raw.hybrid = 0.5;
  } else if (f.periodicity > 0.3 && f.variability > 1.5) {
    raw.hybrid = 2.5; raw.solar_like = 0.8;
  } else if (f.variability > 1.0 && f.periodicity < 0.3) {
    raw.solar_like = 2.5; raw.non_pulsating = 0.5;
  } else {
    raw.non_pulsating = 3.0; raw.solar_like = 0.5;
  }

  const probs = softmax(raw);
  return { classification: topClass(probs), probabilities: probs };
}

function classifySupernova(f: LCFeatures): { classification: string; probabilities: Probs } {
  const raw: Probs = {
    no_transient: 0, supernova_ia: 0, supernova_ii: 0, tde: 0, agn: 0,
  };

  const transientScore = Math.max(0, f.skewness) + f.peak_to_trough * 4;

  if (transientScore < 0.5) {
    raw.no_transient = 4.0;
  } else if (f.skewness > 1.5 && f.peak_to_trough > 0.1) {
    raw.supernova_ia = 2.0; raw.supernova_ii = 1.5; raw.no_transient = 0.2;
  } else if (f.variability > 3.0) {
    raw.agn = 2.0; raw.tde = 1.5; raw.no_transient = 0.5;
  } else {
    raw.no_transient = 1.5; raw.supernova_ii = 1.0; raw.agn = 0.8;
  }

  const probs = softmax(raw);
  return { classification: topClass(probs), probabilities: probs };
}

function classifyGalaxy(f: LCFeatures): { classification: string; probabilities: Probs } {
  const raw: Probs = {
    elliptical: 0, spiral: 0, irregular: 0, merger: 0,
  };

  if (f.variability < 1.0) {
    raw.elliptical = 3.5; raw.spiral = 0.5;
  } else if (f.variability > 3.0 || Math.abs(f.skewness) > 2.0) {
    raw.merger = 2.5; raw.irregular = 1.5;
  } else if (f.variability > 1.5) {
    raw.spiral = 2.5; raw.irregular = 1.0;
  } else {
    raw.spiral = 2.0; raw.elliptical = 1.0; raw.irregular = 0.5;
  }

  const probs = softmax(raw);
  return { classification: topClass(probs), probabilities: probs };
}

function classifySpectralType(f: LCFeatures): { classification: string; probabilities: Probs } {
  const raw: Probs = { O: 0, B: 0, A: 0, F: 0, G: 0, K: 0, M: 0 };

  // Hotter stars (O, B, A): lower variability; cooler stars (K, M): higher variability
  const vs = f.variability;
  if (vs < 0.8) {
    raw.O = 2.0; raw.B = 2.5; raw.A = 1.0;
  } else if (vs < 1.2) {
    raw.B = 1.0; raw.A = 2.5; raw.F = 1.5;
  } else if (vs < 1.8) {
    raw.A = 0.5; raw.F = 2.0; raw.G = 2.0;
  } else if (vs < 3.0) {
    raw.F = 0.5; raw.G = 1.5; raw.K = 2.5;
  } else {
    raw.K = 1.0; raw.M = 3.5;
  }

  const probs = softmax(raw);
  return { classification: topClass(probs), probabilities: probs };
}

function classifyMicrolensing(f: LCFeatures): { classification: string; probabilities: Probs } {
  const raw: Probs = {
    no_event: 0, single_lens: 0, binary_lens: 0, planetary: 0,
  };

  // Microlensing: smooth single brightening → positive skew, no periodicity
  const mlScore = Math.max(0, f.skewness) * 0.5 + f.peak_to_trough * 3;

  if (mlScore < 0.3) {
    raw.no_event = 4.0;
  } else if (mlScore < 1.0 && f.periodicity < 0.2) {
    raw.single_lens = 3.0; raw.binary_lens = 0.5;
  } else if (Math.abs(f.kurtosis) > 1.5) {
    // Complex peak → binary or planetary
    raw.binary_lens = 2.5; raw.planetary = 1.5;
  } else {
    raw.single_lens = 1.5; raw.binary_lens = 1.0; raw.no_event = 0.5;
  }

  const probs = softmax(raw);
  return { classification: topClass(probs), probabilities: probs };
}

function runInference(
  features: LCFeatures,
  modelId: string
): { classification: string; probabilities: Probs } {
  switch (modelId.toUpperCase()) {
    case 'EXOPLANET-001':  return classifyExoplanet(features);
    case 'VSTAR-001':      return classifyVariableStar(features);
    case 'FLARE-001':      return classifyFlare(features);
    case 'ASTERO-001':     return classifyAstero(features);
    case 'SUPERNOVA-001':  return classifySupernova(features);
    case 'GALAXY-001':     return classifyGalaxy(features);
    case 'SPECTYPE-001':   return classifySpectralType(features);
    case 'MICROLENS-001':  return classifyMicrolensing(features);
    default:               return classifyExoplanet(features);
  }
}

// ─── Route handler ────────────────────────────────────────────────────────────

export async function POST(request: NextRequest) {
  const t0 = Date.now();

  try {
    // Validate API key (or fall through for web-UI session requests)
    let authResult: Awaited<ReturnType<typeof validateApiKey>>;
    try {
      authResult = await validateApiKey(request);
    } catch (errResp) {
      return errResp as NextResponse; // invalid/rate-limited key
    }
    void authResult; // auth context available if needed later

    const formData = await request.formData();
    const file = formData.get('file') as File | null;
    const modelId = (formData.get('model_id') as string | null) || 'EXOPLANET-001';

    if (!file) {
      return NextResponse.json({ detail: 'No file provided' }, { status: 400 });
    }

    if (file.size > 50 * 1024 * 1024) { // 50 MB cap
      return NextResponse.json({ detail: 'File too large (max 50 MB)' }, { status: 413 });
    }

    const buffer = await file.arrayBuffer();

    // Detect FITS by magic bytes ("SIMPLE  " at offset 0)
    const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 8));
    let time: number[], flux: number[];

    if (magic.startsWith('SIMPLE')) {
      // FITS binary table
      const parsed = parseFITSLightCurve(buffer);
      time = parsed.time;
      flux = parsed.flux;
    } else {
      // Try as whitespace/comma-delimited ASCII (time, flux, [flux_err])
      const text = new TextDecoder().decode(buffer);
      time = []; flux = [];
      for (const line of text.split('\n')) {
        const l = line.trim();
        if (!l || l.startsWith('#')) continue;
        const parts = l.split(/[,\s\t]+/);
        if (parts.length >= 2) {
          const t = parseFloat(parts[0]);
          const f = parseFloat(parts[1]);
          if (isFinite(t) && isFinite(f)) { time.push(t); flux.push(f); }
        }
      }
    }

    if (time.length < 10) {
      return NextResponse.json(
        { detail: 'Could not parse a light curve from the uploaded file. Please upload a TESS FITS file or a two-column ASCII time-series.' },
        { status: 422 }
      );
    }

    const features = extractFeatures(time, flux);
    const { classification, probabilities } = runInference(features, modelId);
    const confidence = Math.max(...Object.values(probabilities));

    const result: InferenceResult = {
      classification,
      confidence,
      probabilities,
      inference_time_ms: Date.now() - t0,
      memory_used_kb: Math.round(buffer.byteLength / 1024),
    };

    const headers: Record<string, string> = {};
    if (authResult.type === 'api') {
      headers['X-Request-ID'] = authResult.keyId.slice(0, 8);
    }

    return NextResponse.json(result, { headers });
  } catch (err) {
    console.error('[TinyML] analyze error:', err);
    return NextResponse.json(
      { detail: err instanceof Error ? err.message : 'Analysis failed' },
      { status: 500 }
    );
  }
}
