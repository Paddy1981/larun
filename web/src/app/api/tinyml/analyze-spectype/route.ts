import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';

export const maxDuration = 10;

const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL || '';
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || '';

// ─── Spectral type colour index tables ───────────────────────────────────────
// Reference values are mid-points of MK spectral classes on the main sequence.
// Sources: Allen's Astrophysical Quantities (4th ed.), Gaia EDR3, 2MASS calibrations.

type SpType = 'O' | 'B' | 'A' | 'F' | 'G' | 'K' | 'M';
const SPTYPES: SpType[] = ['O', 'B', 'A', 'F', 'G', 'K', 'M'];

interface IndexTable {
  centers: Record<SpType, number>;
  sigma: number;
}

const BV: IndexTable = {
  centers: { O: -0.30, B: -0.12, A: 0.11, F: 0.34, G: 0.58, K: 0.93, M: 1.40 },
  sigma: 0.12,
};
const VR: IndexTable = {
  centers: { O: -0.08, B: -0.02, A: 0.10, F: 0.24, G: 0.39, K: 0.55, M: 0.80 },
  sigma: 0.08,
};
const BPRP: IndexTable = {
  centers: { O: -0.22, B: -0.05, A: 0.28, F: 0.68, G: 1.10, K: 1.78, M: 2.80 },
  sigma: 0.25,
};
const JH: IndexTable = {
  centers: { O: 0.03, B: 0.06, A: 0.16, F: 0.26, G: 0.36, K: 0.54, M: 0.70 },
  sigma: 0.08,
};
const HK: IndexTable = {
  centers: { O: -0.02, B: 0.02, A: 0.07, F: 0.10, G: 0.14, K: 0.21, M: 0.32 },
  sigma: 0.06,
};

// ─── Bayesian colour-index classification ────────────────────────────────────

function gaussianLikelihoods(value: number, table: IndexTable): Record<SpType, number> {
  const out = {} as Record<SpType, number>;
  for (const t of SPTYPES) {
    const z = (value - table.centers[t]) / table.sigma;
    out[t] = Math.exp(-0.5 * z * z);
  }
  return out;
}

function normalise(probs: Record<SpType, number>): Record<SpType, number> {
  const sum = SPTYPES.reduce((s, t) => s + probs[t], 0);
  if (sum === 0) return Object.fromEntries(SPTYPES.map(t => [t, 1 / 7])) as Record<SpType, number>;
  return Object.fromEntries(SPTYPES.map(t => [t, probs[t] / sum])) as Record<SpType, number>;
}

function classifyFromIndices(
  bv?: number | null,
  vr?: number | null,
  bp_rp?: number | null,
  jh?: number | null,
  hk?: number | null,
): { classification: string; confidence: number; probabilities: Record<string, number> } {
  // Start with flat prior, multiply Gaussian likelihoods for each provided index
  const log: Record<SpType, number> = Object.fromEntries(SPTYPES.map(t => [t, 0])) as Record<SpType, number>;

  if (bv   != null && isFinite(bv))   { const l = gaussianLikelihoods(bv,   BV);   SPTYPES.forEach(t => { log[t] += Math.log(l[t] + 1e-300); }); }
  if (vr   != null && isFinite(vr))   { const l = gaussianLikelihoods(vr,   VR);   SPTYPES.forEach(t => { log[t] += Math.log(l[t] + 1e-300); }); }
  if (bp_rp != null && isFinite(bp_rp)) { const l = gaussianLikelihoods(bp_rp, BPRP); SPTYPES.forEach(t => { log[t] += Math.log(l[t] + 1e-300); }); }
  if (jh   != null && isFinite(jh))   { const l = gaussianLikelihoods(jh,   JH);   SPTYPES.forEach(t => { log[t] += Math.log(l[t] + 1e-300); }); }
  if (hk   != null && isFinite(hk))   { const l = gaussianLikelihoods(hk,   HK);   SPTYPES.forEach(t => { log[t] += Math.log(l[t] + 1e-300); }); }

  // Convert log-space back to probabilities (log-sum-exp trick for numerical stability)
  const maxLog = Math.max(...SPTYPES.map(t => log[t]));
  const raw = Object.fromEntries(SPTYPES.map(t => [t, Math.exp(log[t] - maxLog)])) as Record<SpType, number>;
  const probs = normalise(raw);

  const topEntry = SPTYPES.reduce((best, t) => probs[t] > probs[best] ? t : best, SPTYPES[0]);

  return {
    classification: topEntry,
    confidence: probs[topEntry],
    probabilities: probs as Record<string, number>,
  };
}

// ─── Route handler ────────────────────────────────────────────────────────────

export async function POST(request: NextRequest) {
  const t0 = Date.now();

  try {
    const body = await request.json();
    const { bv, vr, bp_rp, jh, hk, user_id } = body as {
      bv?: number | null;
      vr?: number | null;
      bp_rp?: number | null;
      jh?: number | null;
      hk?: number | null;
      user_id?: string | null;
    };

    // Require at least one finite index
    const provided = [bv, vr, bp_rp, jh, hk].filter(v => v != null && isFinite(Number(v)));
    if (provided.length === 0) {
      return NextResponse.json(
        { detail: 'Provide at least one colour index (B−V, V−R, BP−RP, J−H, or H−K).' },
        { status: 422 }
      );
    }

    // Quota check (same logic as main analyze route)
    let userRow: { analyses_this_month: number; analyses_limit: number } | null = null;
    const userId = user_id ?? null;
    if (userId && SUPABASE_URL && SUPABASE_SERVICE_KEY) {
      const service = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);
      const { data } = await service
        .from('users')
        .select('analyses_this_month, analyses_limit')
        .eq('id', userId)
        .maybeSingle();

      if (data) {
        userRow = data as { analyses_this_month: number; analyses_limit: number };
        const limit = userRow.analyses_limit ?? 5;
        const used = userRow.analyses_this_month ?? 0;
        if (limit !== -1 && used >= limit) {
          return NextResponse.json(
            { detail: `Monthly limit of ${limit} analyses reached. Upgrade your plan to continue.` },
            { status: 429 }
          );
        }
      }
    }

    // Classify
    const { classification, confidence, probabilities } = classifyFromIndices(bv, vr, bp_rp, jh, hk);

    const result = {
      classification,
      confidence,
      probabilities,
      inference_time_ms: Date.now() - t0,
      memory_used_kb: 0,
    };

    // Persist result + update quota (fire-and-forget)
    if (userId && SUPABASE_URL && SUPABASE_SERVICE_KEY) {
      const service = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);
      const capturedRow = userRow;
      void (async () => {
        try {
          await service.from('analyses').insert({
            user_id: userId,
            model_id: 'SPECTYPE-001',
            classification,
            confidence,
            inference_time_ms: result.inference_time_ms,
            result,
          });
          if (capturedRow !== null) {
            await service
              .from('users')
              .update({ analyses_this_month: (capturedRow.analyses_this_month ?? 0) + 1 })
              .eq('id', userId);
          }
        } catch { /* persistence failure doesn't affect the response */ }
      })();
    }

    return NextResponse.json(result);
  } catch (err) {
    console.error('[SPECTYPE] classify error:', err);
    return NextResponse.json({ detail: 'Classification failed' }, { status: 500 });
  }
}
