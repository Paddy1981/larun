import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { createClient } from '@supabase/supabase-js';
import { authOptions } from '@/lib/auth';
import { createAnalysis, processAnalysis, getAnalysis } from '@/lib/analysis-store';

export const maxDuration = 60;

/**
 * POST /api/v1/analyze
 *
 * Submit a TIC ID for exoplanet transit detection analysis.
 * Accepts either:
 *   - Supabase auth: pass user_id in the request body (cloud platform)
 *   - NextAuth session: legacy path (old /analyze page)
 *
 * Saves completed results persistently to Supabase `analyses` table.
 */
export async function POST(request: NextRequest) {
  try {
    const sb = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_SERVICE_KEY || ''
    );

    const body = await request.json();
    const { tic_id, user_id: supabaseUserId } = body;

    // ── Auth resolution ───────────────────────────────────────────────────────
    let resolvedUserId: string;
    let resolvedEmail: string | null = null;
    let quotaCheckFn: () => Promise<{ allowed: boolean; limit: number; used: number }>;
    let quotaIncrFn: () => Promise<void>;

    if (supabaseUserId) {
      // Supabase user_id path (cloud platform)
      const { data: userRow } = await sb
        .from('users')
        .select('email, analyses_this_month, analyses_limit')
        .eq('id', supabaseUserId)
        .maybeSingle();

      resolvedUserId  = supabaseUserId;
      resolvedEmail   = userRow?.email ?? null;
      const limit: number = userRow?.analyses_limit ?? 5;
      const used: number  = userRow?.analyses_this_month ?? 0;

      quotaCheckFn = async () => ({ allowed: limit === -1 || used < limit, limit, used });
      quotaIncrFn  = async () => {
        await sb.from('users').update({ analyses_this_month: used + 1 }).eq('id', supabaseUserId);
      };
    } else {
      // NextAuth session path (legacy)
      const session = await getServerSession(authOptions);
      if (!session?.user?.email) {
        return NextResponse.json(
          { error: { code: 'unauthorized', message: 'Authentication required' } },
          { status: 401 }
        );
      }

      const email        = session.user.email;
      const currentMonth = new Date().toISOString().slice(0, 7);

      const { data: userData } = await sb
        .from('users')
        .select('id, analyses_limit')
        .eq('email', email)
        .maybeSingle();
      const analysesLimit = userData?.analyses_limit ?? 5;

      resolvedUserId = userData?.id ?? session.user.id;
      resolvedEmail  = email;

      quotaCheckFn = async () => {
        if (analysesLimit === -1) return { allowed: true, limit: -1, used: 0 };
        const { data: quotaData } = await sb
          .from('monthly_quota')
          .select('analyses_count')
          .eq('user_email', email)
          .eq('month', currentMonth)
          .maybeSingle();
        const used = quotaData?.analyses_count ?? 0;
        return { allowed: used < analysesLimit, limit: analysesLimit, used };
      };
      quotaIncrFn = async () => {
        await sb.rpc('increment_monthly_quota', { p_email: email, p_month: currentMonth });
      };
    }

    // ── Validate TIC ID ───────────────────────────────────────────────────────
    if (!tic_id || typeof tic_id !== 'string') {
      return NextResponse.json(
        { error: { code: 'invalid_request', message: 'tic_id is required' } },
        { status: 400 }
      );
    }
    const normalizedTicId = tic_id.replace(/\D/g, '');
    if (!normalizedTicId) {
      return NextResponse.json(
        { error: { code: 'invalid_tic_id', message: 'Invalid TIC ID format.' } },
        { status: 400 }
      );
    }

    // ── Quota check ───────────────────────────────────────────────────────────
    const { allowed, limit } = await quotaCheckFn();
    if (!allowed) {
      return NextResponse.json(
        { error: { code: 'usage_limit_exceeded', message: `Monthly analysis limit reached (${limit}).` } },
        { status: 429 }
      );
    }

    // ── Run detection (in-memory pipeline) ────────────────────────────────────
    const analysis = createAnalysis(resolvedUserId, normalizedTicId);
    await processAnalysis(analysis.id);
    const completed = getAnalysis(analysis.id);

    // ── Persist to Supabase ───────────────────────────────────────────────────
    if (completed?.status === 'completed' && completed.result) {
      const r = completed.result;
      try {
        await sb.from('analyses').insert({
          user_id:          supabaseUserId || null,
          user_email:       resolvedEmail,
          model_id:         'EXOPLANET-BLS',
          classification:   'tic_analysis',
          confidence:       Math.max(0, Math.min(1, r.confidence ?? 0)),
          inference_time_ms: (r.processing_time_seconds ?? 0) * 1000,
          result: {
            tic_id:             normalizedTicId,
            status:             'completed',
            detection:          r.detection,
            confidence:         r.confidence,
            period_days:        r.period_days,
            depth_ppm:          r.depth_ppm,
            duration_hours:     r.duration_hours,
            epoch_btjd:         r.epoch_btjd,
            snr:                r.snr,
            sectors_used:       r.sectors_used,
            processing_time_seconds: r.processing_time_seconds,
            vetting:            r.vetting,
            tic_info:           r.tic_info,
            folded_curve:       r.folded_curve,
          },
        });
      } catch (dbErr) {
        // Non-fatal — analysis still returns successfully to the client
        console.error('[analyze] Supabase persist error:', dbErr);
      }
    }

    // Increment quota (non-fatal)
    try { await quotaIncrFn(); } catch (e) { console.error('[analyze] quota increment failed:', e); }

    return NextResponse.json({
      analysis_id: analysis.id,
      tic_id:      normalizedTicId,
      status:      completed?.status ?? 'failed',
      result:      completed?.result ?? null,
      error:       completed?.error ?? null,
    });
  } catch (error) {
    console.error('Error submitting analysis:', error);
    return NextResponse.json(
      { error: { code: 'internal_error', message: 'Failed to submit analysis' } },
      { status: 500 }
    );
  }
}
