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
 */
export async function POST(request: NextRequest) {
  try {
    const sb = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_SERVICE_KEY || ''
    );

    const body = await request.json();
    const { tic_id, user_id: supabaseUserId } = body;

    // ── Auth: Supabase user_id (cloud platform) ──────────────────────────────
    let resolvedUserId: string;
    let quotaCheckFn: () => Promise<{ allowed: boolean; limit: number; used: number }>;
    let quotaIncrFn: () => Promise<void>;

    if (supabaseUserId) {
      // Verify the user exists in Supabase
      const { data: userRow } = await sb
        .from('users')
        .select('analyses_this_month, analyses_limit')
        .eq('id', supabaseUserId)
        .maybeSingle();

      resolvedUserId = supabaseUserId;
      const limit: number = userRow?.analyses_limit ?? 5;
      const used: number = userRow?.analyses_this_month ?? 0;

      quotaCheckFn = async () => ({ allowed: limit === -1 || used < limit, limit, used });
      quotaIncrFn = async () => {
        await sb
          .from('users')
          .update({ analyses_this_month: used + 1 })
          .eq('id', supabaseUserId);
        await sb.from('analyses').insert({
          user_id: supabaseUserId,
          model_id: 'EXOPLANET-BLS',
          classification: 'transit_detection',
          confidence: 0,
          inference_time_ms: 0,
          result: {},
        });
      };
    } else {
      // ── Auth: NextAuth session (legacy) ──────────────────────────────────
      const session = await getServerSession(authOptions);
      if (!session?.user?.email) {
        return NextResponse.json(
          { error: { code: 'unauthorized', message: 'Authentication required' } },
          { status: 401 }
        );
      }

      const email = session.user.email;
      const currentMonth = new Date().toISOString().slice(0, 7);

      const { data: userData } = await sb
        .from('users')
        .select('analyses_limit')
        .eq('email', email)
        .maybeSingle();
      const analysesLimit = userData?.analyses_limit ?? 5;

      resolvedUserId = session.user.id;
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
        { error: { code: 'invalid_tic_id', message: 'Invalid TIC ID format. Please enter a numeric TIC ID.' } },
        { status: 400 }
      );
    }

    // ── Quota check ───────────────────────────────────────────────────────────
    const { allowed, limit } = await quotaCheckFn();
    if (!allowed) {
      return NextResponse.json(
        { error: { code: 'usage_limit_exceeded', message: `Monthly analysis limit reached (${limit}). Upgrade to analyze more targets.` } },
        { status: 429 }
      );
    }

    // ── Run detection ─────────────────────────────────────────────────────────
    const analysis = createAnalysis(resolvedUserId, normalizedTicId);
    await processAnalysis(analysis.id);
    const completed = getAnalysis(analysis.id);

    // Increment quota (non-fatal)
    try { await quotaIncrFn(); } catch (e) { console.error('[analyze] quota increment failed:', e); }

    return NextResponse.json({
      analysis_id: analysis.id,
      tic_id: normalizedTicId,
      status: completed?.status ?? 'failed',
      result: completed?.result ?? null,
      error: completed?.error ?? null,
    });
  } catch (error) {
    console.error('Error submitting analysis:', error);
    return NextResponse.json(
      { error: { code: 'internal_error', message: 'Failed to submit analysis' } },
      { status: 500 }
    );
  }
}
