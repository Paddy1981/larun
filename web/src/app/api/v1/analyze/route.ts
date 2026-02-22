import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { createClient } from '@supabase/supabase-js';
import { authOptions } from '@/lib/auth';
import { createAnalysis, processAnalysis, getAnalysis } from '@/lib/analysis-store';

// Allow up to 60s for MAST fetch + BLS detection (Vercel Hobby limit)
// Upgrade to Pro for 300s if MAST is slow
export const maxDuration = 60;

/**
 * POST /api/v1/analyze
 *
 * Submit a TIC ID for exoplanet transit detection analysis.
 * Runs detection synchronously and returns the full result.
 * With Fluid Compute (Active CPU pricing), you only pay for BLS compute
 * time (~2-3s), not MAST network wait time (~10-30s).
 */
export async function POST(request: NextRequest) {
  try {
    // Check authentication
    const session = await getServerSession(authOptions);

    if (!session?.user?.email) {
      return NextResponse.json(
        {
          error: {
            code: 'unauthorized',
            message: 'Authentication required',
          },
        },
        { status: 401 }
      );
    }

    // Parse request body
    const body = await request.json();
    const { tic_id } = body;

    // Validate TIC ID
    if (!tic_id || typeof tic_id !== 'string') {
      return NextResponse.json(
        {
          error: {
            code: 'invalid_request',
            message: 'tic_id is required',
          },
        },
        { status: 400 }
      );
    }

    // Normalize and validate TIC ID format
    const normalizedTicId = tic_id.replace(/\D/g, '');
    if (!normalizedTicId || normalizedTicId.length < 1) {
      return NextResponse.json(
        {
          error: {
            code: 'invalid_tic_id',
            message: 'Invalid TIC ID format. Please enter a numeric TIC ID.',
          },
        },
        { status: 400 }
      );
    }

    // Live DB quota enforcement (authoritative — not JWT which may be stale)
    const sb = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_SERVICE_KEY || ''
    );
    const email = session.user.email;
    const currentMonth = new Date().toISOString().slice(0, 7); // 'YYYY-MM'

    // Get per-user limit from users table (default 5 if no row yet)
    const { data: userData } = await sb
      .from('users')
      .select('analyses_limit')
      .eq('email', email)
      .maybeSingle();
    const analysesLimit = userData?.analyses_limit != null ? userData.analyses_limit : 5;

    // Check current month usage (skip if unlimited)
    if (analysesLimit !== -1) {
      const { data: quotaData } = await sb
        .from('monthly_quota')
        .select('analyses_count')
        .eq('user_email', email)
        .eq('month', currentMonth)
        .maybeSingle();
      const used = quotaData?.analyses_count ?? 0;
      if (used >= analysesLimit) {
        return NextResponse.json(
          {
            error: {
              code: 'usage_limit_exceeded',
              message: `Monthly analysis limit reached (${analysesLimit}). Upgrade to analyze more targets.`,
            },
          },
          { status: 429 }
        );
      }
    }

    // Create analysis record and run detection synchronously
    // (background Promise pattern breaks in serverless — function exits before it completes)
    const analysis = createAnalysis(session.user.id, normalizedTicId);

    await processAnalysis(analysis.id);

    const completed = getAnalysis(analysis.id);

    // Atomic usage increment after success (non-fatal if it fails)
    const { error: quotaErr } = await sb.rpc('increment_monthly_quota', { p_email: email, p_month: currentMonth });
    if (quotaErr) console.error('[analyze] Failed to increment monthly quota:', quotaErr);

    return NextResponse.json(
      {
        analysis_id: analysis.id,
        tic_id: normalizedTicId,
        status: completed?.status ?? 'failed',
        result: completed?.result ?? null,
        error: completed?.error ?? null,
      },
      { status: 200 }
    );
  } catch (error) {
    console.error('Error submitting analysis:', error);
    return NextResponse.json(
      {
        error: {
          code: 'internal_error',
          message: 'Failed to submit analysis',
        },
      },
      { status: 500 }
    );
  }
}
