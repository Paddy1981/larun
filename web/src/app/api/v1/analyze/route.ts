import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
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
    // Auth is optional — authenticated users get quota tracking
    const session = await getServerSession(authOptions);

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

    // Check usage limits for authenticated users only
    if (session?.user) {
      const analysesThisMonth = session.user.analysesThisMonth || 0;
      const analysesLimit = session.user.analysesLimit || 5;
      if (analysesLimit !== -1 && analysesThisMonth >= analysesLimit) {
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

    const userId = session?.user?.id ?? `anon-${normalizedTicId}`;
    const analysis = createAnalysis(userId, normalizedTicId);

    await processAnalysis(analysis.id);

    const completed = getAnalysis(analysis.id);

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
