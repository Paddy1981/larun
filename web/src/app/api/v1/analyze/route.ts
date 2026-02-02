import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth';
import { createAnalysis, processAnalysis } from '@/lib/analysis-store';

/**
 * POST /api/v1/analyze
 *
 * Submit a TIC ID for exoplanet transit detection analysis.
 * Requires authentication.
 */
export async function POST(request: NextRequest) {
  try {
    // Check authentication
    const session = await getServerSession(authOptions);

    if (!session?.user) {
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

    // Check usage limits (for free tier)
    const analysesThisMonth = session.user.analysesThisMonth || 0;
    const analysesLimit = session.user.analysesLimit || 3;

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

    // Create analysis record
    const analysis = createAnalysis(session.user.id, normalizedTicId);

    // Start background processing (non-blocking)
    // In production, this would use a job queue like Redis/BullMQ
    processAnalysis(analysis.id).catch(err => {
      console.error(`Background analysis failed for ${analysis.id}:`, err);
    });

    // Return response immediately
    return NextResponse.json(
      {
        analysis_id: analysis.id,
        status: 'pending',
        message: 'Analysis queued. Poll GET /api/v1/analyze/{id} for results.',
      },
      { status: 202 }
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
