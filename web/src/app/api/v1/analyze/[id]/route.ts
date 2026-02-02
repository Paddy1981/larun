import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth';
import { getAnalysis } from '@/lib/analysis-store';

interface RouteParams {
  params: Promise<{ id: string }>;
}

/**
 * GET /api/v1/analyze/[id]
 *
 * Get the status and results of a submitted analysis.
 * Requires authentication and ownership.
 */
export async function GET(request: NextRequest, { params }: RouteParams) {
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

    const { id: analysisId } = await params;

    // Get analysis (checks ownership via userId)
    const analysis = getAnalysis(analysisId, session.user.id);

    if (!analysis) {
      return NextResponse.json(
        {
          error: {
            code: 'not_found',
            message: 'Analysis not found',
          },
        },
        { status: 404 }
      );
    }

    // Build response
    const response: Record<string, unknown> = {
      id: analysis.id,
      tic_id: analysis.tic_id,
      status: analysis.status,
      created_at: analysis.created_at.toISOString(),
    };

    if (analysis.started_at) {
      response.started_at = analysis.started_at.toISOString();
    }

    if (analysis.completed_at) {
      response.completed_at = analysis.completed_at.toISOString();
    }

    // Add result if completed
    if (analysis.status === 'completed' && analysis.result) {
      response.result = {
        detection: analysis.result.detection,
        confidence: analysis.result.confidence,
        period_days: analysis.result.period_days,
        depth_ppm: analysis.result.depth_ppm,
        duration_hours: analysis.result.duration_hours,
        vetting: analysis.result.vetting
          ? {
              disposition: analysis.result.vetting.disposition,
              odd_even: analysis.result.vetting.odd_even,
              v_shape: analysis.result.vetting.v_shape,
              secondary_eclipse: analysis.result.vetting.secondary_eclipse,
            }
          : undefined,
      };
    }

    // Add error if failed
    if (analysis.status === 'failed') {
      response.error = analysis.error || 'Analysis failed';
    }

    return NextResponse.json(response);
  } catch (error) {
    console.error('Error getting analysis:', error);
    return NextResponse.json(
      {
        error: {
          code: 'internal_error',
          message: 'Failed to get analysis',
        },
      },
      { status: 500 }
    );
  }
}
