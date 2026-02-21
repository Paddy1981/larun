import { NextRequest, NextResponse } from 'next/server';
import { getAnalysisFromDB } from '@/lib/analysis-db';

interface RouteParams {
  params: Promise<{ id: string }>;
}

/**
 * GET /api/v1/analyze/[id]
 *
 * Get the status and results of a submitted analysis from Supabase.
 * No auth required — analysis IDs are UUIDs (not guessable).
 */
export async function GET(request: NextRequest, { params }: RouteParams) {
  try {
    const { id: analysisId } = await params;

    // Fetch from Supabase by ID (no userId filter — UUIDs are not guessable)
    const analysis = await getAnalysisFromDB(analysisId);

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
      created_at: analysis.created_at,
    };

    if (analysis.completed_at) {
      response.completed_at = analysis.completed_at;
    }

    // Add result if completed
    if (analysis.status === 'completed' && analysis.result) {
      response.result = {
        detection: analysis.result.detection,
        confidence: analysis.result.confidence,
        period_days: analysis.result.period_days,
        depth_ppm: analysis.result.depth_ppm,
        duration_hours: analysis.result.duration_hours,
        epoch_btjd: analysis.result.epoch_btjd,
        snr: analysis.result.snr,
        sectors_used: analysis.result.sectors_used,
        processing_time_seconds: analysis.result.processing_time_seconds,
        vetting: analysis.result.vetting
          ? {
              disposition: analysis.result.vetting.disposition,
              confidence: analysis.result.vetting.confidence,
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
