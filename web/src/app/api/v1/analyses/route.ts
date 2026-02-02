import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth';
import { listAnalyses, type AnalysisStatus } from '@/lib/analysis-store';

/**
 * GET /api/v1/analyses
 *
 * List user's analyses with pagination.
 * Requires authentication.
 *
 * Query parameters:
 * - page: Page number (default 1)
 * - per_page: Items per page (default 10, max 100)
 * - status: Filter by status (pending, processing, completed, failed)
 */
export async function GET(request: NextRequest) {
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

    // Parse query parameters
    const searchParams = request.nextUrl.searchParams;
    const page = Math.max(1, parseInt(searchParams.get('page') || '1', 10));
    const perPage = Math.min(100, Math.max(1, parseInt(searchParams.get('per_page') || '10', 10)));
    const statusFilter = searchParams.get('status') as AnalysisStatus | null;

    // Validate status filter
    const validStatuses = ['pending', 'processing', 'completed', 'failed'];
    if (statusFilter && !validStatuses.includes(statusFilter)) {
      return NextResponse.json(
        {
          error: {
            code: 'invalid_status',
            message: `Invalid status filter. Valid values: ${validStatuses.join(', ')}`,
          },
        },
        { status: 400 }
      );
    }

    // Get analyses
    const { analyses, total } = listAnalyses(session.user.id, {
      status: statusFilter || undefined,
      limit: perPage,
      offset: (page - 1) * perPage,
    });

    // Build response
    const formattedAnalyses = analyses.map(analysis => {
      const item: Record<string, unknown> = {
        id: analysis.id,
        tic_id: analysis.tic_id,
        status: analysis.status,
        created_at: analysis.created_at.toISOString(),
      };

      if (analysis.completed_at) {
        item.completed_at = analysis.completed_at.toISOString();
      }

      if (analysis.status === 'completed' && analysis.result) {
        item.result = {
          detection: analysis.result.detection,
          confidence: analysis.result.confidence,
          period_days: analysis.result.period_days,
          depth_ppm: analysis.result.depth_ppm,
          duration_hours: analysis.result.duration_hours,
        };
      }

      if (analysis.status === 'failed') {
        item.error = analysis.error;
      }

      return item;
    });

    return NextResponse.json({
      analyses: formattedAnalyses,
      total,
      page,
      per_page: perPage,
      total_pages: Math.ceil(total / perPage),
    });
  } catch (error) {
    console.error('Error listing analyses:', error);
    return NextResponse.json(
      {
        error: {
          code: 'internal_error',
          message: 'Failed to list analyses',
        },
      },
      { status: 500 }
    );
  }
}
