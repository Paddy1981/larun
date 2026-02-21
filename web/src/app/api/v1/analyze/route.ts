import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth';
import { createAnalysisInDB, updateAnalysisInDB } from '@/lib/analysis-db';

export const maxDuration = 60;

export async function POST(request: NextRequest) {
  try {
    const session = await getServerSession(authOptions);

    const body = await request.json();
    const { tic_id } = body;

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

    // Quota check for authenticated users
    if (session?.user) {
      const analysesThisMonth = session.user.analysesThisMonth || 0;
      const analysesLimit = session.user.analysesLimit || 5;
      if (analysesLimit !== -1 && analysesThisMonth >= analysesLimit) {
        return NextResponse.json(
          { error: { code: 'usage_limit_exceeded', message: `Monthly limit reached (${analysesLimit}). Upgrade to continue.` } },
          { status: 429 }
        );
      }
    }

    const userEmail = session?.user?.email ?? 'anonymous@larun.space';
    const nextAuthId = session?.user?.id ?? 'anon';

    // Create record in Supabase (persists across serverless instances)
    const analysis = await createAnalysisInDB(nextAuthId, userEmail, normalizedTicId);

    // Run detection inline (synchronous, within 60s Vercel limit)
    const { fetchLightCurve, fetchTICInfo, KNOWN_TARGETS } = await import('@/lib/mast-service');
    const { runTransitDetection } = await import('@/lib/transit-detection');

    await updateAnalysisInDB(analysis.id, {
      status: 'processing',
      started_at: new Date().toISOString(),
    });

    // Use known period as hint for confirmed/candidate targets so BLS can find
    // long-period planets (e.g. TOI-700 d at 37.4d) with only 1 sector of data.
    const hintPeriod = KNOWN_TARGETS[normalizedTicId]?.period;

    let result = null;
    let finalStatus: 'completed' | 'failed' = 'completed';
    let errorMessage: string | undefined;

    try {
      const lightCurve = await fetchLightCurve(normalizedTicId);
      const ticInfo = await fetchTICInfo(normalizedTicId);
      result = await runTransitDetection(lightCurve, ticInfo, hintPeriod);

      await updateAnalysisInDB(analysis.id, {
        status: 'completed',
        completed_at: new Date().toISOString(),
        result: {
          ...result,
          vetting: result.vetting ?? undefined,
          // Store tic_info so results page can show sky image + stellar properties
          tic_info: ticInfo ?? undefined,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        } as any,
      });
    } catch (detectionError) {
      finalStatus = 'failed';
      errorMessage = detectionError instanceof Error ? detectionError.message : 'Detection failed';
      await updateAnalysisInDB(analysis.id, {
        status: 'failed',
        completed_at: new Date().toISOString(),
        error: errorMessage,
      });
    }

    return NextResponse.json({
      analysis_id: analysis.id,
      tic_id: normalizedTicId,
      status: finalStatus,
      result,
      error: errorMessage ?? null,
    });
  } catch (error) {
    console.error('Error submitting analysis:', error);
    return NextResponse.json(
      { error: { code: 'internal_error', message: 'Failed to submit analysis' } },
      { status: 500 }
    );
  }
}
