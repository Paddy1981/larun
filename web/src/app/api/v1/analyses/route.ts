import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { createClient } from '@supabase/supabase-js';
import { authOptions } from '@/lib/auth';

/**
 * GET /api/v1/analyses
 *
 * Returns the authenticated user's analysis history from Supabase.
 * Supports both Supabase-auth users (by user_id) and NextAuth users (by email).
 */
export async function GET(request: NextRequest) {
  try {
    const session = await getServerSession(authOptions);
    if (!session?.user) {
      return NextResponse.json(
        { error: { code: 'unauthorized', message: 'Authentication required' } },
        { status: 401 }
      );
    }

    const sb = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_SERVICE_KEY || ''
    );

    const email = session.user.email;

    // Look up Supabase user_id by email (may not exist for pure NextAuth users)
    const { data: userRow } = await sb
      .from('users')
      .select('id')
      .eq('email', email)
      .maybeSingle();
    const supabaseUserId = userRow?.id ?? null;

    // Query analyses by user_email OR user_id — whichever is set
    const { data: rows, error: dbErr } = await sb
      .from('analyses')
      .select('id, created_at, confidence, result, classification')
      .eq('classification', 'tic_analysis')
      .or(
        [
          email         ? `user_email.eq.${email}` : null,
          supabaseUserId ? `user_id.eq.${supabaseUserId}` : null,
        ].filter(Boolean).join(',')
      )
      .order('created_at', { ascending: false })
      .limit(50);

    if (dbErr) throw dbErr;

    const analyses = (rows ?? []).map(row => {
      const r = row.result ?? {};
      return {
        id:         row.id,
        tic_id:     r.tic_id ?? '',
        status:     r.status ?? 'completed',
        created_at: row.created_at,
        result: r.status === 'completed' ? {
          detection:     r.detection,
          confidence:    r.confidence ?? row.confidence,
          period_days:   r.period_days ?? null,
          depth_ppm:     r.depth_ppm ?? null,
          duration_hours: r.duration_hours ?? null,
          vetting:       r.vetting ? { disposition: r.vetting.disposition } : undefined,
        } : undefined,
      };
    });

    return NextResponse.json({ analyses, total: analyses.length });
  } catch (error) {
    console.error('Error listing analyses:', error);
    return NextResponse.json(
      { error: { code: 'internal_error', message: 'Failed to list analyses' } },
      { status: 500 }
    );
  }
}
