import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { createClient } from '@supabase/supabase-js';
import { authOptions } from '@/lib/auth';

const PLAN_LIMITS = {
  free: 5,
  monthly: 50,
  annual: -1, // unlimited
};

export async function GET(request: NextRequest) {
  try {
    const session = await getServerSession(authOptions);

    if (!session?.user?.email) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const supabase = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_SERVICE_KEY || ''
    );
    const email = session.user.email;
    const currentMonth = new Date().toISOString().slice(0, 7);

    const [userResult, quotaResult] = await Promise.all([
      supabase
        .from('users')
        .select('id, subscription_tier, subscription_status, analyses_limit, subscription_current_period_end')
        .eq('email', email)
        .maybeSingle(),
      supabase
        .from('monthly_quota')
        .select('analyses_count')
        .eq('user_email', email)
        .eq('month', currentMonth)
        .maybeSingle(),
    ]);

    if (userResult.error) {
      console.error('Error fetching user:', userResult.error);
      return NextResponse.json({ error: 'Failed to fetch subscription data' }, { status: 500 });
    }

    const user = userResult.data;
    const used = quotaResult.data?.analyses_count ?? 0;

    // If user doesn't exist, return free tier defaults
    if (!user) {
      return NextResponse.json({
        plan: 'free',
        status: 'active',
        current_period_end: null,
        cancel_at_period_end: false,
        analyses_used: used,
        analyses_limit: PLAN_LIMITS.free,
      });
    }

    const plan = user.subscription_tier || 'free';
    const limit = user.analyses_limit != null
      ? user.analyses_limit
      : (PLAN_LIMITS[plan as keyof typeof PLAN_LIMITS] ?? PLAN_LIMITS.free);

    return NextResponse.json({
      plan,
      status: user.subscription_status || 'active',
      current_period_end: user.subscription_current_period_end,
      cancel_at_period_end: false,
      analyses_used: used,
      analyses_limit: limit,
    });
  } catch (error) {
    console.error('Subscription fetch error:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}
