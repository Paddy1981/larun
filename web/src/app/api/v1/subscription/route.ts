import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth';
import { createServerSupabaseClient } from '@/lib/supabase';

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

    const supabase = createServerSupabaseClient();

    // Get user's subscription data
    const { data: user, error: userError } = await supabase
      .from('users')
      .select('id, subscription_status, subscription_plan, analyses_used, analyses_limit, subscription_current_period_end')
      .eq('email', session.user.email)
      .single();

    if (userError && userError.code !== 'PGRST116') {
      console.error('Error fetching user:', userError);
      return NextResponse.json({ error: 'Failed to fetch subscription data' }, { status: 500 });
    }

    // If user doesn't exist, return free tier defaults
    if (!user) {
      return NextResponse.json({
        plan: 'free',
        status: 'active',
        current_period_end: null,
        cancel_at_period_end: false,
        analyses_used: 0,
        analyses_limit: PLAN_LIMITS.free,
      });
    }

    const plan = user.subscription_plan || 'free';
    const limit = user.analyses_limit || PLAN_LIMITS[plan as keyof typeof PLAN_LIMITS] || PLAN_LIMITS.free;

    return NextResponse.json({
      plan: plan,
      status: user.subscription_status || 'active',
      current_period_end: user.subscription_current_period_end,
      cancel_at_period_end: false,
      analyses_used: user.analyses_used || 0,
      analyses_limit: limit,
    });
  } catch (error) {
    console.error('Subscription fetch error:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}
