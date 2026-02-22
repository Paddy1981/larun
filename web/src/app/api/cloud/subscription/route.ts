import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth';

const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL || '';

export async function GET(request: NextRequest) {
  let email: string | null = null;

  // 1. Try Supabase Auth token from request header
  const authHeader = request.headers.get('Authorization');
  if (authHeader?.startsWith('Bearer ') && SUPABASE_URL) {
    const token = authHeader.replace('Bearer ', '');
    const svcKey = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_SERVICE_KEY || '';
    const client = createClient(SUPABASE_URL, svcKey);
    const { data: { user } } = await client.auth.getUser(token);
    if (user?.email) email = user.email;
  }

  // 2. Fall back to NextAuth session
  if (!email) {
    const session = await getServerSession(authOptions);
    if (session?.user?.email) email = session.user.email;
  }

  if (!email) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  // 3. Query with service key (bypasses RLS)
  const svcKey = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_SERVICE_KEY || '';
  const service = createClient(SUPABASE_URL, svcKey);
  const currentMonth = new Date().toISOString().slice(0, 7);

  const [userResult, quotaResult] = await Promise.all([
    service
      .from('users')
      .select('subscription_tier, analyses_limit')
      .eq('email', email)
      .maybeSingle(),
    service
      .from('monthly_quota')
      .select('analyses_count')
      .eq('user_email', email)
      .eq('month', currentMonth)
      .maybeSingle(),
  ]);

  if (!userResult.data) {
    return NextResponse.json({
      subscription_tier: 'free',
      analyses_this_month: quotaResult.data?.analyses_count ?? 0,
      analyses_limit: 5,
    });
  }

  return NextResponse.json({
    subscription_tier: userResult.data.subscription_tier || 'free',
    analyses_this_month: quotaResult.data?.analyses_count ?? 0,
    analyses_limit: userResult.data.analyses_limit ?? 5,
  });
}
