import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth';

const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL || '';
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || '';

export async function GET(request: NextRequest) {
  let email: string | null = null;

  // 1. Try Supabase Auth token from request header
  const authHeader = request.headers.get('Authorization');
  if (authHeader?.startsWith('Bearer ') && SUPABASE_URL && SUPABASE_SERVICE_KEY) {
    const token = authHeader.replace('Bearer ', '');
    const client = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);
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
  const service = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);
  const { data } = await service
    .from('users')
    .select('subscription_tier, analyses_this_month, analyses_limit')
    .eq('email', email)
    .single();

  if (!data) {
    return NextResponse.json({ subscription_tier: 'free', analyses_this_month: 0, analyses_limit: 5 });
  }

  return NextResponse.json(data);
}
