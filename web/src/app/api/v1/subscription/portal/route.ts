import { NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { createClient } from '@supabase/supabase-js';
import { authOptions } from '@/lib/auth';

const LEMONSQUEEZY_API_URL = 'https://api.lemonsqueezy.com/v1';

export async function POST() {
  try {
    // Check authentication
    const session = await getServerSession(authOptions);
    if (!session?.user?.email) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      );
    }

    const apiKey = process.env.LEMONSQUEEZY_API_KEY;

    if (!apiKey) {
      console.error('LEMONSQUEEZY_API_KEY not configured');
      return NextResponse.json(
        { error: 'Payment system not configured' },
        { status: 500 }
      );
    }

    // Fetch customer ID from users table
    const sb = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_SERVICE_KEY || ''
    );
    const { data: user } = await sb
      .from('users')
      .select('lemon_squeezy_customer_id')
      .eq('email', session.user.email)
      .maybeSingle();
    const customerId = user?.lemon_squeezy_customer_id;

    if (customerId) {
      const res = await fetch(`${LEMONSQUEEZY_API_URL}/customers/${customerId}`, {
        headers: {
          'Accept': 'application/vnd.api+json',
          'Authorization': `Bearer ${apiKey}`,
        },
      });
      if (res.ok) {
        const data = await res.json();
        const portalUrl = data.data?.attributes?.urls?.customer_portal;
        if (portalUrl) {
          return NextResponse.json({ url: portalUrl });
        }
      }
    }

    // Fall back to generic URL if no customer ID or API call fails
    return NextResponse.json({
      url: 'https://app.lemonsqueezy.com/my-orders',
    });
  } catch (error) {
    console.error('Portal error:', error);
    return NextResponse.json(
      { error: 'Failed to get billing portal' },
      { status: 500 }
    );
  }
}
