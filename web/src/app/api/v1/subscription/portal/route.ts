import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth';
import { lemonSqueezyApi, STORE_ID } from '@/lib/lemonsqueezy';

export async function POST(request: NextRequest) {
  try {
    // Check authentication
    const session = await getServerSession(authOptions);
    if (!session?.user?.email) {
      return NextResponse.json(
        { error: 'Authentication required' },
        { status: 401 }
      );
    }

    // Find customer by email
    const customersResponse = await lemonSqueezyApi(
      `/customers?filter[store_id]=${STORE_ID}&filter[email]=${encodeURIComponent(session.user.email)}`
    );

    if (!customersResponse.ok) {
      const errorData = await customersResponse.json();
      console.error('LemonSqueezy customers error:', errorData);
      return NextResponse.json(
        { error: 'Failed to find customer record' },
        { status: 500 }
      );
    }

    const customersData = await customersResponse.json();

    if (!customersData.data || customersData.data.length === 0) {
      return NextResponse.json(
        { error: 'No billing account found. Please subscribe to a plan first.' },
        { status: 404 }
      );
    }

    const customer = customersData.data[0];
    const portalUrl = customer.attributes.urls?.customer_portal;

    if (!portalUrl) {
      return NextResponse.json(
        { error: 'Customer portal not available' },
        { status: 404 }
      );
    }

    return NextResponse.json({
      url: portalUrl,
    });
  } catch (error) {
    console.error('Portal error:', error);

    if (error instanceof Error) {
      return NextResponse.json(
        { error: error.message },
        { status: 500 }
      );
    }

    return NextResponse.json(
      { error: 'Failed to access billing portal' },
      { status: 500 }
    );
  }
}
