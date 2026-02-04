import { NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth';

const LEMONSQUEEZY_API_URL = 'https://api.lemonsqueezy.com/v1';

export async function POST() {
  try {
    // Check authentication
    const session = await getServerSession(authOptions);
    if (!session?.user?.id) {
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

    // TODO: Get the customer's LemonSqueezy customer ID from your database
    // For now, we'll return the general customer portal URL
    // In production, you should:
    // 1. Store lemon_squeezy_customer_id when subscription is created
    // 2. Fetch customer's portal URL using their customer ID

    // Example of how to get customer portal URL when you have the customer ID:
    // const customerId = await getCustomerIdFromDatabase(session.user.id);
    // const response = await fetch(`${LEMONSQUEEZY_API_URL}/customers/${customerId}`, {
    //   headers: {
    //     'Accept': 'application/vnd.api+json',
    //     'Authorization': `Bearer ${apiKey}`,
    //   },
    // });
    // const data = await response.json();
    // const portalUrl = data.data.attributes.urls.customer_portal;

    // For now, redirect to LemonSqueezy's general billing page
    // Users can manage their subscriptions there
    return NextResponse.json({
      url: 'https://app.lemonsqueezy.com/my-orders',
      message: 'Customer portal - users can manage subscriptions here',
    });
  } catch (error) {
    console.error('Portal error:', error);
    return NextResponse.json(
      { error: 'Failed to get billing portal' },
      { status: 500 }
    );
  }
}
