import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth';

// LemonSqueezy Variant IDs from environment variables
const VARIANT_IDS = {
  monthly: process.env.LEMONSQUEEZY_VARIANT_MONTHLY || '',
  annual: process.env.LEMONSQUEEZY_VARIANT_ANNUAL || '',
};

const LEMONSQUEEZY_API_URL = 'https://api.lemonsqueezy.com/v1';

export async function POST(request: NextRequest) {
  try {
    // Check authentication
    const session = await getServerSession(authOptions);
    if (!session?.user?.id) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      );
    }

    const { plan } = await request.json();

    if (!plan || !['monthly', 'annual'].includes(plan)) {
      return NextResponse.json(
        { error: 'Invalid plan. Must be "monthly" or "annual"' },
        { status: 400 }
      );
    }

    const apiKey = process.env.LEMONSQUEEZY_API_KEY;
    const storeId = process.env.LEMONSQUEEZY_STORE_ID;

    if (!apiKey || !storeId) {
      console.error('LemonSqueezy credentials not configured');
      return NextResponse.json(
        { error: 'Payment system not configured' },
        { status: 500 }
      );
    }

    const variantId = VARIANT_IDS[plan as keyof typeof VARIANT_IDS];

    // Create checkout session via LemonSqueezy API
    const response = await fetch(`${LEMONSQUEEZY_API_URL}/checkouts`, {
      method: 'POST',
      headers: {
        'Accept': 'application/vnd.api+json',
        'Content-Type': 'application/vnd.api+json',
        'Authorization': `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        data: {
          type: 'checkouts',
          attributes: {
            checkout_data: {
              custom: {
                user_id: session.user.id,
              },
            },
            checkout_options: {
              dark: false,
              logo: true,
              embed: false,
              media: true,
              button_color: '#1a73e8',
            },
            product_options: {
              enabled_variants: [parseInt(variantId)],
              redirect_url: 'https://www.larun.space/dashboard?checkout=success',
              receipt_link_url: 'https://www.larun.space/dashboard',
              receipt_button_text: 'Go to Dashboard',
              receipt_thank_you_note: 'Welcome to Larun! Your subscription is now active.',
            },
            preview: false,
          },
          relationships: {
            store: {
              data: {
                type: 'stores',
                id: storeId,
              },
            },
            variant: {
              data: {
                type: 'variants',
                id: variantId,
              },
            },
          },
        },
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('LemonSqueezy API error:', response.status, errorText);
      return NextResponse.json(
        { error: `LemonSqueezy error: ${response.status}`, details: errorText },
        { status: 500 }
      );
    }

    const data = await response.json();
    const checkoutUrl = data.data.attributes.url;

    return NextResponse.json({ url: checkoutUrl });
  } catch (error) {
    console.error('Checkout error:', error);
    return NextResponse.json(
      { error: 'Failed to create checkout session' },
      { status: 500 }
    );
  }
}
