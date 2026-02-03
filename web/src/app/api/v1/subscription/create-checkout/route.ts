import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth';
import { createCheckout, PLANS, PlanType } from '@/lib/lemonsqueezy';

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

    const body = await request.json();
    const { plan } = body as { plan: PlanType };

    // Validate plan
    if (!plan || !['monthly', 'annual'].includes(plan)) {
      return NextResponse.json(
        { error: 'Invalid plan. Must be "monthly" or "annual"' },
        { status: 400 }
      );
    }

    const selectedPlan = PLANS[plan];
    if (!('variantId' in selectedPlan)) {
      return NextResponse.json(
        { error: 'Plan does not have a price configured' },
        { status: 400 }
      );
    }

    // Create checkout session with LemonSqueezy
    const result = await createCheckout({
      variantId: selectedPlan.variantId,
      email: session.user.email,
      name: session.user.name || undefined,
      customData: {
        user_email: session.user.email,
        plan: plan,
      },
    });

    if ('error' in result) {
      return NextResponse.json(
        { error: result.error },
        { status: 500 }
      );
    }

    return NextResponse.json({
      url: result.checkoutUrl,
    });
  } catch (error) {
    console.error('Checkout error:', error);

    if (error instanceof Error) {
      return NextResponse.json(
        { error: error.message },
        { status: 500 }
      );
    }

    return NextResponse.json(
      { error: 'Failed to create checkout session' },
      { status: 500 }
    );
  }
}
