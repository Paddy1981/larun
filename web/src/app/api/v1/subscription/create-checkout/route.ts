import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth';
import { stripe, PLANS, PlanType } from '@/lib/stripe';

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
    if (!('stripePriceId' in selectedPlan)) {
      return NextResponse.json(
        { error: 'Plan does not have a price configured' },
        { status: 400 }
      );
    }

    // Check if price ID is configured
    if (selectedPlan.stripePriceId.includes('placeholder')) {
      return NextResponse.json(
        { error: 'Stripe price ID not configured. Please set STRIPE_PRICE_MONTHLY and STRIPE_PRICE_ANNUAL environment variables.' },
        { status: 500 }
      );
    }

    // Get or create Stripe customer
    const customers = await stripe.customers.list({
      email: session.user.email,
      limit: 1,
    });

    let customerId: string;
    if (customers.data.length > 0) {
      customerId = customers.data[0].id;
    } else {
      const customer = await stripe.customers.create({
        email: session.user.email,
        name: session.user.name || undefined,
        metadata: {
          userId: session.user.email,
        },
      });
      customerId = customer.id;
    }

    // Create checkout session
    const checkoutSession = await stripe.checkout.sessions.create({
      customer: customerId,
      mode: 'subscription',
      payment_method_types: ['card'],
      line_items: [
        {
          price: selectedPlan.stripePriceId,
          quantity: 1,
        },
      ],
      success_url: `${process.env.NEXTAUTH_URL}/settings/subscription?success=true&session_id={CHECKOUT_SESSION_ID}`,
      cancel_url: `${process.env.NEXTAUTH_URL}/settings/subscription?canceled=true`,
      metadata: {
        userId: session.user.email,
        plan: plan,
      },
      subscription_data: {
        metadata: {
          userId: session.user.email,
          plan: plan,
        },
      },
    });

    return NextResponse.json({
      url: checkoutSession.url,
      sessionId: checkoutSession.id,
    });
  } catch (error) {
    console.error('Stripe checkout error:', error);

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
